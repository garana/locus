#include <filesystem>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/chat/template.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "locus/tok/tokenizer.hpp"
#include "gguf_builder.hpp"

using locus::chat::ChatTemplate;
using locus::chat::Message;
using Family = locus::chat::ChatTemplate::Family;

namespace {

ChatTemplate detect(std::string_view tmpl) {
    GgufBuilder b;
    b.header(0, 1).kv_string("tokenizer.chat_template", tmpl);
    auto g = locus::gguf::GgufFile::parse(b.bytes());
    return ChatTemplate::from_gguf(g);
}

const std::vector<Message> kOneTurn = {{"user", "hi"}};

}  // namespace

TEST_CASE("template family detection", "[chat]") {
    REQUIRE(detect("...<|start_header_id|>...").family() ==
            Family::kLlama3);
    REQUIRE(detect("...<|im_start|>...").family() ==
            Family::kChatML);
    REQUIRE(detect("...[INST]...").family() == Family::kLlama2);
    REQUIRE(detect("...<|user|>...").family() ==
            Family::kZephyr);
    REQUIRE(detect("...\xef\xbd\x9cUser\xef\xbd\x9c...")
                .family() == Family::kDeepSeek);
    REQUIRE(detect("something else").family() == Family::kPlain);

    // DeepSeek-Coder-V2-Lite-Instruct's tokenizer.chat_template.
    const std::string_view v2 =
        R"({% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '

' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '

' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %})";
    REQUIRE(detect(v2).family() == Family::kDeepSeek2);

    GgufBuilder none;
    none.header(0, 0);
    auto g = locus::gguf::GgufFile::parse(none.bytes());
    REQUIRE(ChatTemplate::from_gguf(g).family() ==
            Family::kPlain);
}

TEST_CASE("template rendering", "[chat]") {
    SECTION("llama3") {
        ChatTemplate t(Family::kLlama3);
        REQUIRE(t.apply(kOneTurn) ==
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "hi<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
                "\n\n");
    }
    SECTION("chatml") {
        ChatTemplate t(Family::kChatML);
        REQUIRE(t.apply(kOneTurn) ==
                "<|im_start|>user\nhi<|im_end|>\n"
                "<|im_start|>assistant\n");
    }
    SECTION("zephyr") {
        ChatTemplate t(Family::kZephyr);
        REQUIRE(t.apply(kOneTurn) ==
                "<|user|>\nhi</s>\n<|assistant|>\n");
    }
    SECTION("llama2 folds system into first user turn") {
        ChatTemplate t(Family::kLlama2);
        const std::vector<Message> msgs = {{"system", "sys"},
                                           {"user", "hi"}};
        REQUIRE(t.apply(msgs) ==
                "[INST] <<SYS>>\nsys\n<</SYS>>\n\nhi [/INST]");
    }
    SECTION("plain concatenates content") {
        ChatTemplate t(Family::kPlain);
        const std::vector<Message> msgs = {{"user", "a"},
                                           {"assistant", "b"}};
        REQUIRE(t.apply(msgs) == "a\nb");
    }
    // Expected strings are llama.cpp's llama_chat_apply_template
    // output (deepseek3 / deepseek2) for the same four messages.
    const std::vector<Message> multi = {{"system", "sys"},
                                        {"user", "hi"},
                                        {"assistant", "ok"},
                                        {"user", "again"}};
    SECTION("deepseek (V3 markers)") {
        ChatTemplate t(Family::kDeepSeek);
        REQUIRE(t.apply(multi) ==
                "sys\n\n"
                "<\xef\xbd\x9cUser\xef\xbd\x9c>hi"
                "<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>ok"
                "<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81"
                "sentence\xef\xbd\x9c>"
                "<\xef\xbd\x9cUser\xef\xbd\x9c>again"
                "<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>");
    }
    SECTION("deepseek2") {
        ChatTemplate t(Family::kDeepSeek2);
        REQUIRE(t.apply(multi) ==
                "sys\n\nUser: hi\n\nAssistant: ok"
                "<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81"
                "sentence\xef\xbd\x9c>"
                "User: again\n\nAssistant:");
    }
}

namespace {

/** Greedy chat answer for a single user message. */
std::string chat_answer(const locus::model::TransformerModel& model,
                        const locus::tok::Tokenizer& tok,
                        const ChatTemplate& tmpl,
                        const std::string& user, int max_new) {
    const std::vector<Message> msgs = {{"user", user}};
    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(model.hparams().n_vocab);
    for (auto id : tok.encode(tmpl.apply(msgs), true)) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(id, cache, seq, ws, logits);
    }
    std::vector<locus::tok::TokenId> gen;
    for (int i = 0; i < max_new; ++i) {
        auto next = locus::model::argmax(logits);
        if (next == tok.eos_id()) {
            break;
        }
        gen.push_back(next);
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(next, cache, seq, ws, logits);
    }
    cache.release(seq);
    return tok.decode(gen);
}

}  // namespace

TEST_CASE("llama-3.2 Q4_K_M chat matches llama.cpp on every "
          "backend",
          "[chat][e2e]") {
    const std::string path =
        std::string(LOCUS_SOURCE_DIR) +
        "/tests/models/llama-3.2-1b-q4_k_m.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP("model not present (llama-3.2-1b-q4_k_m.gguf)");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::tokenizer_from_gguf(g);
    auto tmpl = ChatTemplate::from_gguf(g);

    // llama-completion (chat template, --temp 0) answers:
    const std::string want =
        "The capital of France is Paris.";
    for (const auto& b : locus::backend::backends()) {
        if (!b.available || !b.selectable) {
            continue;
        }
        INFO("backend " << b.name);
        model.use_backend(b);
        REQUIRE(chat_answer(model, *tok, tmpl,
                            "The capital of France is", 24) ==
                want);
    }
}

TEST_CASE("llama-3.2 chat matches the llama.cpp answer",
          "[chat][e2e]") {
    const std::string path =
        std::string(LOCUS_SOURCE_DIR) +
        "/tests/models/llama-3.2-1b-q8_0.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP("model not present (llama-3.2-1b-q8_0.gguf)");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::tokenizer_from_gguf(g);
    auto tmpl = ChatTemplate::from_gguf(g);
    REQUIRE(tmpl.family() == Family::kLlama3);

    // llama-completion applies the same template to this model
    // and answers exactly "Paris." at --temp 0.
    const std::vector<Message> msgs = {
        {"user", "The capital of France is"}};
    auto ids = tok->encode(tmpl.apply(msgs), true);

    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(model.hparams().n_vocab);
    for (auto id : ids) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(id, cache, seq, ws, logits);
    }
    std::vector<locus::tok::TokenId> gen;
    for (int i = 0; i < 16; ++i) {
        auto next = locus::model::argmax(logits);
        if (next == tok->eos_id()) {
            break;
        }
        gen.push_back(next);
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(next, cache, seq, ws, logits);
    }
    cache.release(seq);
    REQUIRE(tok->decode(gen) == "Paris.");
}
