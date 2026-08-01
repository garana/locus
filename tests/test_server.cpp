#include <filesystem>
#include <string>
#include <thread>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/grammar.hpp"
#include "locus/model/llama.hpp"
#include "locus/server/server.hpp"
#include "locus/tok/tokenizer.hpp"
#include "httplib.h"
#include "json.hpp"

using nlohmann::json;

namespace {

std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

/** Server bound to an ephemeral port, serving on a thread. */
struct TestServer {
    explicit TestServer(const locus::model::LlamaModel& m,
                        const locus::tok::SpmTokenizer& tok)
        : server(m, tok, {}) {
        port = server.bind_any_port("127.0.0.1");
        REQUIRE(port > 0);
        thread = std::thread([this] {
            server.listen_after_bind();
        });
    }
    ~TestServer() {
        server.stop();
        thread.join();
    }

    locus::server::OpenAiServer server;
    int port = -1;
    std::thread thread;
};

}  // namespace

TEST_CASE("openai endpoints serve completions", "[server][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    TestServer ts(model, tok);
    httplib::Client client("127.0.0.1", ts.port);

    SECTION("health") {
        auto res = client.Get("/health");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        REQUIRE(json::parse(res->body)["status"] == "ok");
    }

    SECTION("model listing") {
        auto res = client.Get("/v1/models");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto body = json::parse(res->body);
        REQUIRE(body["object"] == "list");
        REQUIRE(body["data"].size() == 1);
        const std::string id = body["data"][0]["id"];
        REQUIRE(body["data"][0]["object"] == "model");

        auto one = client.Get("/v1/models/" + id);
        REQUIRE(one);
        REQUIRE(one->status == 200);
        REQUIRE(json::parse(one->body)["id"] == id);

        auto miss = client.Get("/v1/models/nope");
        REQUIRE(miss);
        REQUIRE(miss->status == 404);
    }

    SECTION("metrics count requests and tokens") {
        json req{{"prompt", "Once upon a time"}, {"max_tokens", 4}};
        auto gen = client.Post("/v1/completions", req.dump(),
                               "application/json");
        REQUIRE(gen);
        REQUIRE(gen->status == 200);

        auto res = client.Get("/metrics");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        const std::string& b = res->body;
        REQUIRE(b.find("locus_requests_total") != std::string::npos);
        REQUIRE(b.find("locus_completion_tokens_total") !=
                std::string::npos);
        REQUIRE(b.find("locus_kv_blocks_total") !=
                std::string::npos);
        // At least the request we just issued is counted.
        REQUIRE(b.find("locus_requests_total 0\n") ==
                std::string::npos);
    }

    SECTION("completions, non-streaming") {
        json req{{"prompt", "Once upon a time"},
                 {"max_tokens", 16}};
        auto res = client.Post("/v1/completions", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto body = json::parse(res->body);
        REQUIRE(body["object"] == "text_completion");
        REQUIRE(body["choices"].size() == 1);
        const std::string text = body["choices"][0]["text"];
        REQUIRE(text.find("little girl") != std::string::npos);
        REQUIRE(body["choices"][0]["finish_reason"] == "length");
        REQUIRE(body["usage"]["completion_tokens"] == 16);
        REQUIRE(body["usage"]["total_tokens"] ==
                body["usage"]["prompt_tokens"].get<int>() + 16);
        // No logprobs unless requested.
        REQUIRE(!body["choices"][0].contains("logprobs"));
    }

    SECTION("completions logprobs") {
        json req{{"prompt", "Once upon a time"},
                 {"max_tokens", 5},
                 {"logprobs", 3}};
        auto res = client.Post("/v1/completions", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto lp = json::parse(res->body)["choices"][0]["logprobs"];
        REQUIRE(lp["tokens"].size() == 5);
        REQUIRE(lp["token_logprobs"].size() == 5);
        REQUIRE(lp["top_logprobs"].size() == 5);
        // Each step lists 3 alternatives; all logprobs <= 0.
        REQUIRE(lp["top_logprobs"][0].size() == 3);
        for (const auto& v : lp["token_logprobs"]) {
            REQUIRE(v.get<double>() <= 0.0);
        }
    }

    SECTION("chat logprobs") {
        json req{{"messages",
                  json::array({{{"role", "user"},
                                {"content", "Once upon a time"}}})},
                 {"max_tokens", 4},
                 {"logprobs", true},
                 {"top_logprobs", 2}};
        auto res = client.Post("/v1/chat/completions", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto content =
            json::parse(res->body)["choices"][0]["logprobs"]
                                  ["content"];
        REQUIRE(content.size() == 4);
        REQUIRE(content[0].contains("token"));
        REQUIRE(content[0]["top_logprobs"].size() == 2);
    }

    SECTION("chat completions, non-streaming") {
        json req{{"messages",
                  json::array(
                      {{{"role", "user"},
                        {"content", "Once upon a time"}}})},
                 {"max_tokens", 12}};
        auto res = client.Post("/v1/chat/completions", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto body = json::parse(res->body);
        REQUIRE(body["object"] == "chat.completion");
        REQUIRE(body["choices"][0]["message"]["role"] ==
                "assistant");
        REQUIRE(!body["choices"][0]["message"]["content"]
                     .get<std::string>()
                     .empty());
    }

    SECTION("chat completions with tools is accepted") {
        // The toy model won't emit a valid tool call, so this just
        // confirms the tools plumbing (system injection + response
        // branch) doesn't break a normal request.
        json req{
            {"messages",
             json::array({{{"role", "user"},
                           {"content", "Once upon a time"}}})},
            {"max_tokens", 8},
            {"tools",
             json::array(
                 {{{"type", "function"},
                   {"function",
                    {{"name", "get_weather"},
                     {"description", "weather"},
                     {"parameters", {{"type", "object"}}}}}}})}};
        auto res = client.Post("/v1/chat/completions", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto body = json::parse(res->body);
        REQUIRE(body["choices"][0].contains("message"));
    }

    SECTION("response_format json_object constrains to valid JSON") {
        json req{
            {"messages",
             json::array({{{"role", "user"},
                           {"content", "Give me an object"}}})},
            {"max_tokens", 24},
            {"response_format", {{"type", "json_object"}}}};
        auto res = client.Post("/v1/chat/completions", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        const std::string content = json::parse(
            res->body)["choices"][0]["message"]["content"];
        REQUIRE(!content.empty());
        // Every byte must keep the output a valid JSON prefix.
        locus::model::JsonGrammar g;
        for (unsigned char c : content) {
            INFO("content: " << content);
            REQUIRE(g.feed(c));
        }
    }

    SECTION("anthropic messages, non-streaming") {
        json req{{"system", "You are a storyteller."},
                 {"messages",
                  json::array(
                      {{{"role", "user"},
                        {"content", "Once upon a time"}}})},
                 {"max_tokens", 12}};
        auto res = client.Post("/v1/messages", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto body = json::parse(res->body);
        REQUIRE(body["type"] == "message");
        REQUIRE(body["role"] == "assistant");
        REQUIRE(body["content"][0]["type"] == "text");
        REQUIRE(!body["content"][0]["text"]
                     .get<std::string>()
                     .empty());
        REQUIRE((body["stop_reason"] == "end_turn" ||
                 body["stop_reason"] == "max_tokens"));
        REQUIRE(body["usage"]["input_tokens"].get<int>() > 0);
    }

    SECTION("anthropic messages, content blocks") {
        // Content as an array of text blocks must also work.
        json req{{"messages",
                  json::array({{{"role", "user"},
                                {"content",
                                 json::array({{{"type", "text"},
                                               {"text",
                                                "Once upon"}}})}}})},
                 {"max_tokens", 8}};
        auto res = client.Post("/v1/messages", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        REQUIRE(json::parse(res->body)["type"] == "message");
    }

    SECTION("anthropic messages, streaming SSE") {
        json req{{"messages",
                  json::array({{{"role", "user"},
                                {"content", "Once upon a time"}}})},
                 {"max_tokens", 8},
                 {"stream", true}};
        auto res = client.Post("/v1/messages", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        REQUIRE(res->get_header_value("Content-Type") ==
                "text/event-stream");
        const std::string& sse = res->body;
        REQUIRE(sse.find("event: message_start") == 0);
        REQUIRE(sse.find("event: content_block_start") !=
                std::string::npos);
        REQUIRE(sse.find("event: content_block_delta") !=
                std::string::npos);
        REQUIRE(sse.find("\"type\":\"text_delta\"") !=
                std::string::npos);
        REQUIRE(sse.find("event: message_delta") !=
                std::string::npos);
        REQUIRE(sse.find("event: message_stop") !=
                std::string::npos);
        // Reassemble the streamed text_delta pieces (parse each
        // SSE data line as JSON -- key order is not guaranteed).
        std::string text;
        std::size_t at = 0;
        const std::string dp = "data: ";
        while ((at = sse.find(dp, at)) != std::string::npos) {
            at += dp.size();
            const std::size_t end = sse.find("\n\n", at);
            const auto j = json::parse(sse.substr(at, end - at));
            if (j.value("type", "") == "content_block_delta") {
                text += j["delta"]["text"].get<std::string>();
            }
            at = end;
        }
        REQUIRE(!text.empty());
    }

    SECTION("completions, streaming SSE") {
        json req{{"prompt", "Once upon a time"},
                 {"max_tokens", 8},
                 {"stream", true}};
        std::string sse;
        auto res = client.Post(
            "/v1/completions", req.dump(), "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        REQUIRE(res->get_header_value("Content-Type") ==
                "text/event-stream");
        sse = res->body;
        REQUIRE(sse.find("data: ") == 0);
        REQUIRE(sse.find("data: [DONE]\n\n") != std::string::npos);
        // Concatenating the streamed text chunks reproduces the
        // non-streaming completion.
        std::string text;
        std::size_t at = 0;
        while ((at = sse.find("data: ", at)) !=
               std::string::npos) {
            at += 6;
            const std::size_t end = sse.find("\n\n", at);
            const std::string part = sse.substr(at, end - at);
            if (part == "[DONE]") {
                break;
            }
            text += json::parse(part)["choices"][0]["text"]
                        .get<std::string>();
        }
        REQUIRE(text.find(", there was") != std::string::npos);
    }

    SECTION("bad requests get 400") {
        auto res = client.Post("/v1/completions", "{not json",
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 400);

        json req{{"prompt", 42}};
        res = client.Post("/v1/completions", req.dump(),
                          "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 400);
        REQUIRE(json::parse(res->body).contains("error"));
    }

    SECTION("concurrent clients are served") {
        auto worker = [&](std::string* out) {
            httplib::Client c("127.0.0.1", ts.port);
            c.set_read_timeout(60, 0);
            json req{{"prompt", "Once upon a time"},
                     {"max_tokens", 12}};
            auto res = c.Post("/v1/completions", req.dump(),
                              "application/json");
            if (res && res->status == 200) {
                *out = json::parse(res->body)["choices"][0]
                                 ["text"]
                                     .get<std::string>();
            }
        };
        std::string t1, t2, t3;
        std::thread a(worker, &t1), b(worker, &t2),
            c(worker, &t3);
        a.join();
        b.join();
        c.join();
        REQUIRE(t1 == t2);
        REQUIRE(t2 == t3);
        REQUIRE(!t1.empty());
    }
}

TEST_CASE("embeddings endpoint returns normalized vectors",
          "[server][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    // Embeddings need a CPU/CUDA backend (the Vulkan full-forward
    // does not surface the hidden state); force the scalar backend.
    model.use_backend(*locus::backend::find_backend("scalar"));
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    TestServer ts(model, tok);
    httplib::Client client("127.0.0.1", ts.port);

    SECTION("single input, unit-normalized, right dimension") {
        json req{{"input", "Once upon a time"}};
        auto res = client.Post("/v1/embeddings", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto body = json::parse(res->body);
        REQUIRE(body["object"] == "list");
        auto emb = body["data"][0]["embedding"];
        REQUIRE(emb.size() == model.hparams().n_embd);
        double n = 0.0;
        for (double v : emb) {
            n += v * v;
        }
        REQUIRE(n == Catch::Approx(1.0).margin(1e-4));
    }

    SECTION("batch input and determinism") {
        json req{{"input", json::array({"hello", "hello"})}};
        auto res = client.Post("/v1/embeddings", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto data = json::parse(res->body)["data"];
        REQUIRE(data.size() == 2);
        REQUIRE(data[0]["embedding"] == data[1]["embedding"]);
    }
}
