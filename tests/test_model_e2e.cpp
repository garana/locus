#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/llama.hpp"
#include "locus/tok/tokenizer.hpp"

namespace {

/** Real-model fixture path; see scripts/fetch-test-model.sh. */
std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

std::string llama32_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/llama-3.2-1b-q8_0.gguf";
}

std::string qwen_moe_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/Qwen1.5-MoE-A2.7B.Q4_K_M.gguf";
}

}  // namespace

TEST_CASE("loads a real Llama-family GGUF model", "[e2e]") {
    const std::string path = model_path();
    if (!std::filesystem::exists(path)) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }

    auto g = locus::gguf::GgufFile::open(path);
    REQUIRE(g.get_string("general.architecture") == "llama");
    REQUIRE_FALSE(g.tensors().empty());

    // Every tensor must be fully addressable inside the mapping.
    for (const auto& t : g.tensors()) {
        REQUIRE(g.tensor_data(t).size() == t.nbytes);
    }

    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    REQUIRE(tok.vocab_size() > 0);

    auto ids = tok.encode("Once upon a time", true);
    REQUIRE(ids.size() > 1);
    REQUIRE(ids.front() == tok.bos_id());

    auto text = tok.decode(ids);
    REQUIRE(text.find("Once upon a time") != std::string::npos);
}

namespace {

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) {
        s.pop_back();
    }
    return s;
}

}  // namespace

namespace {

/** Greedy generation returning the full decoded text. */
std::string generate_text(locus::model::LlamaModel& model,
                          const locus::tok::SpmTokenizer& tok,
                          const std::string& prompt, int n_gen) {
    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(model.hparams().n_vocab);
    auto ids = tok.encode(prompt, true);
    for (auto id : ids) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(id, cache, seq, ws, logits);
    }
    for (int i = 0; i < n_gen; ++i) {
        auto next = locus::model::argmax(logits);
        if (next == tok.eos_id()) {
            break;
        }
        ids.push_back(next);
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(next, cache, seq, ws, logits);
    }
    cache.release(seq);
    return tok.decode(ids);
}

/**
 * Teacher-forces `ids` through a cache of `kv_type` and returns the
 * next-token logits after each position (one row per input token).
 */
std::vector<std::vector<float>> teacher_forced_logits(
    locus::model::LlamaModel& model,
    const std::vector<locus::tok::TokenId>& ids,
    locus::kv::KvType kv_type) {
    auto cache = model.make_cache(0, kv_type);
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(model.hparams().n_vocab);
    std::vector<std::vector<float>> out;
    for (auto id : ids) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(id, cache, seq, ws, logits);
        out.push_back(logits);
    }
    cache.release(seq);
    return out;
}

/** Cosine similarity of two equal-length vectors. */
double cosine(const std::vector<float>& a,
              const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

}  // namespace

TEST_CASE("quantized KV cache tracks the F32 forward", "[e2e][kv]") {
    const std::string path = model_path();
    if (!std::filesystem::exists(path)) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    // Quantized KV runs on the CPU attention path; force a CPU
    // backend so forward() does not route to the GPU.
    model.use_backend(*locus::backend::find_backend("scalar"));

    if (model.make_cache().geometry().kv_dim % 32 != 0) {
        SKIP("model kv_dim is not a multiple of the quant block");
    }

    const auto ids = tok.encode("Once upon a time there was a", true);
    const auto f32 = teacher_forced_logits(model, ids,
                                           locus::kv::KvType::kF32);
    const auto q8 = teacher_forced_logits(model, ids,
                                          locus::kv::KvType::kQ8);
    const auto q4 = teacher_forced_logits(model, ids,
                                          locus::kv::KvType::kQ4);

    REQUIRE(q8.size() == f32.size());
    for (std::size_t i = 0; i < f32.size(); ++i) {
        // Q8 KV closely approximates the exact forward; Q4 is coarser
        // (this fixture is a tiny 64-wide toy model, so 4-bit blocks
        // are noticeably lossier) but still strongly correlated, and
        // never better than Q8.
        REQUIRE(cosine(f32[i], q8[i]) > 0.999);
        REQUIRE(cosine(f32[i], q4[i]) > 0.90);
        REQUIRE(cosine(f32[i], q8[i]) >= cosine(f32[i], q4[i]) - 1e-6);
    }
}

// #43 validation slice: quant-KV is a CPU path, but CUDA offloads the
// wk/wv/wq matvecs -- so this reruns the KV-precision comparison with
// the CUDA backend live (GPU matvec in the loop) to prove quant-KV is
// correct on real GPU hardware, and pins the resident-cache footprint
// shrink (the point of quantizing KV). Runtime-gated on a CUDA device.
TEST_CASE("quantized KV cache tracks F32 under the CUDA backend",
          "[e2e][kv][cuda]") {
    if (!locus::backend::cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    const std::string path = model_path();
    if (!std::filesystem::exists(path)) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    model.use_backend(*locus::backend::find_backend("cuda"));

    const auto geom = model.make_cache().geometry();
    if (geom.kv_dim % 32 != 0) {
        SKIP("model kv_dim is not a multiple of the quant block");
    }

    const auto ids = tok.encode("Once upon a time there was a", true);
    const auto f32 = teacher_forced_logits(model, ids,
                                           locus::kv::KvType::kF32);
    const auto q8 = teacher_forced_logits(model, ids,
                                          locus::kv::KvType::kQ8);
    const auto q4 = teacher_forced_logits(model, ids,
                                          locus::kv::KvType::kQ4);
    REQUIRE(q8.size() == f32.size());
    for (std::size_t i = 0; i < f32.size(); ++i) {
        REQUIRE(cosine(f32[i], q8[i]) > 0.999);
        REQUIRE(cosine(f32[i], q4[i]) > 0.90);
        REQUIRE(cosine(f32[i], q8[i]) >= cosine(f32[i], q4[i]) - 1e-6);
    }

    // Footprint: quantized KV rows are much smaller than F32. F32 is
    // 4 B/elem; Q8 ~34 B/32-block (f16 scale + 32 int8), Q4 ~18 B/
    // 32-block (f16 scale + 32 nibbles) -> ~3.76x / ~7.1x on the KV
    // bytes (the f16 scale is the only overhead vs 4x / 8x).
    const std::size_t f32_row =
        static_cast<std::size_t>(geom.kv_dim) * sizeof(float);
    const std::size_t q8_row =
        locus::kv::kv_row_bytes(geom.kv_dim, locus::kv::KvType::kQ8);
    const std::size_t q4_row =
        locus::kv::kv_row_bytes(geom.kv_dim, locus::kv::KvType::kQ4);
    const double q8_ratio =
        static_cast<double>(f32_row) / static_cast<double>(q8_row);
    const double q4_ratio =
        static_cast<double>(f32_row) / static_cast<double>(q4_row);
    WARN("KV footprint (kv_dim="
         << geom.kv_dim << "): F32 " << f32_row << " B/row, Q8 "
         << q8_row << " B (" << q8_ratio << "x), Q4 " << q4_row
         << " B (" << q4_ratio << "x)");
    REQUIRE(q8_ratio > 3.5);
    REQUIRE(q4_ratio > 6.5);
}

// #45 validation: quantized KV read/write in the Vulkan attention
// shaders (attn_paged_q + kv_store_q) off the GPU-mapped byte pool.
// Uses llama-1b (real kv_dim=512, a multiple of 32; ~0.8GB, in-RAM
// safe on the 4GB Pi) because the stories260K toy is too narrow to
// exercise the GPU path. Runtime-gated on a usable Vulkan device.
TEST_CASE("quantized KV cache tracks F32 under the Vulkan backend",
          "[e2e][kv][vulkan]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    const std::string path = std::string(LOCUS_SOURCE_DIR) +
        "/tests/models/llama-3.2-1b-q4_k_m.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP("llama-3.2-1b model not present");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::tokenizer_from_gguf(g);
    model.use_backend(*locus::backend::find_backend("vulkan"));

    const auto geom = model.make_cache().geometry();
    if (geom.kv_dim % 32 != 0) {
        SKIP("model kv_dim is not a multiple of the quant block");
    }
    const auto ids =
        tok->encode("Once upon a time there was a", true);
    const auto f32 = teacher_forced_logits(model, ids,
                                           locus::kv::KvType::kF32);
    const auto q8 = teacher_forced_logits(model, ids,
                                          locus::kv::KvType::kQ8);
    const auto q4 = teacher_forced_logits(model, ids,
                                          locus::kv::KvType::kQ4);
    REQUIRE(q8.size() == f32.size());
    for (std::size_t i = 0; i < f32.size(); ++i) {
        REQUIRE(cosine(f32[i], q8[i]) > 0.999);
        REQUIRE(cosine(f32[i], q4[i]) > 0.90);
        REQUIRE(cosine(f32[i], q8[i]) >=
                cosine(f32[i], q4[i]) - 1e-6);
    }
}

// #37: qwen2moe (Qwen1.5-MoE-A2.7B) end-to-end. Greedy decode must
// reproduce the llama.cpp golden (llama-simple, same Q4_K_M GGUF)
// token-exact -- validates the whole arch: GQA + q/k/v bias + NEOX
// rope + norm_topk=false routing + the gated shared expert. BPE
// (gpt2) tokenizer, so use tokenizer_from_gguf, not SpmTokenizer.
TEST_CASE("qwen2moe greedy decode matches the llama.cpp golden",
          "[e2e][qwen]") {
    const std::string path = qwen_moe_path();
    if (!std::filesystem::exists(path)) {
        SKIP("Qwen-MoE model not present (9.5GB, gitignored)");
    }
    auto g = locus::gguf::GgufFile::open(path);
    REQUIRE(g.get_string("general.architecture") == "qwen2moe");
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::tokenizer_from_gguf(g);

    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(model.hparams().n_vocab);
    auto ids = tok->encode("Once upon a time", true);
    for (auto id : ids) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(id, cache, seq, ws, logits);
    }
    for (int i = 0; i < 32; ++i) {
        const auto next = locus::model::argmax(logits);
        if (next == tok->eos_id()) {
            break;
        }
        ids.push_back(next);
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(next, cache, seq, ws, logits);
    }
    cache.release(seq);
    const std::string golden =
        "Once upon a time, there was a little girl who loved to "
        "play with her dolls. She would spend hours dressing them "
        "up, brushing their hair, and playing with them. One";
    REQUIRE(tok->decode(ids) == golden);
}

TEST_CASE("greedy decode matches the llama.cpp golden output",
          "[e2e]") {
    const std::string path = model_path();
    if (!std::filesystem::exists(path)) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    // Produced by: llama-completion -m stories260K.gguf
    //   -p "Once upon a time" -n 40 --temp 0
    std::ifstream golden_file(std::string(LOCUS_SOURCE_DIR) +
                              "/tests/golden/stories260K_once.txt");
    REQUIRE(golden_file.good());
    std::stringstream ss;
    ss << golden_file.rdbuf();
    const std::string golden = trim(ss.str());

    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

    REQUIRE(trim(generate_text(model, tok, "Once upon a time",
                               40)) == golden);
}

TEST_CASE("llama-3.2 (BPE + scaled rope) runs consistently",
          "[e2e][backend]") {
    if (!std::filesystem::exists(llama32_path())) {
        SKIP("model not present (llama-3.2-1b-q8_0.gguf)");
    }
    auto g = locus::gguf::GgufFile::open(llama32_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok_ptr = locus::tok::tokenizer_from_gguf(g);
    REQUIRE(!model.rope_factors().empty());  // llama3 scaling

    // All selectable backends must agree with each other; the
    // text must answer the prompt (no llama.cpp golden here: it
    // auto-applies the chat template to this model).
    std::string reference;
    for (const auto& b : locus::backend::backends()) {
        if (!b.available || !b.selectable) {
            continue;
        }
        INFO("backend " << b.name);
        model.use_backend(b);
        auto cache = model.make_cache();
        auto ws = model.make_workspace();
        locus::kv::PagedKvCache::Seq seq;
        std::vector<float> logits(model.hparams().n_vocab);
        auto ids =
            tok_ptr->encode("The capital of France is", true);
        for (auto id : ids) {
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(id, cache, seq, ws, logits);
        }
        for (int i = 0; i < 8; ++i) {
            auto next = locus::model::argmax(logits);
            ids.push_back(next);
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(next, cache, seq, ws, logits);
        }
        cache.release(seq);
        const std::string text = tok_ptr->decode(ids);
        REQUIRE(text.find("Paris") != std::string::npos);
        if (reference.empty()) {
            reference = text;
        } else {
            REQUIRE(text == reference);
        }
    }
}

TEST_CASE("every selectable backend reproduces the golden output",
          "[e2e][backend]") {
    const std::string path = model_path();
    if (!std::filesystem::exists(path)) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    std::ifstream golden_file(std::string(LOCUS_SOURCE_DIR) +
                              "/tests/golden/stories260K_once.txt");
    REQUIRE(golden_file.good());
    std::stringstream ss;
    ss << golden_file.rdbuf();
    const std::string golden = trim(ss.str());

    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

    for (const auto& b : locus::backend::backends()) {
        if (!b.available || !b.selectable) {
            continue;
        }
        DYNAMIC_SECTION("backend " << b.name) {
            model.use_backend(b);
            REQUIRE(trim(generate_text(model, tok,
                                       "Once upon a time", 40)) ==
                    golden);
        }
    }
}
