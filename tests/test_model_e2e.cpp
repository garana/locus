#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan_forward.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
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

std::string qwen3_coder_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/qwen3-coder-30b-a3b-instruct-q4_k_m.gguf";
}

std::string dbrx_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/dbrx-instruct-16x12b-iq2_xxs.gguf";
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
std::string generate_text(locus::model::TransformerModel& model,
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
    locus::model::TransformerModel& model,
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
    auto model = locus::model::TransformerModel::load(g);
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
    const auto f16 = teacher_forced_logits(model, ids,
                                           locus::kv::KvType::kF16);

    REQUIRE(q8.size() == f32.size());
    REQUIRE(f16.size() == f32.size());
    for (std::size_t i = 0; i < f32.size(); ++i) {
        // F16 KV keeps 11 significant bits: closest of the three.
        REQUIRE(cosine(f32[i], f16[i]) > 0.9999);
        REQUIRE(cosine(f32[i], f16[i]) >= cosine(f32[i], q8[i]) - 1e-6);
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
    auto model = locus::model::TransformerModel::load(g);
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
    auto model = locus::model::TransformerModel::load(g);
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
    auto model = locus::model::TransformerModel::load(g);
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

// qwen3moe (Qwen3-Coder-30B-A3B-Instruct, unsloth Q4_K_M GGUF, 18.6GB:
// huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF). Golden
// prompt token counts and greedy continuations are llama.cpp's (via
// ollama on the same GGUF: raw prompt, temperature 0, no repeat
// penalty). The prompts exercise the qwen2 pretokenizer (a punctuation
// char prefixing a word, tabs, single digits) and the arch (per-head
// q/k norm, head_dim 128 != n_embd / n_heads, routed-only MoE).
TEST_CASE("qwen3moe greedy decode matches the llama.cpp golden",
          "[e2e][qwen]") {
    const std::string path = qwen3_coder_path();
    if (!std::filesystem::exists(path)) {
        SKIP("Qwen3-Coder model not present (18.6GB, gitignored)");
    }
    auto g = locus::gguf::GgufFile::open(path);
    REQUIRE(g.get_string("general.architecture") == "qwen3moe");
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::tokenizer_from_gguf(g);
    // The CPU path, and the Vulkan full GPU forward when a device is
    // there (per-head q/k norm + NEOX rope on the GPU).
    std::vector<const char*> backends = {"neon"};
    if (locus::backend::find_backend("neon") == nullptr) {
        backends = {"scalar"};
    }
    if (locus::backend::vulkan_backend_usable()) {
        backends.push_back("vulkan");
    }

    // llama.cpp's prompt token counts (qwen2 pretokenizer; the gpt2
    // fallback gave 6, 6, 10 and 31).
    const std::pair<const char*, std::size_t> counts[] = {
        {"x.append(item)", 4},
        {"foo_bar(baz)", 5},
        {"\tif (n<2) return n;", 9},
        {"  for (int i = 0; i < 10; ++i) {\n\tprintf(\"%d\\n\", i);\n}",
         27},
    };
    for (const auto& [text, n] : counts) {
        INFO(text);
        REQUIRE(tok->encode(text, true).size() == n);
    }

    struct Golden {
        const char* prompt;
        std::size_t prompt_tokens;
        const char* continuation;  // llama.cpp, 24 tokens
    };
    const Golden goldens[] = {
        {"def fibonacci(n):", 4,
         "\n    if n <= 1:\n        return n\n    else:\n        "
         "return fibonacci(n-1) + fibonacci"},
        {"x.append(item)\n\tif (n<2) return n;", 13,
         "\n\tfor (int i=2; i<n; i++) {\n\t\tif (x[i-1] > x"},
    };
    for (const char* be : backends) {
        model.use_backend(*locus::backend::find_backend(be));
        for (const Golden& gd : goldens) {
            INFO(be << ": " << gd.prompt);
            auto cache = model.make_cache();
            auto ws = model.make_workspace();
            locus::kv::PagedKvCache::Seq seq;
            std::vector<float> logits(model.hparams().n_vocab);
            auto ids = tok->encode(gd.prompt, true);
            REQUIRE(ids.size() == gd.prompt_tokens);
            for (auto id : ids) {
                REQUIRE(cache.ensure_capacity(seq, 1));
                model.forward(id, cache, seq, ws, logits);
            }
            std::vector<locus::tok::TokenId> gen;
            for (int i = 0; i < 24; ++i) {
                const auto next = locus::model::argmax(logits);
                gen.push_back(next);
                REQUIRE(cache.ensure_capacity(seq, 1));
                model.forward(next, cache, seq, ws, logits);
            }
            cache.release(seq);
            REQUIRE(tok->decode(gen) == gd.continuation);
        }
        // Batched prefill (per-head q/k norm and NEOX rope over a
        // chunk's rows) ingesting the longer prompt, then decode. On
        // vulkan the GPU batch path itself must take the chunk (Q4_K /
        // Q6_K matmuls and expert id-matmuls), not the per-token
        // fallback forward_batch would quietly use.
        if (model.supports_batch_prefill()) {
            const Golden& gd = goldens[1];
            INFO(be << " batched prefill: " << gd.prompt);
            auto cache = model.make_cache();
            auto ws = model.make_workspace();
            locus::kv::PagedKvCache::Seq seq;
            std::vector<float> logits(model.hparams().n_vocab);
            const auto ids = tok->encode(gd.prompt, true);
            REQUIRE(cache.ensure_capacity(
                seq, static_cast<std::uint32_t>(ids.size())));
            if (std::string(be) == "vulkan") {
                REQUIRE(locus::backend::vulkan_forward_batch(
                    model, ids, cache, seq, logits, false));
            } else {
                model.forward_batch(ids, cache, seq, ws, logits);
            }
            std::vector<locus::tok::TokenId> gen;
            for (int i = 0; i < 24; ++i) {
                const auto next = locus::model::argmax(logits);
                gen.push_back(next);
                REQUIRE(cache.ensure_capacity(seq, 1));
                model.forward(next, cache, seq, ws, logits);
            }
            cache.release(seq);
            REQUIRE(tok->decode(gen) == gd.continuation);
        }
    }
}

// R16 opt-in Q8_K-activation mode (use_q8k_activations): the QK_K
// matmuls integer-dot a Q8_K-quantized activation instead of the f32
// dot. Exact output is NOT byte-stable vs the f32 golden -- Q8_K's
// final accumulation order is implementation-specific -- so this is a
// prefix smoke test on the unambiguous opening, confirming the mode
// runs end-to-end and stays coherent. Deeper reconciliation to the
// llama.cpp reference is a per-model, tolerance-based exercise.
TEST_CASE("qwen2moe runs under opt-in Q8_K-activation mode",
          "[e2e][qwen][q8k]") {
    const std::string path = qwen_moe_path();
    if (!std::filesystem::exists(path)) {
        SKIP("Qwen-MoE model not present (9.5GB, gitignored)");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    model.use_q8k_activations(true);  // the mode under test
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
    for (int i = 0; i < 20; ++i) {
        const auto next = locus::model::argmax(logits);
        if (next == tok->eos_id()) {
            break;
        }
        ids.push_back(next);
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(next, cache, seq, ws, logits);
    }
    cache.release(seq);
    // The opening sentence is argmax-unambiguous, so it holds across
    // f32/Q8_K/llama.cpp regardless of accumulation order.
    REQUIRE(tok->decode(ids).starts_with(
        "Once upon a time, there was a little girl who loved to "
        "play with her dolls."));
}

// Same qwen2moe golden with the CUDA backend live: attention is CPU
// but the projection / expert matvecs offload to the GPU, so this
// proves the new arch is token-exact with GPU matvec in the loop
// (Q4_K on device). Runtime-gated on a CUDA device.
TEST_CASE("qwen2moe golden holds under the CUDA backend",
          "[e2e][qwen][cuda]") {
    if (!locus::backend::cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    const std::string path = qwen_moe_path();
    if (!std::filesystem::exists(path)) {
        SKIP("Qwen-MoE model not present (9.5GB, gitignored)");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::tokenizer_from_gguf(g);
    model.use_backend(*locus::backend::find_backend("cuda"));

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

// #50: DBRX end-to-end. Validates the arch (fused QKV + clip_qkv,
// LayerNorm, NEOX rope, fine-grained MoE 16x/4 no-shared) against the
// llama.cpp golden. 20 tokens: locus does an f32-activation dot while
// llama.cpp quantizes activations to Q8_K, so on this 2-bit (IQ2_XXS)
// model the argmax tips ~token 27; 20 is comfortably exact and pins
// the arch. BPE (gpt2 fallback) tokenizer.
TEST_CASE("dbrx greedy decode matches the llama.cpp golden",
          "[e2e][dbrx]") {
    const std::string path = dbrx_path();
    if (!std::filesystem::exists(path)) {
        SKIP("DBRX model not present (32GB, gitignored)");
    }
    auto g = locus::gguf::GgufFile::open(path);
    REQUIRE(g.get_string("general.architecture") == "dbrx");
    auto model = locus::model::TransformerModel::load(g);
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
    for (int i = 0; i < 20; ++i) {
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
        "Once upon a time, there was a magical land called the "
        "Internet. The Internet was a magical land because it was a";
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
    auto model = locus::model::TransformerModel::load(g);
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
    auto model = locus::model::TransformerModel::load(g);
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
    auto model = locus::model::TransformerModel::load(g);
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
