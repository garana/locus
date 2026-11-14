// Micro-benchmarks for the matvec paths. Hidden ([.]) so they never
// run in the normal suite; invoke explicitly:
//   ./build/tests/locus_tests "[bench]"
// Reports ns per matvec so the Q8_K opt-in path (DESIGN.md R16) can be
// compared to the f32-dequant default -- Q8_K exists for integer-dot
// throughput, and this is where that win is verified.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/embed/embedding_model.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/http/metrics.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/codec.hpp"
#include "locus/model/gqa_attend.hpp"
#include "locus/model/transformer.hpp"
#include "locus/tok/tokenizer.hpp"
#include "locus/sys/features.hpp"

using namespace locus::backend;

namespace {

// A rows x cols k-quant weight with plausible per-block scales.
std::vector<std::byte> make_kquant(std::uint32_t rows,
                                   std::uint32_t cols, std::size_t bytes,
                                   std::size_t d_off, int dmin_off,
                                   std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte_d(0, 255);
    std::vector<std::byte> w(static_cast<std::size_t>(rows) *
                             (cols / 256) * bytes);
    for (auto& b : w) {
        b = static_cast<std::byte>(byte_d(rng));
    }
    for (std::size_t bl = 0; bl < w.size() / bytes; ++bl) {
        std::byte* blk = w.data() + bl * bytes;
        const std::uint16_t d = f32_to_f16(0.01f);
        const std::uint16_t dmin = f32_to_f16(0.005f);
        std::memcpy(blk + d_off, &d, 2);
        if (dmin_off >= 0) {
            std::memcpy(blk + dmin_off, &dmin, 2);
        }
    }
    return w;
}

double time_matvec(void (*mv)(const Mat&, std::span<const float>,
                              std::span<float>),
                   const Mat& m, std::span<const float> x,
                   std::span<float> out, int iters) {
    for (int i = 0; i < 3; ++i) {
        mv(m, x, out);  // warm up
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        mv(m, x, out);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns =
        std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns / iters;
}

}  // namespace

TEST_CASE("bench: k-quant matvec f32 vs Q8_K", "[.][bench]") {
    using TT = locus::gguf::TensorType;
    struct Q {
        TT type;
        const char* name;
        std::size_t bytes;
        std::size_t d_off;
        int dmin_off;
    };
    const std::uint32_t rows = 4096, cols = 4096;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(cols), out(rows);
    for (auto& v : x) {
        v = dist(rng);
    }
    const int iters = 200;
    for (const Q& q : {Q{TT::kQ4_K, "Q4_K", 144, 0, 2},
                       Q{TT::kQ5_K, "Q5_K", 176, 0, 2},
                       Q{TT::kQ6_K, "Q6_K", 210, 208, -1}}) {
        auto w = make_kquant(rows, cols, q.bytes, q.d_off, q.dmin_off,
                             7);
        Mat m{q.type, w.data(), rows, cols};
        const double s_f32 = time_matvec(&matvec, m, x, out, iters);
        const double s_q8k = time_matvec(&matvec_q8k, m, x, out, iters);
        WARN("scalar " << q.name << ": f32=" << s_f32 << "ns q8k="
                       << s_q8k << "ns speedup=" << s_f32 / s_q8k
                       << "x");
#if defined(__x86_64__)
        if (locus::sys::detect().sse4) {
            const double x_f32 =
                time_matvec(&matvec_sse4, m, x, out, iters);
            const double x_q8k =
                time_matvec(&matvec_sse4_q8k, m, x, out, iters);
            WARN("sse4   " << q.name << ": f32=" << x_f32 << "ns q8k="
                           << x_q8k << "ns speedup=" << x_f32 / x_q8k
                           << "x");
        }
#endif
    }
    SUCCEED();
}

#if defined(__aarch64__)
// The NEON Q8_K dots of the IQ, TQ and Q2_K types against the scalar
// ones they match bit for bit. (IQ1_M's f16 scale is assembled from
// nibbles; random bytes may make it inf or NaN, which does not change
// the timing.)
TEST_CASE("bench: neon q8k IQ TQ and Q2_K", "[.][bench]") {
    using TT = locus::gguf::TensorType;
    struct Q {
        TT type;
        const char* name;
        std::size_t bytes;
        std::size_t d_off;
        int dmin_off;
    };
    const std::uint32_t rows = 2048, cols = 4096;
    std::mt19937 rng(744);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(cols), out(rows);
    for (auto& v : x) {
        v = dist(rng);
    }
    const int iters = 20;
    for (const Q& q : {Q{TT::kIQ3_XXS, "IQ3_XXS", 98, 0, -1},
                       Q{TT::kIQ3_S, "IQ3_S", 110, 0, -1},
                       Q{TT::kIQ1_M, "IQ1_M", 56, 0, -1},
                       Q{TT::kIQ4_XS, "IQ4_XS", 136, 0, -1},
                       Q{TT::kQ2_K, "Q2_K", 84, 80, 82},
                       Q{TT::kIQ2_XS, "IQ2_XS", 74, 0, -1},
                       Q{TT::kIQ1_S, "IQ1_S", 50, 0, -1},
                       Q{TT::kTQ1_0, "TQ1_0", 54, 52, -1},
                       Q{TT::kTQ2_0, "TQ2_0", 66, 64, -1}}) {
        auto w = make_kquant(rows, cols, q.bytes, q.d_off, q.dmin_off, 7);
        Mat m{q.type, w.data(), rows, cols};
        const double s = time_matvec(&matvec_q8k, m, x, out, iters);
        const double v = time_matvec(&matvec_neon_q8k, m, x, out, iters);
        // 16 tokens batched (each block decoded once per 4 tokens)
        // against 16 one-token calls, ns per token.
        constexpr std::uint32_t kN = 16;
        std::vector<float> xb(std::size_t{kN} * cols);
        std::vector<float> ob(std::size_t{kN} * rows);
        for (std::uint32_t t = 0; t < kN; ++t) {
            std::copy(x.begin(), x.end(), xb.begin() + std::size_t{t} * cols);
        }
        matvec_batch_neon_q8k(m, xb, ob, kN);  // warm
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters / 4; ++i) {
            matvec_batch_neon_q8k(m, xb, ob, kN);
        }
        const double b = std::chrono::duration<double, std::nano>(
                             std::chrono::steady_clock::now() - t0)
                             .count() /
                         (iters / 4) / kN;
        WARN(q.name << ": scalar q8k=" << s << "ns neon q8k=" << v
                    << "ns speedup=" << s / v << "x; batched 16: " << b
                    << "ns/token (" << v / b << "x one-token)");
    }
    SUCCEED();
}
#endif

// i#52 measure-first: the last stage projects out_w_ (the LM head, the
// biggest tensor) once PER token via matvec. Amortizing it into one
// matvec_batch reads the weight once for the whole batch; this measures
// whether that actually wins on THIS GPU at the batch widths we run,
// before committing a bigger activation buffer + a transpose to it.
// Synthetic Q6_K LM head (the usual output.weight type) at vocab x embd.
TEST_CASE("bench: out_w_ n x matvec_cuda vs matvec_batch_cuda",
          "[.][bench][cuda]") {
    if (!cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    using TT = locus::gguf::TensorType;
    const std::uint32_t rows = 32000;  // vocab
    const std::uint32_t cols = 2048;   // embd
    // Q6_K, Q5_K, and Q8_0 (a tied head): 256 columns are 8 Q8_0
    // blocks, 272 bytes; scales past the first are random f16, which
    // does not change the time.
    struct Head {
        TT type;
        std::size_t bytes, d_off;
        const char* name;
    };
    for (const Head h : {Head{TT::kQ6_K, 210, 208, "Q6_K"},
                         Head{TT::kQ5_K, 176, 0, "Q5_K"},
                         Head{TT::kQ8_0, 272, 0, "Q8_0"}}) {
        auto w = make_kquant(rows, cols, h.bytes, h.d_off,
                             /*dmin_off=*/-1, 7);
        Mat m{h.type, w.data(), rows, cols};
        std::mt19937 rng(5);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        const int iters = 50;
        for (std::uint32_t n : {1u, 4u, 8u, 16u}) {
            std::vector<float> xb(static_cast<std::size_t>(n) * cols);
            for (auto& v : xb) {
                v = dist(rng);
            }
            std::vector<float> col(rows);
            std::vector<float> outb(static_cast<std::size_t>(rows) * n);
            cuda_pool_reset();
            auto per_token = [&] {
                for (std::uint32_t t = 0; t < n; ++t) {
                    matvec_cuda(m,
                                {xb.data() +
                                     static_cast<std::size_t>(t) * cols,
                                 cols},
                                col);
                }
            };
            auto batched = [&] { matvec_batch_cuda(m, xb, outb, n); };
            for (int i = 0; i < 3; ++i) {  // warm (upload + JIT)
                per_token();
                batched();
            }
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                per_token();
            }
            const auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                batched();
            }
            const auto t2 = std::chrono::steady_clock::now();
            const double per_ns =
                std::chrono::duration<double, std::nano>(t1 - t0)
                    .count() /
                iters;
            const double bat_ns =
                std::chrono::duration<double, std::nano>(t2 - t1)
                    .count() /
                iters;
            WARN("cuda out_w_ "
                 << h.name << " n=" << n
                 << ": per-token=" << per_ns << "ns batched=" << bat_ns
                 << "ns speedup=" << per_ns / bat_ns << "x");
        }
    }
    SUCCEED();
}

namespace {

/**
 * One embedding pass on the CPU against `gpu` for each encoder model
 * present, at the sizes EmbeddingModel's GPU thresholds (kGpuMinTokens,
 * kGpuMacs) were set from on Apple. Each input is the same sentence
 * repeated; the best of three runs after a warm-up.
 *
 * @param name The GPU's name in the report.
 * @param reset Frees the GPU's weight pool before each model.
 */
void bench_encoders(const char* name, const Ops& gpu,
                    void (*reset)()) {
    const auto& cpu = best_backend().ops;
    const std::string sentence =
        "Beijing is the capital of the People's Republic of China. ";
    const auto best_ms = [](const auto& fn) {
        fn();  // warm (weight upload, first touch)
        double best = 1e30;
        for (int i = 0; i < 3; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            const auto t1 = std::chrono::steady_clock::now();
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return best;
    };
    for (const char* file :
         {"bge-small-en-v1.5-q8_0.gguf", "bge-m3-q8_0.gguf",
          "nomic-embed-text-v1.5.Q8_0.gguf"}) {
        const std::string path =
            std::string(LOCUS_SOURCE_DIR) + "/tests/models/" + file;
        if (!std::filesystem::exists(path)) continue;
        const auto m = locus::embed::EmbeddingModel::load(path);
        reset();
        // About 15, 60, 120 and 470 tokens in one pass: up to eight
        // inputs of one sentence, or one long input under the 512-token
        // limit (13 tokens a sentence on bge-small).
        for (const auto& [inputs, reps] :
             {std::pair{1, 1}, std::pair{4, 1}, std::pair{8, 1},
              std::pair{1, 36}}) {
            std::string text;
            for (int r = 0; r < reps; ++r) text += sentence;
            const std::vector<std::string> in(
                static_cast<std::size_t>(inputs), text);
            std::size_t tokens = 0;
            m->embed(in, cpu, &tokens);
            const double c = best_ms([&] { m->embed(in, cpu); });
            const double g = best_ms([&] { m->embed(in, gpu); });
            WARN(file << " " << tokens << " tokens: cpu " << c << " ms, "
                      << name << " " << g << " ms (" << c / g << "x)");
        }
    }
}

}  // namespace

// i#519: the encoder passes on the CUDA ops.
TEST_CASE("bench: encoder pass cpu vs cuda", "[.][bench][cuda]") {
    const auto* cuda = find_backend("cuda");
    if (cuda == nullptr || !cuda->available) {
        SKIP("no usable CUDA device");
    }
    bench_encoders("cuda", cuda->ops, &cuda_pool_reset);
    SUCCEED();
}

// i#680: the encoder passes on the Vulkan image ops, which copy the
// weights into the device pool where the GPU does not import host
// memory. The server uses them only where Vulkan is measured faster
// than the CPU (vulkan_device_beats_cpu); this times any device.
TEST_CASE("bench: encoder pass cpu vs vulkan", "[.][bench][vulkan]") {
    if (!vulkan_backend_usable()) {
        SKIP("no Vulkan device");
    }
    bench_encoders("vulkan", vulkan_image_ops(), &vulkan_reset_state);
    SUCCEED();
}

// Orpheus speech decode on CUDA (i#689): ms per token with the speech
// head rows and with the full head. Run it once as is, and once with
// LOCUS_CUDA_FORWARD=0 for the per-op path (read once per process).
TEST_CASE("bench: orpheus decode step on cuda with speech rows",
          "[.][bench][cuda]") {
    if (!cuda_backend_usable()) {
        SKIP("no usable CUDA device");
    }
    const std::string path = std::string(LOCUS_SOURCE_DIR) +
                             "/tests/models/orpheus-3b-0.1-ft-Q4_K_M.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP(path + " absent");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    model.use_backend(*find_backend("cuda"));
    const auto tok = locus::tok::tokenizer_from_gguf(g);
    // The server's speech rows: <custom_token_0> through the 7 bands.
    const auto base =
        static_cast<std::uint32_t>(locus::model::orpheus_band_base(*tok));
    const std::uint32_t nv = model.hparams().n_vocab;
    const locus::model::LogitRows speech{base - 10,
                                         std::min(nv, base + 7u * 4096u)};
    const auto prompt = tok->encode("tara: Hello there, how are you?", true);
    constexpr std::uint32_t kSteps = 48;
    auto ws = model.make_workspace();
    std::vector<float> logits(nv);
    const auto ms_per_token = [&](locus::model::LogitRows rows) {
        auto cache = model.make_cache();
        locus::kv::PagedKvCache::Seq seq;
        for (const auto id : prompt) {
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(id, cache, seq, ws, logits, rows);
        }
        // Fixed inputs (the band base) so both runs do the same work.
        const auto t0 = std::chrono::steady_clock::now();
        for (std::uint32_t i = 0; i < kSteps; ++i) {
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(static_cast<locus::tok::TokenId>(base), cache, seq,
                          ws, logits, rows);
        }
        const auto t1 = std::chrono::steady_clock::now();
        cache.release(seq);
        return std::chrono::duration<double, std::milli>(t1 - t0).count() /
               kSteps;
    };
    ms_per_token({});  // warm: weight pool, first touch
    const double rows_ms = ms_per_token(speech);
    const double full_ms = ms_per_token({});
    WARN("orpheus on cuda, " << kSteps << " steps: speech rows ["
                             << speech.begin << ", " << speech.end << ") "
                             << rows_ms << " ms/token, full head " << full_ms
                             << " ms/token");
    SUCCEED();
}

// The same on a model that fits a 2 GB card whole (Orpheus 3B pages
// through the weight pool there, which hides the path): Llama 3.2 1B
// with a range as wide as Orpheus's speech rows (28,682).
TEST_CASE("bench: llama 1b decode step on cuda with a head row range",
          "[.][bench][cuda]") {
    if (!cuda_backend_usable()) {
        SKIP("no usable CUDA device");
    }
    const std::string path = std::string(LOCUS_SOURCE_DIR) +
                             "/tests/models/llama-3.2-1b-q4_k_m.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP(path + " absent");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    model.use_backend(*find_backend("cuda"));
    const std::uint32_t nv = model.hparams().n_vocab;
    const locus::model::LogitRows part{nv - 28682, nv};
    constexpr std::uint32_t kSteps = 128;
    auto ws = model.make_workspace();
    std::vector<float> logits(nv);
    const auto ms_per_token = [&](locus::model::LogitRows rows) {
        auto cache = model.make_cache();
        locus::kv::PagedKvCache::Seq seq;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::uint32_t i = 0; i < kSteps; ++i) {
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(static_cast<locus::tok::TokenId>(1000 + i), cache,
                          seq, ws, logits, rows);
        }
        const auto t1 = std::chrono::steady_clock::now();
        cache.release(seq);
        return std::chrono::duration<double, std::milli>(t1 - t0).count() /
               kSteps;
    };
    ms_per_token({});  // warm: weight pool, first touch
    const double rows_ms = ms_per_token(part);
    const double full_ms = ms_per_token({});
    WARN("llama 1b on cuda, " << kSteps << " steps: rows [" << part.begin
                              << ", " << part.end << ") " << rows_ms
                              << " ms/token, full head " << full_ms
                              << " ms/token");
    SUCCEED();
}

#if defined(__x86_64__)
// i#719: the batched SSE4 K-quant rows (prefill's matvec) on a
// feed-forward-sized weight (2048 x 8192) for a 64-token chunk, one
// thread. Random bytes with a small f16 scale: only the time counts.
// i#721: the per-token (decode) SSE4 q8k matvec on the same 2048 x 8192
// weights, one thread, with the weight bytes read per second. Random
// bytes with a small f16 scale: only the time counts.
TEST_CASE("bench: sse4 q8k decode rows", "[.][bench]") {
    if (!locus::sys::detect().sse4) {
        SKIP("CPU lacks SSE4.1");
    }
    constexpr std::uint32_t rows = 2048, cols = 8192, reps = 50;
    std::mt19937 rng(12);
    std::uniform_int_distribution<int> byte_d(0, 255);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = xd(rng);
    }
    struct Q {
        const char* name;
        locus::gguf::TensorType type;
        std::size_t bytes, d_off;
    };
    for (const Q& q : {Q{"q4_k", locus::gguf::TensorType::kQ4_K, 144, 0},
                       Q{"q6_k", locus::gguf::TensorType::kQ6_K, 210, 208}}) {
        std::vector<std::byte> w(std::size_t{rows} * (cols / 256) * q.bytes);
        for (auto& b : w) {
            b = static_cast<std::byte>(byte_d(rng));
        }
        for (std::size_t bl = 0; bl < w.size() / q.bytes; ++bl) {
            const std::uint16_t d = f32_to_f16(0.01f);
            std::memcpy(w.data() + bl * q.bytes + q.d_off, &d, 2);
        }
        const Mat m{q.type, w.data(), rows, cols};
        std::vector<float> out(rows);
        matvec_sse4_q8k(m, x, out);  // warm
        const auto t0 = std::chrono::steady_clock::now();
        for (std::uint32_t r = 0; r < reps; ++r) {
            matvec_sse4_q8k(m, x, out);
        }
        const double s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0)
                             .count() /
                         reps;
        WARN(q.name << " " << rows << " x " << cols << ", 1 token: "
                    << s * 1e3 << " ms per call, "
                    << static_cast<double>(w.size()) / s / 1e9
                    << " GB/s of weights");
    }
    SUCCEED();
}

TEST_CASE("bench: sse4 q8k batch rows", "[.][bench]") {
    if (!locus::sys::detect().sse4) {
        SKIP("CPU lacks SSE4.1");
    }
    constexpr std::uint32_t rows = 2048, cols = 8192, n = 64, reps = 5;
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> byte_d(0, 255);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    std::vector<float> x(std::size_t{n} * cols);
    for (auto& v : x) {
        v = xd(rng);
    }
    // bytes and d_off are per scale block (32 columns for Q8_0, 256 for
    // the K-quants).
    struct Q {
        const char* name;
        locus::gguf::TensorType type;
        std::size_t block, bytes, d_off;
    };
    using Fn = void (*)(const Mat&, std::span<const float>, std::span<float>,
                        std::uint32_t);
    std::vector<std::pair<const char*, Fn>> builds = {
        {"sse4", &matvec_batch_sse4_q8k}};
#if defined(LOCUS_HAS_SSE4_AVX_VARIANT)
    if (locus::sys::detect().avx) {
        builds.push_back({"sse4_avx", &sse4_avx::matvec_batch_sse4_q8k});
    }
#endif
    for (const Q& q :
         {Q{"q8_0", locus::gguf::TensorType::kQ8_0, 32, 34, 0},
          Q{"q4_k", locus::gguf::TensorType::kQ4_K, 256, 144, 0},
          Q{"q6_k", locus::gguf::TensorType::kQ6_K, 256, 210, 208}}) {
        std::vector<std::byte> w(std::size_t{rows} * (cols / q.block) *
                                 q.bytes);
        for (auto& b : w) {
            b = static_cast<std::byte>(byte_d(rng));
        }
        for (std::size_t bl = 0; bl < w.size() / q.bytes; ++bl) {
            const std::uint16_t d = f32_to_f16(0.01f);
            std::memcpy(w.data() + bl * q.bytes + q.d_off, &d, 2);
        }
        const Mat m{q.type, w.data(), rows, cols};
        std::vector<float> out(std::size_t{rows} * n);
        for (const auto& [build, fn] : builds) {
            fn(m, x, out, n);  // warm
            const auto t0 = std::chrono::steady_clock::now();
            for (std::uint32_t r = 0; r < reps; ++r) {
                fn(m, x, out, n);
            }
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count() /
                              reps;
            WARN(build << " " << q.name << " " << rows << " x " << cols
                       << ", " << n << " tokens: " << ms << " ms per call");
        }
    }
    SUCCEED();
}
#endif

// Per-type CUDA matvec time for the IQ quants on a 4096 x 4096 weight
// (fully resident). The wall time includes the per-call copies and
// sync; run it under nsys for kernel-only times. Random bytes with a
// small f16 at offset 0: values are meaningless, only the time counts.
TEST_CASE("bench: cuda iq matvec", "[.][bench][cuda]") {
    if (!cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    using TT = locus::gguf::TensorType;
    struct Q {
        TT type;
        const char* name;
        std::uint32_t block;  // weights per block
        std::size_t bytes;    // bytes per block
    };
    const std::uint32_t rows = 4096, cols = 4096;
    std::mt19937 rng(13);
    std::uniform_int_distribution<int> byte_d(0, 255);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(cols), out(rows);
    for (auto& v : x) {
        v = dist(rng);
    }
    const int iters = 50;
    for (const Q& q : {Q{TT::kIQ4_XS, "IQ4_XS", 256, 136},
                       Q{TT::kIQ4_NL, "IQ4_NL", 32, 18},
                       Q{TT::kIQ3_XXS, "IQ3_XXS", 256, 98},
                       Q{TT::kIQ3_S, "IQ3_S", 256, 110},
                       Q{TT::kIQ2_XXS, "IQ2_XXS", 256, 66},
                       Q{TT::kIQ2_XS, "IQ2_XS", 256, 74},
                       Q{TT::kIQ2_S, "IQ2_S", 256, 82},
                       Q{TT::kIQ1_S, "IQ1_S", 256, 50},
                       Q{TT::kIQ1_M, "IQ1_M", 256, 56}}) {
        const std::size_t nblk =
            static_cast<std::size_t>(rows) * (cols / q.block);
        std::vector<std::byte> w(nblk * q.bytes);
        for (auto& b : w) {
            b = static_cast<std::byte>(byte_d(rng));
        }
        for (std::size_t bl = 0; bl < nblk; ++bl) {
            const std::uint16_t d = f32_to_f16(0.01f);
            std::memcpy(w.data() + bl * q.bytes, &d, 2);
        }
        Mat m{q.type, w.data(), rows, cols};
        cuda_pool_reset();
        const double ns = time_matvec(&matvec_cuda, m, x, out, iters);
        WARN("cuda " << q.name << " 4096x4096: " << ns / 1000.0
                     << " us per matvec");
    }
    SUCCEED();
}

// i#701: the CPU prefill attention proper (scores, softmax, value sums;
// no projections) for one layer of Llama 3.2 1B (32 query heads, 8 KV
// heads, head size 64) over a whole prompt, in the 64-token chunks the
// CPU prefill feeds gqa_attend_batch. Multiply by the layer count for a
// model's share.
TEST_CASE("bench: cpu prefill attention per layer", "[.][bench]") {
    constexpr std::uint32_t heads = 32, kv_heads = 8, hd = 64,
                            group = heads / kv_heads, chunk = 64;
    constexpr std::size_t kv_dim = std::size_t{kv_heads} * hd;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (const std::uint32_t n_pos : {2177u, 8702u}) {
        std::vector<float> k(std::size_t{n_pos} * kv_dim),
            v(std::size_t{n_pos} * kv_dim),
            q(std::size_t{chunk} * heads * hd),
            out(std::size_t{chunk} * heads * hd);
        for (auto* buf : {&k, &v, &q}) {
            for (auto& x : *buf) {
                x = d(rng);
            }
        }
        // The 4-wide build, then (on an AVX CPU) the 8-wide one.
        for (const bool off : {true, false}) {
            locus::model::attend_avx_off() = off;
            if (!off && !locus::model::attend_avx()) {
                break;
            }
            const auto t0 = std::chrono::steady_clock::now();
            for (std::uint32_t base = 0; base < n_pos; base += chunk) {
                const std::uint32_t n = std::min(chunk, n_pos - base);
                locus::model::gqa_attend_batch(
                    q.data(), out.data(), std::size_t{heads} * hd, n, base,
                    heads, hd, group, 0.125f,
                    [&](std::uint32_t p) { return k.data() + p * kv_dim; },
                    [&](std::uint32_t p) { return v.data() + p * kv_dim; });
            }
            const double s = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count();
            WARN((off ? "4-wide" : "avx") << " prefill attention, one layer, "
                                          << n_pos << " positions: " << s
                                          << " s");
        }
        locus::model::attend_avx_off() = false;
    }
    // The softmax's share: one row per (query head, position) as the
    // prefill runs it, timed on 1/64 of the 8,702-position rows on one
    // thread and scaled to the layer and the pool; then the same for
    // the LOCUS_CPU_FAST softmax (i#705).
    for (const bool fast : {false, true}) {
        constexpr std::uint32_t n_pos = 8702, sample = 64;
        std::vector<float> row(n_pos);
        double total = 0.0;
        for (std::uint32_t p = 0; p < n_pos; p += sample) {
            for (std::uint32_t i = 0; i <= p; ++i) {
                row[i] = d(rng);
            }
            const auto t0 = std::chrono::steady_clock::now();
            for (std::uint32_t h = 0; h < heads; ++h) {
                if (fast) {
                    locus::backend::softmax_fast_inplace({row.data(), p + 1});
                } else {
                    locus::backend::softmax_inplace({row.data(), p + 1});
                }
            }
            total += std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
        }
        const double threads =
            static_cast<double>(locus::model::mt_fanout());
        WARN((fast ? "fast " : "") << "softmax share, one layer, 8702 "
             << "positions: about " << total * sample / threads << " s on "
             << threads << " threads");
    }
    SUCCEED();
}

// i#285: what the HTTP metrics add to each answered request: the route
// lookup (the server's 11 routes, the path last in the list) and one
// record() (a code count, three histogram observations).
TEST_CASE("bench: http metrics per request", "[.][bench]") {
    locus::http::HttpMetrics m({"/metrics", "/health", "/v1/models",
                                "/v1/models/", "/v1/completions",
                                "/v1/chat/completions", "/v1/messages",
                                "/v1/responses", "/v1/embeddings",
                                "/v1/audio/speech",
                                "/v1/images/generations"});
    constexpr int kN = 1'000'000;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kN; ++i) {
        m.record(m.route("/v1/images/generations"), 200, 1234,
                 static_cast<std::uint64_t>(i), 5'000'000);
    }
    const double ns = std::chrono::duration<double, std::nano>(
                          std::chrono::steady_clock::now() - t0)
                          .count() /
                      kN;
    WARN("http metrics: " << ns << " ns per request (route + record)");
    REQUIRE(m.at(m.route("/v1/images/generations")).seconds.snapshot()
                .count == static_cast<std::uint64_t>(kN));
}
