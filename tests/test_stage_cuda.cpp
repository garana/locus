#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/llama.hpp"
#include "locus/pipeline/message.hpp"
#include "locus/pipeline/stage.hpp"
#include "locus/pipeline/stage_executor.hpp"

using locus::pipeline::Message;
using locus::pipeline::MsgType;
using locus::pipeline::PipelineStage;
using locus::pipeline::StageExecutor;
using Seq = locus::kv::PagedKvCache::Seq;

namespace {
std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

// A Q4_K model so matvec_batch_cuda's register-blocked batch kernel
// (Q4_K/Q6_K only; F32 falls back to per-token) actually runs at n > 1.
std::string q4k_model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/llama-3.2-1b-q4_k_m.gguf";
}

std::size_t argmax(const std::vector<float>& v) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return best;
}

void require_bit_eq(const std::vector<float>& a,
                    const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        // Compare the raw bits, not with ==: "byte-exact" must catch a
        // sign-of-zero difference (-0.0f == +0.0f but the bits differ)
        // and must treat two identical NaNs as equal.
        REQUIRE(std::bit_cast<std::uint32_t>(a[i]) ==
                std::bit_cast<std::uint32_t>(b[i]));
    }
}

std::vector<PipelineStage::BatchInput> make_batch(
    std::vector<Seq>& seq, const std::vector<Message>& ins) {
    std::vector<PipelineStage::BatchInput> b;
    b.reserve(seq.size());
    for (std::size_t i = 0; i < seq.size(); ++i) {
        b.push_back({&seq[i], &ins[i]});
    }
    return b;
}

void collect(StageExecutor& ex, int wake_r,
             std::vector<StageExecutor::Completion>& out,
             std::size_t want) {
    while (out.size() < want) {
        char buf[64];
        const ssize_t n = ::read(wake_r, buf, sizeof(buf));
        if (n <= 0) {
            break;
        }
        ex.drain(out);
    }
}

// First non-CUDA backend usable for inference, for the cross-backend
// argmax check (sse4/avx2/neon/scalar -- whichever this host built).
const locus::backend::Backend* some_cpu_backend() {
    for (const auto& b : locus::backend::backends()) {
        if (b.available && b.selectable && b.name != "cuda" &&
            b.name != "vulkan") {
            return &b;
        }
    }
    return nullptr;
}
}  // namespace

// Fail-loud device selection (i#24 inc 4). Runs on EVERY host, CUDA or
// not: cuda_set_device must throw -- never silently fall back to device
// 0 -- for a bad ordinal, and on a CUDA-less host/build even device 0
// must throw. This is the failure an operator actually hits, it needs no
// GPU, so it is real coverage on every machine in the cluster.
TEST_CASE("cuda_set_device fails loud; never silently falls back",
          "[cuda][device]") {
    using locus::backend::cuda_backend_usable;
    using locus::backend::cuda_set_device;
    // A negative ordinal always throws; the exact type varies by BUILD
    // (out_of_range in a CUDA build, runtime_error in the non-CUDA stub),
    // so this one only pins "throws, never a silent no-op".
    REQUIRE_THROWS(cuda_set_device(-1));
    if (cuda_backend_usable()) {
        REQUIRE_NOTHROW(cuda_set_device(0));  // a real device: 0 is valid
        // Past the device count: pin that it is the RANGE check that
        // fires (out_of_range), not some incidental failure.
        REQUIRE_THROWS_AS(cuda_set_device(9999), std::out_of_range);
    } else {
        // No CUDA device / non-CUDA build: the device-count check fires
        // first, so every ordinal is a runtime_error -- never a no-op.
        REQUIRE_THROWS_AS(cuda_set_device(0), std::runtime_error);
        REQUIRE_THROWS_AS(cuda_set_device(9999), std::runtime_error);
    }
}

// Byte-exact on the device at batch width > 1, plus a right-answer check
// against the CPU. A FIRST stage ([0,L), layer_begin == 0) runs the embed
// and the first on-worker device matvec, so the executor's worker thread
// exercises the full on-worker CUDA entry sequence the per-thread device
// bind must cover.
//
// What vx can prove: it has a single GPU, so cross-device binding (inc
// 4b) is not validatable here. What IS validated: the CUDA batched
// forward is BIT-identical to CUDA per-token stepping (same backend) at
// batch > 1, both on the main thread (step_batch) and through the
// executor's worker thread; and its argmax agrees with the CPU path
// (cross-backend float order differs, so logits are not bitwise equal --
// the top token is the right-answer check).
TEST_CASE("CUDA stage executor is byte-exact at batch>1 and agrees w/ CPU",
          "[executor][cuda][batch]") {
    if (!locus::backend::cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    REQUIRE(locus::backend::find_backend("cuda") != nullptr);
    locus::backend::cuda_pool_reset();
    // Select device 0 so g_cuda_device >= 0 and the per-thread bind path
    // (ensure_device's cudaSetDevice + the pool's device-record) actually
    // RUNS -- without this it early-returns and the new code is never
    // exercised. On this single-GPU host the bind is to the default
    // device, so it drives the code path but cannot show cross-device
    // divergence; that (and the pool device-match throw) is inherently a
    // multi-GPU property, validated under inc 4b on a multi-GPU host.
    locus::backend::cuda_set_device(0);

    auto g = locus::gguf::GgufFile::open(model_path());
    const int N = 4;
    const std::vector<locus::tok::TokenId> tok = {5, 6, 7, 8};

    auto m_cuda = locus::model::LlamaModel::load(g);
    m_cuda.use_backend(*locus::backend::find_backend("cuda"));
    const std::uint32_t L = m_cuda.hparams().n_layers;

    // CUDA per-token reference (main thread), full stage (embed+logits).
    std::vector<std::vector<float>> ref(N);
    {
        PipelineStage s(m_cuda, 0, L);
        std::vector<Seq> seq(N);
        for (int i = 0; i < N; ++i) {
            ref[i] = s.step(seq[i],
                            locus::pipeline::make_token(i, 0, tok[i]))
                         .data;
        }
    }

    // (A) step_batch on CUDA: guaranteed batch width N > 1, main thread.
    // Bit-identical to CUDA per-token stepping.
    {
        PipelineStage s(m_cuda, 0, L);
        std::vector<Seq> seq(N);
        std::vector<Message> ins;
        for (int i = 0; i < N; ++i) {
            ins.push_back(locus::pipeline::make_token(i, 0, tok[i]));
        }
        auto b = make_batch(seq, ins);
        std::vector<PipelineStage::BatchOutput> outs;
        s.step_batch(b, outs);
        REQUIRE(outs.size() == static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) {
            REQUIRE(outs[i].ok);
            REQUIRE(outs[i].out.type == MsgType::kLogits);
            require_bit_eq(outs[i].out.data, ref[i]);
        }
    }

    // (B) Through the StageExecutor's worker thread: the device work runs
    // off the main thread, so ensure_device() binds the WORKER thread
    // (not just main). Coalescing is timing-dependent, so this is a
    // threading/worker-bind check, not a guaranteed-batch-width one (part
    // A is that). Still bit-identical to the CUDA per-token reference.
    int wake[2];
    REQUIRE(::pipe(wake) == 0);
    ::fcntl(wake[1], F_SETFL, ::fcntl(wake[1], F_GETFL, 0) | O_NONBLOCK);
    std::vector<StageExecutor::Completion> done;
    {
        PipelineStage stage(m_cuda, 0, L);
        std::vector<Seq> seq(N);
        StageExecutor ex(stage, wake[1]);
        for (int i = 0; i < N; ++i) {
            ex.submit_step(i, &seq[i],
                           locus::pipeline::make_token(i, 0, tok[i]));
        }
        collect(ex, wake[0], done, N);
        std::vector<StageExecutor::Completion> rel;
        for (int i = 0; i < N; ++i) {
            ex.submit_release(i, &seq[i]);
        }
        collect(ex, wake[0], rel, N);
    }
    ::close(wake[0]);
    ::close(wake[1]);
    REQUIRE(done.size() == static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        REQUIRE(done[i].ok);
        REQUIRE(done[i].out.type == MsgType::kLogits);
        require_bit_eq(done[i].out.data, ref[i]);
    }

    // (C) Right-answer check vs the CPU: the device must pick the same
    // next token. Logits are not bitwise equal across backends (float
    // accumulation order differs), so compare argmax, not bits.
    const locus::backend::Backend* cpu = some_cpu_backend();
    REQUIRE(cpu != nullptr);
    auto m_cpu = locus::model::LlamaModel::load(g);
    m_cpu.use_backend(*cpu);
    {
        PipelineStage s(m_cpu, 0, L);
        std::vector<Seq> seq(N);
        for (int i = 0; i < N; ++i) {
            const auto cpu_logits =
                s.step(seq[i], locus::pipeline::make_token(i, 0, tok[i]))
                    .data;
            // Guard against a degenerate match: if logits were all equal,
            // both argmaxes would be 0 and agree vacuously. Require a
            // strict max so the top-token agreement is meaningful.
            REQUIRE(*std::max_element(ref[i].begin(), ref[i].end()) >
                    *std::min_element(ref[i].begin(), ref[i].end()));
            REQUIRE(argmax(cpu_logits) == argmax(ref[i]));
        }
    }
}

// The register-blocked batched CUDA kernel runs only for Q4_K/Q6_K; the
// F32 case above exercises the batch ORCHESTRATION but falls back to the
// per-token kernel (stories260K is F32). Load a Q4_K model so
// matvec_batch_cuda's register-blocked kernel actually executes at batch
// width > 1, and confirm it stays bit-identical to CUDA per-token
// stepping -- the "byte-exact at batch > 1" claim for the quantized
// weights that are the production case.
TEST_CASE("CUDA batched Q4_K matvec is byte-exact vs per-token",
          "[executor][cuda][batch]") {
    if (!locus::backend::cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    if (!std::filesystem::exists(q4k_model_path())) {
        SKIP("Q4_K model not present (llama-3.2-1b-q4_k_m.gguf)");
    }
    locus::backend::cuda_pool_reset();
    locus::backend::cuda_set_device(0);

    auto g = locus::gguf::GgufFile::open(q4k_model_path());
    auto model = locus::model::LlamaModel::load(g);
    model.use_backend(*locus::backend::find_backend("cuda"));
    const std::uint32_t L = model.hparams().n_layers;
    const int N = 3;
    const std::vector<locus::tok::TokenId> tok = {5, 6, 7};

    std::vector<std::vector<float>> ref(N);
    {
        PipelineStage s(model, 0, L);
        std::vector<Seq> seq(N);
        for (int i = 0; i < N; ++i) {
            ref[i] = s.step(seq[i],
                            locus::pipeline::make_token(i, 0, tok[i]))
                         .data;
        }
    }
    {
        PipelineStage s(model, 0, L);
        std::vector<Seq> seq(N);
        std::vector<Message> ins;
        for (int i = 0; i < N; ++i) {
            ins.push_back(locus::pipeline::make_token(i, 0, tok[i]));
        }
        auto b = make_batch(seq, ins);
        std::vector<PipelineStage::BatchOutput> outs;
        s.step_batch(b, outs);  // one forward at width N > 1
        REQUIRE(outs.size() == static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) {
            REQUIRE(outs[i].ok);
            require_bit_eq(outs[i].out.data, ref[i]);
        }
    }
}
