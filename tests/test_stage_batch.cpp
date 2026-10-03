#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
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

void require_eq(const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(a[i] == b[i]);
    }
}

// Warms session i's sequence to position pre[i] by stepping deterministic
// tokens through `stage`, so a later batched step sees ragged positions.
void warm(PipelineStage& stage, std::vector<Seq>& seq,
          const std::vector<int>& pre) {
    for (std::size_t i = 0; i < seq.size(); ++i) {
        for (int j = 0; j < pre[i]; ++j) {
            const auto wt = static_cast<locus::tok::TokenId>(
                1 + ((static_cast<int>(i) + j) % 20));
            stage.step(seq[i],
                       locus::pipeline::make_token(i, j, wt));
        }
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
}  // namespace

// step_batch must be byte-identical to N separate step() calls, with the
// sequences at DIFFERENT (ragged) positions -- the normal case once the
// executor coalesces independent sessions. A full stage ([0,L)) is both
// first and last, so this covers embed + logits + the final-stage advance
// in one shot.
TEST_CASE("step_batch equals N step on a full stage (ragged)",
          "[pipeline][batch]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    const std::uint32_t L = model.hparams().n_layers;
    const int N = 4;
    const std::vector<int> pre = {2, 0, 3, 1};
    const std::vector<locus::tok::TokenId> step_tok = {5, 6, 7, 8};

    std::vector<std::vector<float>> ref(N);
    {
        PipelineStage s(model, 0, L);
        std::vector<Seq> seq(N);
        warm(s, seq, pre);
        for (int i = 0; i < N; ++i) {
            ref[i] = s.step(seq[i],
                            locus::pipeline::make_token(
                                i, pre[i], step_tok[i]))
                         .data;
        }
    }

    std::vector<std::vector<float>> bat(N);
    {
        PipelineStage s(model, 0, L);
        std::vector<Seq> seq(N);
        warm(s, seq, pre);
        std::vector<Message> ins;
        for (int i = 0; i < N; ++i) {
            ins.push_back(locus::pipeline::make_token(i, pre[i],
                                                      step_tok[i]));
        }
        auto b = make_batch(seq, ins);
        std::vector<PipelineStage::BatchOutput> outs;
        s.step_batch(b, outs);
        REQUIRE(outs.size() == static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) {
            REQUIRE(outs[i].ok);
            REQUIRE(outs[i].out.type == MsgType::kLogits);
            REQUIRE(outs[i].out.position == static_cast<std::uint32_t>(
                                                pre[i]));
            bat[i] = outs[i].out.data;
        }
    }

    for (int i = 0; i < N; ++i) {
        require_eq(bat[i], ref[i]);
    }
}

// Two-stage split [0,k)+[k,L): the first stage's step_batch must emit
// activations and advance its non-final sequences; the last stage's
// step_batch must consume those activations and emit logits. Ragged
// positions, byte-exact against serial step() through both stages.
TEST_CASE("step_batch two-stage equals serial step (first + last roles)",
          "[pipeline][batch]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    const std::uint32_t L = model.hparams().n_layers;
    REQUIRE(L >= 2);
    const std::uint32_t k = L / 2 == 0 ? 1 : L / 2;
    const int N = 4;
    const std::vector<int> pre = {1, 3, 0, 2};
    const std::vector<locus::tok::TokenId> step_tok = {5, 6, 7, 8};

    // Warm both stages of a two-stage pipeline for session i to pre[i]:
    // one token advances the first stage (non-final bump) and the last
    // stage (consumes it), keeping their sequences in lockstep.
    auto warm2 = [&](PipelineStage& a, PipelineStage& b,
                     std::vector<Seq>& sa, std::vector<Seq>& sb) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < pre[i]; ++j) {
                const auto wt = static_cast<locus::tok::TokenId>(
                    1 + ((i + j) % 20));
                Message act =
                    a.step(sa[i], locus::pipeline::make_token(i, j, wt));
                b.step(sb[i], act);
            }
        }
    };

    std::vector<std::vector<float>> ref(N);
    {
        PipelineStage a(model, 0, k), b(model, k, L);
        std::vector<Seq> sa(N), sb(N);
        warm2(a, b, sa, sb);
        for (int i = 0; i < N; ++i) {
            Message act = a.step(
                sa[i],
                locus::pipeline::make_token(i, pre[i], step_tok[i]));
            ref[i] = b.step(sb[i], act).data;
        }
    }

    std::vector<std::vector<float>> bat(N);
    {
        PipelineStage a(model, 0, k), b(model, k, L);
        std::vector<Seq> sa(N), sb(N);
        warm2(a, b, sa, sb);

        std::vector<Message> tin;
        for (int i = 0; i < N; ++i) {
            tin.push_back(locus::pipeline::make_token(i, pre[i],
                                                      step_tok[i]));
        }
        auto ba = make_batch(sa, tin);
        std::vector<PipelineStage::BatchOutput> acts;
        a.step_batch(ba, acts);

        std::vector<Message> ain;
        for (int i = 0; i < N; ++i) {
            REQUIRE(acts[i].ok);
            REQUIRE(acts[i].out.type == MsgType::kActivation);
            ain.push_back(std::move(acts[i].out));
        }
        auto bb = make_batch(sb, ain);
        std::vector<PipelineStage::BatchOutput> logs;
        b.step_batch(bb, logs);
        for (int i = 0; i < N; ++i) {
            REQUIRE(logs[i].ok);
            REQUIRE(logs[i].out.type == MsgType::kLogits);
            bat[i] = logs[i].out.data;
        }
    }

    for (int i = 0; i < N; ++i) {
        require_eq(bat[i], ref[i]);
    }
}

// A malformed entry fails only its own slot; the valid entries still run
// and are byte-identical to a solo step().
TEST_CASE("step_batch isolates a bad frame", "[pipeline][batch]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    const std::uint32_t L = model.hparams().n_layers;

    // Solo reference for the two valid tokens (fresh seqs, position 0).
    std::vector<float> r0, r2;
    {
        PipelineStage s(model, 0, L);
        Seq a, b;
        r0 = s.step(a, locus::pipeline::make_token(0, 0, 5)).data;
        r2 = s.step(b, locus::pipeline::make_token(2, 0, 7)).data;
    }

    PipelineStage s(model, 0, L);
    std::vector<Seq> seq(3);
    std::vector<Message> ins;
    ins.push_back(locus::pipeline::make_token(0, 0, 5));
    // Wrong kind for a first stage (expects kToken): fails its own slot.
    ins.push_back(locus::pipeline::make_activation(
        1, 0, std::vector<float>(model.hparams().n_embd, 0.0f)));
    ins.push_back(locus::pipeline::make_token(2, 0, 7));

    auto b = make_batch(seq, ins);
    std::vector<PipelineStage::BatchOutput> outs;
    s.step_batch(b, outs);

    REQUIRE(outs.size() == 3);
    REQUIRE(outs[0].ok);
    REQUIRE_FALSE(outs[1].ok);
    REQUIRE_FALSE(outs[1].err.empty());
    REQUIRE(outs[2].ok);
    require_eq(outs[0].out.data, r0);
    require_eq(outs[2].out.data, r2);
}

namespace {
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
}  // namespace

// End to end: the executor coalesces steps from DISTINCT sessions into
// one batched forward. Whether or not a given run actually batches (it
// depends on how many jobs are queued when the worker wakes), each
// completion must be byte-identical to a solo step() for that session and
// arrive in submit order.
TEST_CASE("StageExecutor batches distinct sessions transparently",
          "[executor][batch]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    const std::uint32_t L = model.hparams().n_layers;
    const int N = 5;
    const std::vector<locus::tok::TokenId> step_tok = {5, 6, 7, 8, 9};

    std::vector<std::vector<float>> ref(N);
    {
        PipelineStage s(model, 0, L);
        std::vector<Seq> seq(N);
        for (int i = 0; i < N; ++i) {
            ref[i] = s.step(seq[i], locus::pipeline::make_token(
                                        i, 0, step_tok[i]))
                         .data;
        }
    }

    int wake[2];
    REQUIRE(::pipe(wake) == 0);
    ::fcntl(wake[1], F_SETFL,
            ::fcntl(wake[1], F_GETFL, 0) | O_NONBLOCK);

    PipelineStage stage(model, 0, L);
    std::vector<Seq> seq(N);
    std::vector<StageExecutor::Completion> done;
    {
        StageExecutor ex(stage, wake[1]);
        for (int i = 0; i < N; ++i) {
            ex.submit_step(100 + i, &seq[i],
                           locus::pipeline::make_token(i, 0,
                                                       step_tok[i]));
        }
        collect(ex, wake[0], done, N);

        std::vector<StageExecutor::Completion> rel;
        for (int i = 0; i < N; ++i) {
            ex.submit_release(100 + i, &seq[i]);
        }
        collect(ex, wake[0], rel, N);
        REQUIRE(rel.size() == static_cast<std::size_t>(N));
    }
    ::close(wake[0]);
    ::close(wake[1]);

    REQUIRE(done.size() == static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        REQUIRE(done[i].session == 100 + i);  // submit order preserved
        REQUIRE_FALSE(done[i].is_release);
        REQUIRE(done[i].ok);
        REQUIRE(done[i].out.type == MsgType::kLogits);
        require_eq(done[i].out.data, ref[i]);
    }
}
