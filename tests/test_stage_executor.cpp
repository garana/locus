#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/transformer.hpp"
#include "locus/pipeline/message.hpp"
#include "locus/pipeline/stage.hpp"
#include "locus/pipeline/stage_executor.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::pipeline::Message;
using locus::pipeline::MsgType;
using locus::pipeline::PipelineStage;
using locus::pipeline::StageExecutor;

namespace {
std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

// Blocks until `want` completions have been drained from `ex`, draining
// on each wake of `wake_r`. Mirrors how the serve loop consumes them:
// level-triggered, so drain all per wake rather than one-per-wake.
void collect(StageExecutor& ex, int wake_r,
             std::vector<StageExecutor::Completion>& out, std::size_t want) {
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

// The executor runs real forward steps off the caller's thread and
// hands back one completion per job, matching what inline step() would
// produce. Driven with an all-in-one stage ([0,L)) so each step yields
// kLogits. Also exercises release().
TEST_CASE("StageExecutor runs steps off-thread and matches inline step",
          "[executor][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const std::uint32_t L = model.hparams().n_layers;
    const auto prompt = tok.encode("Once upon a time", true);

    // Reference: the same tokens through inline step() on a fresh stage.
    std::vector<locus::tok::TokenId> ref_argmax;
    {
        PipelineStage s(model, 0, L);
        locus::kv::PagedKvCache::Seq seq;
        std::uint32_t pos = 0;
        for (auto t : prompt) {
            const Message out =
                s.step(seq, locus::pipeline::make_token(1, pos++, t));
            ref_argmax.push_back(locus::model::argmax(out.data));
        }
    }

    int wake[2];
    REQUIRE(::pipe(wake) == 0);
    // Non-blocking write end, per the executor's contract (a blocking
    // one could stall the worker if the loop fell behind on draining).
    ::fcntl(wake[1], F_SETFL, ::fcntl(wake[1], F_GETFL, 0) | O_NONBLOCK);

    PipelineStage stage(model, 0, L);
    locus::kv::PagedKvCache::Seq seq;
    std::vector<StageExecutor::Completion> done;
    {
        StageExecutor ex(stage, wake[1]);
        // Submit the prompt as sequential steps on one session's seq.
        std::uint32_t pos = 0;
        for (auto t : prompt) {
            ex.submit_step(42, &seq, locus::pipeline::make_token(1, pos++, t));
        }
        collect(ex, wake[0], done, prompt.size());

        REQUIRE(done.size() == prompt.size());
        for (std::size_t i = 0; i < done.size(); ++i) {
            REQUIRE(done[i].session == 42);
            REQUIRE_FALSE(done[i].is_release);
            REQUIRE(done[i].ok);
            REQUIRE(done[i].out.type == MsgType::kLogits);
            REQUIRE(done[i].out.position == static_cast<std::uint32_t>(i));
            // Off-thread result is identical to inline step().
            REQUIRE(locus::model::argmax(done[i].out.data) == ref_argmax[i]);
        }

        // Release the sequence: a release completion comes back.
        std::vector<StageExecutor::Completion> rel;
        ex.submit_release(42, &seq);
        collect(ex, wake[0], rel, 1);
        REQUIRE(rel.size() == 1);
        REQUIRE(rel[0].session == 42);
        REQUIRE(rel[0].is_release);
    }  // executor joins here

    ::close(wake[0]);
    ::close(wake[1]);
}

// A step that throws (position out of lockstep) comes back ok=false with
// the reason, not as a crash -- the serve loop turns this into a logged
// drop.
TEST_CASE("StageExecutor reports a failed step rather than throwing",
          "[executor][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    const std::uint32_t L = model.hparams().n_layers;

    int wake[2];
    REQUIRE(::pipe(wake) == 0);
    // Non-blocking write end, per the executor's contract (a blocking
    // one could stall the worker if the loop fell behind on draining).
    ::fcntl(wake[1], F_SETFL, ::fcntl(wake[1], F_GETFL, 0) | O_NONBLOCK);
    PipelineStage stage(model, 0, L);
    locus::kv::PagedKvCache::Seq seq;
    std::vector<StageExecutor::Completion> done;
    {
        StageExecutor ex(stage, wake[1]);
        // position 5 with an empty seq (n_tokens 0) is out of lockstep.
        ex.submit_step(7, &seq, locus::pipeline::make_token(1, 5, 1));
        collect(ex, wake[0], done, 1);
        REQUIRE(done.size() == 1);
        REQUIRE(done[0].session == 7);
        REQUIRE_FALSE(done[0].ok);
        REQUIRE_FALSE(done[0].err.empty());
    }
    ::close(wake[0]);
    ::close(wake[1]);
}
