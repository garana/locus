#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/socket.h>
#include <unistd.h>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/llama.hpp"
#include "locus/pipeline/message.hpp"
#include "locus/pipeline/stage.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::pipeline::Message;
using locus::pipeline::MsgType;
using locus::pipeline::PipelineStage;
using locus::pipeline::ReadResult;

namespace {

std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

}  // namespace

// Multi-server (#71) increment 3: chaining PipelineStage workers, each
// owning a layer slice and its own KV cache, connected by sockets, must
// reproduce a single-process generation byte-for-byte. This is the
// in-process proof of the whole stage protocol (kToken in, kActivation
// between stages, kLogits out) and per-stage position advancement,
// before real TCP / separate processes.
TEST_CASE("pipeline stages reproduce single-process generation",
          "[pipeline][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const std::uint32_t L = model.hparams().n_layers;
    const std::uint32_t V = model.hparams().n_vocab;
    REQUIRE(L >= 2);
    const auto prompt =
        tok.encode("Once upon a time, there was a little", true);
    constexpr int kGen = 12;

    // Single-process reference.
    auto mono = [&]() {
        auto cache = model.make_cache();
        auto ws = model.make_workspace();
        locus::kv::PagedKvCache::Seq seq;
        std::vector<float> logits(V);
        for (auto t : prompt) {
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(t, cache, seq, ws, logits);
        }
        std::vector<locus::tok::TokenId> gen;
        for (int i = 0; i < kGen; ++i) {
            const auto n = locus::model::argmax(logits);
            gen.push_back(n);
            if (n == tok.eos_id()) {
                break;
            }
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(n, cache, seq, ws, logits);
        }
        return gen;
    };
    const auto ref = mono();
    REQUIRE(ref.size() >= 1);

    // Runs generation across the stages defined by `bounds` (0..L),
    // each stage on its own thread, connected by socketpairs. Returns
    // the generated tokens (empty on a flow error).
    auto pipeline = [&](const std::vector<std::uint32_t>& bounds) {
        const std::size_t N = bounds.size() - 1;
        std::vector<std::unique_ptr<PipelineStage>> stages;
        for (std::size_t i = 0; i < N; ++i) {
            stages.push_back(std::make_unique<PipelineStage>(
                model, bounds[i], bounds[i + 1]));
        }
        // Per-stage in/out fds; socketpair convention: [1] writer,
        // [0] reader.
        std::vector<int> in_fd(N), out_fd(N);
        int entry_w = -1, entry_r = -1;
        {
            int p[2];
            REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, p) == 0);
            entry_w = p[1];
            in_fd[0] = p[0];
        }
        for (std::size_t i = 0; i + 1 < N; ++i) {
            int p[2];
            REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, p) == 0);
            out_fd[i] = p[1];
            in_fd[i + 1] = p[0];
        }
        {
            int p[2];
            REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, p) == 0);
            out_fd[N - 1] = p[1];
            entry_r = p[0];
        }

        std::vector<int> results(N, 0);
        std::vector<std::thread> th;
        for (std::size_t i = 0; i < N; ++i) {
            th.emplace_back([&, i] {
                results[i] =
                    stages[i]->run(in_fd[i], out_fd[i]) ? 1 : 0;
            });
        }

        // Entry: send one token, read one logits, repeat. Strictly
        // serial (one token in flight), which is all functional
        // correctness needs.
        bool flow = true;
        std::uint32_t pos = 0;
        std::vector<float> logits;
        auto round_trip = [&](locus::tok::TokenId t) {
            const Message in = locus::pipeline::make_token(1, pos, t);
            if (!locus::pipeline::write_message(entry_w, in)) {
                flow = false;
                return;
            }
            Message lg;
            if (locus::pipeline::read_message(entry_r, lg) !=
                    ReadResult::kOk ||
                lg.type != MsgType::kLogits) {
                flow = false;
                return;
            }
            logits = std::move(lg.data);
            ++pos;
        };

        std::vector<locus::tok::TokenId> gen;
        for (auto t : prompt) {
            round_trip(t);
            if (!flow) {
                break;
            }
        }
        for (int i = 0; i < kGen && flow; ++i) {
            const auto n = locus::model::argmax(logits);
            gen.push_back(n);
            if (n == tok.eos_id()) {
                break;
            }
            round_trip(n);
        }

        ::close(entry_w);  // EOF cascades down the chain
        for (auto& t : th) {
            t.join();
        }
        ::close(entry_r);

        REQUIRE(flow);
        for (std::size_t i = 0; i < N; ++i) {
            REQUIRE(results[i] == 1);  // every stage exited cleanly
        }
        return gen;
    };

    SECTION("two stages, every split point") {
        for (std::uint32_t k = 1; k < L; ++k) {
            CAPTURE(k);
            REQUIRE(pipeline({0, k, L}) == ref);
        }
    }

    SECTION("three stages (a middle stage relays hidden states)") {
        if (L >= 3) {
            REQUIRE(pipeline({0, 1, L - 1, L}) == ref);
        }
    }
}
