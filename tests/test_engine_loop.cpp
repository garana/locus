#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/engine/engine.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/llama.hpp"
#include "locus/server/engine_loop.hpp"
#include "locus/tok/tokenizer.hpp"

namespace {
std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}
}  // namespace

// R1 regression (#59): the worker must release mu_ between step()s
// during active generation so client threads see progress within one
// iteration -- the "block for at most one iteration" contract in the
// EngineLoop class doc. The pre-fix bug held mu_ across the WHOLE
// generation, releasing it only on the idle park, so a client blocked
// in wait_progress could not re-acquire mu_ to observe a new token
// until the request was already kDone: exactly ONE observation, i.e.
// SSE arrives all-at-once at the end. The fix yields per-token
// streaming (many observations).
//
// (view()/timing can't test this: view() can race in during the
// worker's wake-up window before it re-acquires mu_, returning fast
// regardless of the bug. wait_progress genuinely blocks -- it waits on
// progress_cv_ and must re-lock mu_ to see each new token.)
TEST_CASE("EngineLoop streams progress incrementally, not all-at-once",
          "[e2e][engine_loop]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("stories260K model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    locus::engine::Engine::Config cfg;
    locus::server::EngineLoop loop(model, tok.eos_id(), cfg);

    const auto prompt = tok.encode("Once upon a time", true);
    // Warm up so weights are paged in (a cold first step is slow and
    // would let the observer lag); then a long generation gives a wide
    // window to observe streaming.
    loop.wait_done(loop.submit(prompt, 4));

    const std::uint32_t max_new = 256;
    const std::uint64_t id = loop.submit(prompt, max_new);

    // Streaming observer: block for each next token, count distinct
    // progress points before the terminal state.
    std::size_t seen = 0;
    int observations = 0;
    for (;;) {
        const auto v = loop.wait_progress(id, seen);
        ++observations;
        seen = v.generated.size();
        if (v.status == locus::engine::Status::kDone ||
            v.status == locus::engine::Status::kFailed) {
            break;
        }
    }
    INFO("observations=" << observations << " tokens=" << seen);
    // Pre-fix: exactly 1 (only observable once done). Post-fix: many.
    REQUIRE(observations > 1);
}
