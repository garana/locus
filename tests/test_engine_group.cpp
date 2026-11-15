#include "locus/server/engine_group.hpp"

#include <atomic>
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
#include "locus/model/transformer.hpp"
#include "locus/server/engine_loop.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::server::EngineGroup;
using locus::server::EngineLoop;

namespace {

std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

/** Greedy tokens of each prompt, one at a time on one engine. */
std::vector<std::vector<locus::tok::TokenId>> alone(
    const locus::model::TransformerModel& model, locus::tok::TokenId eos,
    const std::vector<std::vector<locus::tok::TokenId>>& prompts,
    std::uint32_t n) {
    EngineLoop loop(model, eos, {});
    std::vector<std::vector<locus::tok::TokenId>> out;
    for (const auto& p : prompts) {
        const std::uint64_t id = loop.submit(p, n);
        out.push_back(loop.wait_done(id).generated);
        loop.release(id);
    }
    return out;
}

/** Submits both prompts at once (one per engine), then waits. */
std::vector<std::vector<locus::tok::TokenId>> together(
    EngineGroup& group,
    const std::vector<std::vector<locus::tok::TokenId>>& prompts,
    std::uint32_t n, std::vector<std::size_t>& engines) {
    std::vector<std::uint64_t> ids;
    for (const auto& p : prompts) {
        ids.push_back(group.submit(p, n));
        engines.push_back(group.engine_of(ids.back()));
    }
    std::vector<std::vector<locus::tok::TokenId>> out;
    for (const std::uint64_t id : ids) {
        const auto v = group.wait_done(id);
        CHECK(v.status == locus::engine::Status::kDone);
        out.push_back(v.generated);
        group.release(id);
        group.release(id);  // idempotent, as a stream may do
    }
    return out;
}

}  // namespace

TEST_CASE("engine slots: CPU shares, or one device per engine",
          "[engine_group]") {
    const auto one = EngineGroup::plan(1, {});
    REQUIRE(one.size() == 1);
    CHECK(one[0].threads == 0);  // the process pool, as before
    CHECK(one[0].vulkan_device == -1);

    const auto three = EngineGroup::plan(3, {});
    REQUIRE(three.size() == 3);
    for (const auto& s : three) {
        CHECK(s.threads >= 1);
        CHECK(s.vulkan_device == -1);
    }

    const auto two = EngineGroup::plan(2, {1, 0});
    REQUIRE(two.size() == 2);
    CHECK(two[0].vulkan_device == 1);
    CHECK(two[1].vulkan_device == 0);
    CHECK(two[0].threads == 0);

    CHECK_THROWS_AS(EngineGroup::plan(2, {0}), std::invalid_argument);
}

TEST_CASE("two CPU engines give each request the tokens of one engine",
          "[engine_group][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("stories260K model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const std::vector<std::vector<locus::tok::TokenId>> prompts = {
        tok.encode("Once upon a time", true),
        tok.encode("The little dog", true)};
    const auto want = alone(model, tok.eos_id(), prompts, 12);

    EngineGroup group(model, tok.eos_id(), {}, EngineGroup::plan(2, {}));
    REQUIRE(group.size() == 2);
    std::vector<std::size_t> engines;
    const auto got = together(group, prompts, 12, engines);
    CHECK(got == want);
    // Least-loaded: the second request went to the other engine.
    CHECK(engines[0] != engines[1]);
    // Both released: the next request goes to engine 0 again.
    const std::uint64_t id = group.submit(prompts[0], 2);
    CHECK(group.engine_of(id) == 0);
    group.wait_done(id);
    group.release(id);
    // The pools are summed.
    const auto s = group.stats();
    CHECK(s.total_blocks > 0);
    CHECK(s.free_blocks <= s.total_blocks);
}

// i#989: follow and take reach the engine that holds the id, and one
// hook serves both engines' workers.
TEST_CASE("two CPU engines: a followed request on each is taken whole",
          "[engine_group][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("stories260K model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const std::vector<std::vector<locus::tok::TokenId>> prompts = {
        tok.encode("Once upon a time", true),
        tok.encode("The little dog", true)};
    const auto want = alone(model, tok.eos_id(), prompts, 12);

    EngineGroup group(model, tok.eos_id(), {}, EngineGroup::plan(2, {}));
    std::atomic<int> calls{0};
    group.set_progress_hook([&calls] { calls.fetch_add(1); });
    std::vector<std::uint64_t> ids;
    for (const auto& p : prompts) {
        ids.push_back(group.submit(p, 12));
        group.follow(ids.back());
    }
    REQUIRE(group.engine_of(ids[0]) != group.engine_of(ids[1]));
    std::vector<EngineLoop::View> acc(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto v = group.wait_done(ids[i]);
        REQUIRE(v.status == locus::engine::Status::kDone);
        // The terminal step queued everything before wait_done saw it.
        while (group.take(ids[i], acc[i])) {
        }
        CHECK(acc[i].status == locus::engine::Status::kDone);
        CHECK(acc[i].generated == want[i]);
        group.release(ids[i]);
    }
    CHECK(calls.load() > 0);
}

// i#940: each engine has its own prefix cache, so a prompt goes to the
// engine that last served its first tokens (and reuses their KV there),
// unless that engine is busier than the least-loaded one by more than
// the slack (one request).
TEST_CASE("a prompt goes to the engine that holds its prefix",
          "[engine_group][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("stories260K model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    locus::engine::Engine::Config cfg;
    cfg.prefix_cache = true;
    // An EOS the model never emits: the long request runs until the
    // context ends it.
    const locus::tok::TokenId never = model.hparams().n_vocab;
    EngineGroup group(model, never, cfg, EngineGroup::plan(2, {}));
    std::vector<locus::tok::TokenId> shared;
    for (int i = 0; i < 40; ++i) {
        shared.push_back(1 + i % 200);
    }
    const auto with_tail = [&](locus::tok::TokenId t) {
        auto p = shared;
        p.push_back(t);
        return p;
    };
    std::vector<locus::tok::TokenId> other;
    for (int i = 0; i < 20; ++i) {
        other.push_back(300 + i);
    }

    // The prefix's first request: engine 0 (both idle), done, released.
    const std::uint64_t a = group.submit(with_tail(250), 2);
    REQUIRE(group.engine_of(a) == 0);
    group.wait_done(a);
    group.release(a);
    // Engine 0 takes one more request (another prefix), kept unreleased.
    const std::uint64_t x = group.submit(other, 4096);
    REQUIRE(group.engine_of(x) == 0);
    // By load the next would go to engine 1; its prefix sends it to 0,
    // one request over, and its first blocks come from the cache there.
    const std::uint64_t b = group.submit(with_tail(251), 2);
    CHECK(group.engine_of(b) == 0);
    CHECK(group.wait_done(b).reused_prefix_tokens > 0);
    // Engine 0 now holds two over engine 1: past the slack, so engine 1.
    const std::uint64_t c = group.submit(with_tail(252), 2);
    CHECK(group.engine_of(c) == 1);
    group.wait_done(c);
    for (const std::uint64_t id : {c, b}) {
        group.release(id);
    }
    group.cancel(x);
    group.wait_done(x);
    group.release(x);
}

// i#988: chat prompts share their template's first block, so routing by
// that block alone sends every conversation to the engine that took the
// last one. Two conversations with one 16-token preamble each go back to
// the engine that holds their own history (the longest matching prefix).
TEST_CASE("two conversations with one preamble keep their own engines",
          "[engine_group][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("stories260K model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    locus::engine::Engine::Config cfg;
    cfg.prefix_cache = true;
    const locus::tok::TokenId never = model.hparams().n_vocab;
    EngineGroup group(model, never, cfg, EngineGroup::plan(2, {}));
    const auto run = [](int from, int n) {
        std::vector<locus::tok::TokenId> v;
        for (int i = 0; i < n; ++i) {
            v.push_back(from + i);
        }
        return v;
    };
    const auto cat = [](std::vector<std::vector<locus::tok::TokenId>> parts) {
        std::vector<locus::tok::TokenId> v;
        for (const auto& p : parts) {
            v.insert(v.end(), p.begin(), p.end());
        }
        return v;
    };
    const auto pre = run(1, 16);  // the shared preamble: one block
    const auto a1 = run(100, 16), a2 = run(120, 16);
    const auto b1 = run(200, 16), b2 = run(220, 16);
    const auto other = run(300, 20);
    const auto finish = [&](std::uint64_t id) {
        const auto v = group.wait_done(id);
        group.release(id);
        return v;
    };

    // Conversation A's first turn: engine 0 (both idle).
    const std::uint64_t ta1 = group.submit(cat({pre, a1, {50}}), 2);
    REQUIRE(group.engine_of(ta1) == 0);
    finish(ta1);
    // Engine 0 busy by two (the second follows the first's prefix), so
    // conversation B's first turn goes to engine 1.
    const std::uint64_t x1 = group.submit(other, 4096);
    const std::uint64_t x2 = group.submit(cat({other, {330}}), 4096);
    REQUIRE(group.engine_of(x1) == 0);
    REQUIRE(group.engine_of(x2) == 0);
    const std::uint64_t tb1 = group.submit(cat({pre, b1, {51}}), 2);
    REQUIRE(group.engine_of(tb1) == 1);
    finish(tb1);
    for (const std::uint64_t id : {x1, x2}) {
        group.cancel(id);
        finish(id);
    }

    // Both idle. The preamble's block last went to engine 1, but each
    // second turn goes to the engine of its own first turn, and reuses
    // the preamble and its first message from that engine's cache.
    const std::uint64_t ta2 = group.submit(cat({pre, a1, a2, {52}}), 2);
    CHECK(group.engine_of(ta2) == 0);
    CHECK(finish(ta2).reused_prefix_tokens >= 32);
    const std::uint64_t tb2 = group.submit(cat({pre, b1, b2, {53}}), 2);
    CHECK(group.engine_of(tb2) == 1);
    CHECK(finish(tb2).reused_prefix_tokens >= 32);
}

TEST_CASE("two Vulkan devices run one engine each with one device's tokens",
          "[engine_group][vulkan]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("stories260K model not present");
    }
    if (!locus::backend::vulkan_backend_usable() ||
        locus::backend::vulkan_devices().size() < 2) {
        SKIP("needs two Vulkan devices");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    model.use_backend(*locus::backend::find_backend("vulkan"));
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const std::vector<std::vector<locus::tok::TokenId>> prompts = {
        tok.encode("Once upon a time", true),
        tok.encode("The little dog", true)};
    // The reference: prompt i alone on device i (the devices may round
    // differently, e.g. V3D and llvmpipe), each a one-engine group.
    const int before = locus::backend::vulkan_thread_device();
    std::vector<std::vector<locus::tok::TokenId>> want;
    for (int d = 0; d < 2; ++d) {
        EngineGroup one(model, tok.eos_id(), {}, EngineGroup::plan(1, {d}));
        const std::uint64_t id = one.submit(prompts[d], 12);
        want.push_back(one.wait_done(id).generated);
        one.release(id);
    }

    EngineGroup group(model, tok.eos_id(), {}, EngineGroup::plan(2, {0, 1}));
    CHECK(locus::backend::vulkan_thread_device() == before);  // restored
    std::vector<std::size_t> engines;
    const auto got = together(group, prompts, 12, engines);
    // Least-loaded on an idle group: prompt i on engine i, device i.
    REQUIRE(engines == std::vector<std::size_t>{0, 1});
    CHECK(got == want);
}
