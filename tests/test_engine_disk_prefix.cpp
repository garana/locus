#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/engine/disk_prefix_store.hpp"
#include "locus/engine/engine.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::engine::DiskKey;
using locus::engine::DiskPrefixStore;
using locus::engine::Engine;
using locus::engine::Status;
using locus::tok::TokenId;
namespace fs = std::filesystem;

namespace {

std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

/** A fresh directory, removed when the test ends. */
struct TempDir {
    fs::path dir;
    explicit TempDir(const std::string& name)
        : dir(fs::temp_directory_path() /
              (name + "-" + std::to_string(::getpid()))) {
        fs::remove_all(dir);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

/** The disk key and block size for `model`'s default f32 cache. */
DiskKey key_for(const locus::model::TransformerModel& model,
                std::size_t& block_bytes) {
    const auto cache = model.make_cache();
    const auto& g = cache.geometry();
    DiskKey k;
    k.model_hash = 0x5104;
    k.model_size = 1;
    k.backend = "test";
    k.kv_type = g.kv_type;
    k.block_tokens = g.block_tokens;
    k.kv_dim = g.kv_dim;
    k.n_layers = g.n_layers;
    k.layer_base = g.layer_base;
    block_bytes = cache.block_bytes();
    return k;
}

/** A store whose reads wait for `gate` (a slow disk). */
class SlowStore : public DiskPrefixStore {
  public:
    SlowStore(fs::path dir, DiskKey key, std::size_t bb,
              std::shared_future<void> gate)
        : DiskPrefixStore(std::move(dir), std::move(key), bb, 64 << 20),
          gate_(std::move(gate)) {}
    bool read(std::span<const TokenId> prompt, std::uint32_t n_blocks,
              std::span<std::uint8_t* const> dst) override {
        gate_.wait();
        return DiskPrefixStore::read(prompt, n_blocks, dst);
    }

  private:
    std::shared_future<void> gate_;
};

/** Opens a SlowStore's gate once, at the latest on scope exit, so a
 *  failed REQUIRE cannot leave ~Engine joining a blocked read. */
struct Gate {
    std::promise<void> p;
    std::shared_future<void> f = p.get_future().share();
    bool opened = false;
    void open() {
        if (!opened) {
            opened = true;
            p.set_value();
        }
    }
};
struct OpenOnExit {
    Gate& g;
    ~OpenOnExit() { g.open(); }
};

struct Fixture {
    locus::gguf::GgufFile g;
    locus::model::TransformerModel model;
    locus::tok::SpmTokenizer tok;
    std::vector<TokenId> p;  // the prompt that goes to disk
    std::vector<TokenId> q;  // another one, to evict p from RAM
    std::vector<TokenId> r;  // never cached: always computed
    std::vector<TokenId> want;

    Fixture()
        : g(locus::gguf::GgufFile::open(model_path())),
          model(locus::model::TransformerModel::load(g)),
          tok(locus::tok::SpmTokenizer::from_gguf(g)) {
        p = tok.encode(
            "Once upon a time, there was a little girl named Lily who "
            "loved to explore the forest near her home every day. One "
            "day she found a big red ball under a tree.",
            true);
        q = tok.encode(
            "The cat sat on the mat and looked at the bird in the sky. "
            "The bird sang a happy song.",
            true);
        r = tok.encode("A dog ran.", true);
        Engine baseline(model, tok.eos_id(), Engine::Config{});
        const auto id = baseline.submit(p, 20);
        baseline.run_to_completion();
        want = baseline.get(id)->generated;
    }

    Engine::Config config(std::shared_ptr<DiskPrefixStore> store) const {
        Engine::Config cfg;
        cfg.prefix_cache_slots = 1;  // q's entry evicts p's
        cfg.disk_prefix = std::move(store);
        return cfg;
    }

    /** Runs p then q, so p's prefix is written back to the store. */
    void fill(Engine& e, const DiskPrefixStore& store) const {
        e.submit(p, 20);
        e.run_to_completion();
        e.submit(q, 4);
        e.run_to_completion();
        e.disk_prefix_drain();
        REQUIRE(store.find(p) >= 2);
    }
};

}  // namespace

TEST_CASE("engine reads an evicted prefix back from the disk tier",
          "[engine][disk_prefix]") {
    if (!fs::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    Fixture f;
    REQUIRE(f.p.size() > 2 * 16);
    REQUIRE(f.q.size() > 16);
    TempDir t("locus-edp-hit");
    std::size_t bb;
    const DiskKey key = key_for(f.model, bb);
    {
        auto store = std::make_shared<DiskPrefixStore>(t.dir, key, bb,
                                                       64 << 20);
        Engine e(f.model, f.tok.eos_id(), f.config(store));
        f.fill(e, *store);
        const auto id = e.submit(f.p, 20);
        e.run_to_completion();
        REQUIRE(e.get(id)->generated == f.want);
        REQUIRE(e.disk_prefix_hits() == 1);
        REQUIRE(e.disk_prefix_loaded_tokens() >= 32);
        REQUIRE(e.get(id)->reused_prefix_tokens ==
                e.disk_prefix_loaded_tokens());
    }
    // A new engine and store on the same directory: a restart.
    auto store = std::make_shared<DiskPrefixStore>(t.dir, key, bb, 64 << 20);
    Engine e(f.model, f.tok.eos_id(), f.config(store));
    const auto id = e.submit(f.p, 20);
    e.run_to_completion();
    REQUIRE(e.get(id)->generated == f.want);
    REQUIRE(e.disk_prefix_hits() == 1);
}

TEST_CASE("engine falls back to prefill on a bad disk-tier file",
          "[engine][disk_prefix]") {
    if (!fs::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    Fixture f;
    TempDir t("locus-edp-bad");
    std::size_t bb;
    auto store = std::make_shared<DiskPrefixStore>(t.dir, key_for(f.model, bb),
                                                   bb, 64 << 20);
    Engine e(f.model, f.tok.eos_id(), f.config(store));
    f.fill(e, *store);
    const fs::path file = store->path_for(f.p, store->find(f.p));
    {
        // Flip a byte in the last block.
        std::fstream s(file, std::ios::in | std::ios::out | std::ios::binary);
        s.seekg(-1, std::ios::end);
        char c = 0;
        s.read(&c, 1);
        c = static_cast<char>(c ^ 1);
        s.seekp(-1, std::ios::end);
        s.write(&c, 1);
    }
    const auto id = e.submit(f.p, 20);
    e.run_to_completion();
    REQUIRE(e.get(id)->generated == f.want);
    REQUIRE(e.disk_prefix_hits() == 0);
    REQUIRE(e.get(id)->reused_prefix_tokens == 0);
}

TEST_CASE("engine keeps decoding while a request loads from disk",
          "[engine][disk_prefix]") {
    if (!fs::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    Fixture f;
    TempDir t("locus-edp-slow");
    std::size_t bb;
    const DiskKey key = key_for(f.model, bb);
    {
        auto fill = std::make_shared<DiskPrefixStore>(t.dir, key, bb,
                                                      64 << 20);
        Engine e(f.model, f.tok.eos_id(), f.config(fill));
        f.fill(e, *fill);
    }
    Gate gate;
    auto store = std::make_shared<SlowStore>(t.dir, key, bb, gate.f);
    Engine e(f.model, f.tok.eos_id(), f.config(store));
    OpenOnExit opener{gate};
    // r: shorter than a block, so it never loads from disk (the fill
    // engine wrote q too, at shutdown).
    REQUIRE(f.r.size() < 16);
    const auto busy = e.submit(f.r, 40);
    e.step();
    e.step();
    const auto id = e.submit(f.p, 20);
    e.step();
    REQUIRE(e.get(id)->status == Status::kLoading);
    REQUIRE(e.loading());

    SECTION("the other request decodes meanwhile") {
        const auto before = e.get(busy)->generated.size();
        for (int i = 0; i < 5; ++i) {
            e.step();
        }
        REQUIRE(e.get(id)->status == Status::kLoading);
        REQUIRE(e.get(busy)->generated.size() > before);
        gate.open();
        e.run_to_completion();
        REQUIRE(e.get(id)->generated == f.want);
        REQUIRE(e.disk_prefix_hits() == 1);
    }
    SECTION("a cancel leaves the pool as it was") {
        e.cancel(busy);
        REQUIRE(e.loading());
        // Free blocks once the loading request's blocks come back.
        const std::uint32_t held = store->find(f.p);
        const std::uint32_t free_after = e.free_blocks() + held;
        e.cancel(id);
        REQUIRE(e.get(id)->status == Status::kCancelled);
        // The read still runs: the driver must keep polling for it.
        REQUIRE(e.loading());
        REQUIRE(e.free_blocks() == free_after - held);  // still held
        gate.open();
        e.disk_prefix_drain();
        REQUIRE(e.has_runnable_work());  // the finished read
        e.step();  // takes the finished read and releases its blocks
        REQUIRE(e.free_blocks() == free_after);
        REQUIRE_FALSE(e.loading());
        REQUIRE(e.idle());
    }
}

TEST_CASE("engine parks when loads fill max_running",
          "[engine][disk_prefix]") {
    if (!fs::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    Fixture f;
    TempDir t("locus-edp-cap");
    std::size_t bb;
    const DiskKey key = key_for(f.model, bb);
    {
        auto fill = std::make_shared<DiskPrefixStore>(t.dir, key, bb,
                                                      64 << 20);
        Engine e(f.model, f.tok.eos_id(), f.config(fill));
        f.fill(e, *fill);
    }
    Gate gate;
    auto store = std::make_shared<SlowStore>(t.dir, key, bb, gate.f);
    Engine::Config cfg = f.config(store);
    cfg.max_running = 1;
    Engine e(f.model, f.tok.eos_id(), cfg);
    OpenOnExit opener{gate};
    const auto id = e.submit(f.p, 20);
    e.step();
    REQUIRE(e.get(id)->status == Status::kLoading);
    const auto next = e.submit(f.q, 4);
    e.step();
    REQUIRE(e.get(next)->status == Status::kWaiting);
    // Nothing step() can do until the read ends: a driver that spun
    // here would hold its lock for the whole read.
    REQUIRE_FALSE(e.has_runnable_work());
    REQUIRE(e.loading());
    gate.open();
    e.run_to_completion();
    REQUIRE(e.get(id)->generated == f.want);
    REQUIRE(e.get(next)->status == Status::kDone);
    REQUIRE(e.idle());
}

TEST_CASE("engine refuses a disk tier whose key does not match",
          "[engine][disk_prefix]") {
    if (!fs::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    TempDir t("locus-edp-key");
    std::size_t bb;
    DiskKey key = key_for(model, bb);
    key.kv_dim += 1;
    Engine::Config cfg;
    cfg.disk_prefix =
        std::make_shared<DiskPrefixStore>(t.dir, key, bb, 1 << 20);
    REQUIRE_THROWS_AS(Engine(model, tok.eos_id(), cfg), std::invalid_argument);
    key.kv_dim -= 1;
    cfg.disk_prefix =
        std::make_shared<DiskPrefixStore>(t.dir, key, bb + 4, 1 << 20);
    REQUIRE_THROWS_AS(Engine(model, tok.eos_id(), cfg), std::invalid_argument);
}
