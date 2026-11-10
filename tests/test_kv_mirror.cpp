// Device mirror of the paged KV pool (i#466): the mirror must hold the
// host pool's bits at the same offsets after every synced write.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/transformer.hpp"

using locus::kv::PagedKvCache;
using Seq = PagedKvCache::Seq;

namespace {

/** Stores left before the fake mirror fails one (negative: never). */
int g_fail_after = -1;

bool fake_store(void* dev, std::size_t off, const float* src,
                std::size_t n) {
    if (g_fail_after == 0) {
        return false;
    }
    if (g_fail_after > 0) {
        --g_fail_after;
    }
    auto& v = *static_cast<std::vector<float>*>(dev);
    REQUIRE(off + n <= v.size());
    std::memcpy(v.data() + off, src, n * sizeof(float));
    return true;
}

void fake_release(void* dev) { delete static_cast<std::vector<float>*>(dev); }

PagedKvCache::Geometry small_geom() {
    PagedKvCache::Geometry g;
    g.n_layers = 2;
    g.kv_dim = 4;
    g.block_tokens = 4;
    g.n_blocks = 8;
    return g;
}

/** Attaches a host-memory mirror and @returns its storage. */
const std::vector<float>& attach_fake(PagedKvCache& cache) {
    auto* v = new std::vector<float>(
        PagedKvCache::pool_floats(cache.geometry()), 0.0f);
    cache.attach_mirror({v, &fake_store, &fake_release});
    return *v;
}

/** Writes distinct K and V values for positions [pos, pos + n). */
void write_rows(PagedKvCache& cache, const Seq& seq, std::uint32_t layer,
                std::uint32_t pos, std::uint32_t n, float base) {
    const std::uint32_t d = cache.geometry().kv_dim;
    for (std::uint32_t p = pos; p < pos + n; ++p) {
        for (std::uint32_t i = 0; i < d; ++i) {
            cache.k(seq, layer, p)[i] = base + p * 10.0f + i;
            cache.v(seq, layer, p)[i] = -(base + p * 10.0f + i);
        }
    }
}

/** @returns true when the mirror holds the host pool's exact bits. */
bool same_bits(const PagedKvCache& cache, const float* mirror) {
    return std::memcmp(cache.pool_data(), mirror,
                       PagedKvCache::pool_floats(cache.geometry()) *
                           sizeof(float)) == 0;
}

}  // namespace

TEST_CASE("kv mirror: synced rows match the host pool", "[kv][mirror]") {
    PagedKvCache cache(small_geom());
    REQUIRE(cache.mirror_data() == nullptr);
    const auto& m = attach_fake(cache);
    REQUIRE(cache.mirror_data() == &m);
    Seq seq;
    REQUIRE(cache.ensure_capacity(seq, 11));

    // A range that crosses two block boundaries (block_tokens 4).
    write_rows(cache, seq, 1, 2, 9, 1.0f);
    REQUIRE_FALSE(same_bits(cache, m.data()));
    cache.sync_rows(seq, 1, 2, 9);
    REQUIRE(same_bits(cache, m.data()));

    // A row rewritten after a rollback (speculative decode) is pushed
    // again by its sync.
    write_rows(cache, seq, 1, 5, 1, 100.0f);
    REQUIRE_FALSE(same_bits(cache, m.data()));
    cache.sync_rows(seq, 1, 5, 1);
    REQUIRE(same_bits(cache, m.data()));

    // Layer 0 rows land at their own offsets.
    write_rows(cache, seq, 0, 0, 3, 7.0f);
    cache.sync_rows(seq, 0, 0, 3);
    REQUIRE(same_bits(cache, m.data()));
}

TEST_CASE("kv mirror: fork pushes the copied tail block", "[kv][mirror]") {
    PagedKvCache cache(small_geom());
    const auto& m = attach_fake(cache);
    Seq parent;
    REQUIRE(cache.ensure_capacity(parent, 6));
    for (std::uint32_t l = 0; l < 2; ++l) {
        write_rows(cache, parent, l, 0, 6, 1.0f + l);
        cache.sync_rows(parent, l, 0, 6);
    }
    parent.n_tokens = 6;
    REQUIRE(same_bits(cache, m.data()));

    Seq child;
    REQUIRE(cache.fork(parent, child));
    REQUIRE(child.blocks[1] != parent.blocks[1]);  // the copied tail
    REQUIRE(same_bits(cache, m.data()));
}

TEST_CASE("kv mirror: a failed store drops the mirror", "[kv][mirror]") {
    PagedKvCache cache(small_geom());
    attach_fake(cache);
    Seq seq;
    REQUIRE(cache.ensure_capacity(seq, 4));
    write_rows(cache, seq, 0, 0, 4, 1.0f);
    g_fail_after = 1;  // the K store succeeds, the V store fails
    cache.sync_rows(seq, 0, 0, 4);
    g_fail_after = -1;
    REQUIRE(cache.mirror_data() == nullptr);
    cache.sync_rows(seq, 0, 0, 4);  // no mirror: a no-op
    REQUIRE(cache.mirror_data() == nullptr);
}

TEST_CASE("kv mirror: attach rejects a quantized cache or a bad mirror",
          "[kv][mirror]") {
    std::vector<float> dev(16);
    {
        auto g = small_geom();
        g.kv_dim = 32;
        g.kv_type = locus::kv::KvType::kQ8;
        PagedKvCache q(g);
        REQUIRE_THROWS_AS(q.attach_mirror({&dev, &fake_store, nullptr}),
                          std::invalid_argument);
    }
    PagedKvCache cache(small_geom());
    REQUIRE_THROWS_AS(cache.attach_mirror({nullptr, &fake_store, nullptr}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(cache.attach_mirror({&dev, nullptr, nullptr}),
                      std::invalid_argument);
    REQUIRE(cache.mirror_data() == nullptr);
}

namespace {

/** @returns the CUDA mirror's contents (whole pool). */
std::vector<float> load_mirror(const PagedKvCache& cache) {
    std::vector<float> out(PagedKvCache::pool_floats(cache.geometry()));
    REQUIRE(locus::backend::cuda_kv_mirror_load(cache.mirror_data(), 0,
                                                out.data(), out.size()));
    return out;
}

}  // namespace

TEST_CASE("kv mirror: the CUDA mirror holds the stored bits",
          "[kv][mirror][cuda]") {
    if (!locus::backend::cuda_backend_usable()) {
        SKIP("no usable CUDA device");
    }
    PagedKvCache cache(small_geom());
    void* dev = locus::backend::cuda_kv_mirror_alloc(
        PagedKvCache::pool_floats(cache.geometry()));
    REQUIRE(dev != nullptr);
    cache.attach_mirror({dev, &locus::backend::cuda_kv_mirror_store,
                         &locus::backend::cuda_kv_mirror_free});
    Seq seq;
    REQUIRE(cache.ensure_capacity(seq, 7));
    std::mt19937 rng(466);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    for (std::uint32_t l = 0; l < 2; ++l) {
        for (std::uint32_t p = 0; p < 7; ++p) {
            for (std::uint32_t i = 0; i < 4; ++i) {
                cache.k(seq, l, p)[i] = ud(rng);
                cache.v(seq, l, p)[i] = ud(rng);
            }
        }
        cache.sync_rows(seq, l, 0, 7);
    }
    REQUIRE(same_bits(cache, load_mirror(cache).data()));
}

TEST_CASE("kv mirror: CUDA forwards keep the mirror equal to the host",
          "[kv][mirror][cuda]") {
    const auto* cuda = locus::backend::find_backend("cuda");
    if (cuda == nullptr || !cuda->available) {
        SKIP("no usable CUDA backend");
    }
    const std::string path =
        std::string(LOCUS_SOURCE_DIR) + "/tests/models/stories260K.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP(path + " absent");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    model.use_backend(*cuda);
    auto cache = model.make_cache(16);
    REQUIRE(cache.mirror_data() != nullptr);
    auto ws = model.make_workspace();
    const std::uint32_t nv = model.hparams().n_vocab;
    std::vector<float> logits(2 * nv);

    // Prefill a prompt, decode on it, fork it, then decode both
    // sequences in one batch: every path that writes K/V rows.
    Seq a;
    const std::vector<locus::tok::TokenId> prompt{1, 403, 407, 261, 89};
    REQUIRE(cache.ensure_capacity(a, 64));
    model.forward_batch(prompt, cache, a, ws, {logits.data(), nv});
    model.forward(17, cache, a, ws, {logits.data(), nv});
    Seq b;
    REQUIRE(cache.fork(a, b));
    REQUIRE(cache.ensure_capacity(b, 8));
    Seq* seqs[] = {&a, &b};
    const std::vector<locus::tok::TokenId> step{23, 42};
    model.forward_batch_decode(step, cache, seqs, ws, logits);
    REQUIRE(a.n_tokens == 7);
    REQUIRE(b.n_tokens == 7);
    REQUIRE(same_bits(cache, load_mirror(cache).data()));
}
