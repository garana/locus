#include <filesystem>
#include <string>
#include <utility>

#include "catch_amalgamated.hpp"

#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"

namespace {

// #56: the Vulkan backend keeps a process-lifetime State singleton
// whose weight pool is unbounded (never evicts); its GPU-mapped KV
// pools now go with their caches (i#428), but the weights stay. A test process that runs many models
// across many cases therefore accumulates device allocations until
// V3DV (or any Vulkan device) runs out -- surfacing as
// "vulkan: allocate memory" in a later case (deterministic on the
// 4GB Pi, a rare 1-assertion-short flake on 32GB MoltenVK). Free
// that state after EVERY test case so allocations do not pile up
// across the run; a listener covers new vulkan tests automatically.
// Mirrors the CUDA reset (505805d). Test lifecycle only -- in
// production the resident pool is the intended per-process working
// set, so the reset is not wired into forward()/make_cache().
struct VulkanResetListener : Catch::EventListenerBase {
    using Catch::EventListenerBase::EventListenerBase;

    void testCaseEnded(Catch::TestCaseStats const&) override {
        if (locus::backend::vulkan_backend_usable()) {
            locus::backend::vulkan_reset_state();
        }
    }
};

}  // namespace

CATCH_REGISTER_LISTENER(VulkanResetListener)

// i#428: a KV cache built on the Vulkan backend frees its GPU pool when
// it is destroyed (a moved-from cache does not free it twice), so a
// process that builds caches again and again does not run out.
TEST_CASE("vulkan KV pools are freed with their cache", "[vulkan]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device");
    }
    const std::string path =
        std::string(LOCUS_SOURCE_DIR) + "/tests/models/stories260K.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP("stories260K.gguf absent");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    model.use_backend(locus::backend::resolve_backend("vulkan"));
    const std::size_t base = locus::backend::vulkan_kv_pool_count();
    for (int i = 0; i < 20; ++i) {
        auto a = model.make_cache(8);
        REQUIRE(locus::backend::vulkan_kv_pool_count() == base + 1);
        auto b = std::move(a);  // ownership moves; still one pool
        REQUIRE(locus::backend::vulkan_kv_pool_count() == base + 1);
    }
    REQUIRE(locus::backend::vulkan_kv_pool_count() == base);
}
