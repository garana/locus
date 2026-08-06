#include "catch_amalgamated.hpp"

#include "locus/backend/variants.hpp"

namespace {

// #56: the Vulkan backend keeps a process-lifetime State singleton
// whose weight pool is unbounded (never evicts) and whose GPU-mapped
// KV pools are never freed. A test process that runs many models
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
