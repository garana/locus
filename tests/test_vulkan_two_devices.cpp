#include <algorithm>
#include <filesystem>
#include <latch>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"

using locus::model::TransformerModel;

namespace {

std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

/** Greedy-decodes `n` tokens from `start` on the calling thread's
 *  Vulkan device, with a KV cache made there. */
std::vector<int> decode(const TransformerModel& m, int start, int n) {
    auto cache = m.make_cache();
    auto ws = m.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(m.hparams().n_vocab);
    std::vector<int> out;
    int t = start;
    for (int i = 0; i < n; ++i) {
        // A throw, not a REQUIRE: this runs on a worker thread.
        if (!cache.ensure_capacity(seq, 1)) {
            throw std::runtime_error("KV capacity");
        }
        m.forward(t, cache, seq, ws, logits);
        t = static_cast<int>(std::max_element(logits.begin(), logits.end()) -
                             logits.begin());
        out.push_back(t);
    }
    return out;
}

}  // namespace

// i#573 (the #47 exit test): two threads bound to two Vulkan devices run
// the same model at the same time, each on its own device's state, and
// agree on the greedy tokens.
TEST_CASE("two threads on two vulkan devices decode independently",
          "[vulkan][device]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    if (locus::backend::vulkan_devices().size() < 2) {
        SKIP("needs two Vulkan devices (e.g. a GPU and llvmpipe)");
    }
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = TransformerModel::load(g);
    model.use_backend(*locus::backend::find_backend("vulkan"));

    const auto devs = locus::backend::vulkan_devices();
    std::vector<int> on[2];
    std::string err[2], name[2];
    int bound[2] = {-1, -1};
    std::size_t pools[2] = {0, 0};
    // Both threads start decoding only once both States exist, so the
    // decodes run at the same time.
    std::latch ready(2);
    {
        std::vector<std::jthread> threads;
        for (int d = 0; d < 2; ++d) {
            threads.emplace_back([&, d] {
                try {
                    locus::backend::vulkan_bind_thread(d);
                    bound[d] = locus::backend::vulkan_thread_device();
                    // The device this thread's state runs on.
                    name[d] = locus::backend::vulkan_device_name();
                    ready.arrive_and_wait();
                    auto cache = model.make_cache();
                    pools[d] = locus::backend::vulkan_kv_pool_count();
                    on[d] = decode(model, 1, 96);
                } catch (const std::exception& e) {
                    err[d] = e.what();
                }
            });
        }
    }
    for (int d = 0; d < 2; ++d) {
        CAPTURE(d, err[d]);
        REQUIRE(err[d].empty());
        REQUIRE(bound[d] == d);
        REQUIRE(name[d] == devs[static_cast<std::size_t>(d)].name);
        REQUIRE(pools[d] >= 1);  // this device's own KV pool
        REQUIRE(on[d].size() == 96);
    }
    REQUIRE(on[0] == on[1]);
}

TEST_CASE("a vulkan KV cache cannot run on another device",
          "[vulkan][device]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    if (locus::backend::vulkan_devices().size() < 2) {
        SKIP("needs two Vulkan devices (e.g. a GPU and llvmpipe)");
    }
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = TransformerModel::load(g);
    model.use_backend(*locus::backend::find_backend("vulkan"));
    std::string err;
    bool capacity = false;
    std::size_t pools_before = 0, pools_after = 0;
    std::jthread([&] {
        locus::backend::vulkan_bind_thread(0);
        pools_before = locus::backend::vulkan_kv_pool_count();
        auto cache = model.make_cache();  // device 0's pool
        auto ws = model.make_workspace();
        locus::kv::PagedKvCache::Seq seq;
        capacity = cache.ensure_capacity(seq, 1);
        std::vector<float> logits(model.hparams().n_vocab);
        locus::backend::vulkan_bind_thread(1);
        try {
            model.forward(1, cache, seq, ws, logits);
        } catch (const std::runtime_error& e) {
            err = e.what();
        }
        // Still bound to device 1: the cache is freed on device 0, its
        // owner, from here.
    }).join();
    std::jthread([&] {
        locus::backend::vulkan_bind_thread(0);
        pools_after = locus::backend::vulkan_kv_pool_count();
    }).join();
    REQUIRE(pools_after == pools_before);
    REQUIRE(capacity);
    CAPTURE(err);
    REQUIRE(err.find("made on device 0") != std::string::npos);
}

TEST_CASE("vulkan_bind_thread refuses a device it does not have",
          "[vulkan][device]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    const int n = static_cast<int>(locus::backend::vulkan_devices().size());
    REQUIRE_THROWS_AS(locus::backend::vulkan_bind_thread(n),
                      std::out_of_range);
    REQUIRE_THROWS_AS(locus::backend::vulkan_bind_thread(-1),
                      std::out_of_range);
}
