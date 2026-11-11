#include <stdexcept>
#include <string>

#include "catch_amalgamated.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan/context.hpp"

using locus::backend::vk::VulkanContext;

// i#572: the devices with a compute queue are listed without opening
// one, and a context opens on any of them by index.
TEST_CASE("vulkan lists its devices and opens each by index",
          "[vulkan][device]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    const auto devs = locus::backend::vulkan_devices();
    REQUIRE_FALSE(devs.empty());
    for (std::size_t i = 0; i < devs.size(); ++i) {
        CAPTURE(i, devs[i].name, devs[i].type);
        REQUIRE_FALSE(devs[i].name.empty());
        REQUIRE_FALSE(devs[i].type.empty());
        VulkanContext ctx(static_cast<int>(i));
        REQUIRE(ctx.device_name() == devs[i].name);
    }
    // The default is the first one.
    VulkanContext first;
    REQUIRE(first.device_name() == devs[0].name);
    // An index past the list fails loudly, never a silent pick.
    REQUIRE_THROWS_AS(VulkanContext(static_cast<int>(devs.size())),
                      std::out_of_range);
    REQUIRE_THROWS_AS(
        locus::backend::vulkan_set_device(static_cast<int>(devs.size())),
        std::out_of_range);
    REQUIRE_THROWS_AS(locus::backend::vulkan_set_device(-1),
                      std::out_of_range);
}

TEST_CASE("vulkan_set_device cannot move a running backend",
          "[vulkan][device]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    const auto devs = locus::backend::vulkan_devices();
    // Open the backend on whatever device it runs on now.
    const std::string running = locus::backend::vulkan_device_name();
    REQUIRE_FALSE(running.empty());
    std::size_t at = 0;
    while (at < devs.size() && devs[at].name != running) {
        ++at;
    }
    REQUIRE(at < devs.size());
    // Naming the device it runs on is fine; another one throws.
    locus::backend::vulkan_set_device(static_cast<int>(at));
    if (devs.size() < 2) {
        SKIP("one Vulkan device: nothing to move to");
    }
    const int other = at == 0 ? 1 : 0;
    REQUIRE_THROWS_AS(locus::backend::vulkan_set_device(other),
                      std::runtime_error);
    REQUIRE(locus::backend::vulkan_device_name() == running);
}
