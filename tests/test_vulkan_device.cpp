#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan_forward.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "locus/backend/vulkan/context.hpp"

using locus::backend::vk::VulkanContext;

namespace {

/**
 * @returns The path of this test binary, to exec a fresh child of it;
 *     empty where it is not known.
 */
std::string self_exe() {
#if defined(__linux__)
    return "/proc/self/exe";
#elif defined(__APPLE__)
    std::uint32_t n = 0;
    _NSGetExecutablePath(nullptr, &n);
    std::vector<char> buf(n);
    if (_NSGetExecutablePath(buf.data(), &n) != 0) {
        return {};
    }
    return buf.data();
#else
    return {};
#endif
}

}  // namespace

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

// i#1119: the child of the test below. Hidden, so it runs only when
// named: it builds the backend's State and keeps a KV pool alive, then
// returns, so the process exits with the State still there.
TEST_CASE("vulkan exit child builds the state and keeps a pool", "[.]") {
    float* pool = locus::backend::vulkan_alloc_kv(1024);
    REQUIRE(pool != nullptr);
}

// i#1119: the child of the profile test below. One forward on the
// backend, so LOCUS_VK_PROFILE has a batch to report at exit.
TEST_CASE("vulkan exit child runs one forward", "[.]") {
    const std::string path =
        std::string(LOCUS_SOURCE_DIR) + "/tests/models/stories260K.gguf";
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    model.use_backend(*locus::backend::find_backend("vulkan"));
    std::vector<float> logits(model.hparams().n_vocab);
    auto c = model.make_cache();
    locus::kv::PagedKvCache::Seq s;
    REQUIRE(c.ensure_capacity(s, 1));
    REQUIRE(locus::backend::vulkan_forward(model, 1, c, s, logits));
    c.release(s);
}

namespace {

/** A child's wait status and what it wrote to stderr. */
struct ChildRun {
    int status = 0;
    std::string err;
};

/**
 * Runs the hidden test case `name` in a fresh exec of this binary, with
 * LOCUS_VK_PROFILE=1 set when `profile`, so the child's driver starts
 * clean.
 *
 * @param exe This binary (self_exe()).
 * @param name The test case to run.
 * @param profile Whether to set LOCUS_VK_PROFILE=1 in the child.
 * @returns The wait status and the child's stderr.
 */
ChildRun run_child(const std::string& exe, const char* name, bool profile) {
    int fds[2];
    REQUIRE(::pipe(fds) == 0);
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::close(fds[0]);
        const int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            ::dup2(devnull, 1);
        }
        ::dup2(fds[1], 2);
        if (profile) {
            ::setenv("LOCUS_VK_PROFILE", "1", 1);
        }
        ::execl(exe.c_str(), "locus_tests", name,
                static_cast<char*>(nullptr));
        _exit(127);  // exec failed
    }
    REQUIRE(pid >= 0);
    ::close(fds[1]);
    ChildRun r;
    char buf[4096];
    for (ssize_t n; (n = ::read(fds[0], buf, sizeof buf)) > 0;) {
        r.err.append(buf, static_cast<std::size_t>(n));
    }
    ::close(fds[0]);
    REQUIRE(::waitpid(pid, &r.status, 0) == pid);
    return r;
}

}  // namespace

// i#1119: a process that used the backend exits 0. A static State was
// destroyed from the exit handlers, after the NVIDIA driver's own
// teardown, and every such process died with SIGSEGV after its work.
TEST_CASE("a process that used vulkan exits cleanly", "[vulkan][device]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    const std::string exe = self_exe();
    if (exe.empty()) {
        SKIP("this OS gives no path to the test binary");
    }
    const auto r = run_child(
        exe, "vulkan exit child builds the state and keeps a pool", false);
    CAPTURE(WIFSIGNALED(r.status) ? WTERMSIG(r.status) : 0, r.err);
    REQUIRE(WIFEXITED(r.status));
    REQUIRE(WEXITSTATUS(r.status) == 0);
}

// i#1119: the States are never destroyed, so their reports print from
// an exit handler instead of ~State: LOCUS_VK_PROFILE still prints.
TEST_CASE("LOCUS_VK_PROFILE still reports at exit", "[vulkan][device]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    if (!std::filesystem::exists(std::string(LOCUS_SOURCE_DIR) +
                                 "/tests/models/stories260K.gguf")) {
        SKIP("model not present; run scripts/tests/fetch-test-model.sh");
    }
    const std::string exe = self_exe();
    if (exe.empty()) {
        SKIP("this OS gives no path to the test binary");
    }
    const auto r = run_child(exe, "vulkan exit child runs one forward", true);
    CAPTURE(r.err);
    REQUIRE(WIFEXITED(r.status));
    REQUIRE(WEXITSTATUS(r.status) == 0);
    REQUIRE(r.err.find("vk-profile:") != std::string::npos);
}
