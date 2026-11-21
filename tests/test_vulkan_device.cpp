#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan_forward.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "locus/backend/vulkan/context.hpp"

using locus::backend::vk::VulkanContext;

// The process environment, for the exec'd children (no header declares
// it on macOS).
extern char** environ;

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
    // The child's argv and environment are built here, before fork: the
    // binary has other threads, so the child may only make
    // async-signal-safe calls (no malloc, so no setenv) until execve.
    std::vector<std::string> env_s;
    for (char** e = environ; *e != nullptr; ++e) {
        if (std::string_view(*e).starts_with("LOCUS_VK_PROFILE=")) {
            continue;
        }
        env_s.emplace_back(*e);
    }
    if (profile) {
        env_s.emplace_back("LOCUS_VK_PROFILE=1");
    }
    std::vector<char*> envp;
    for (auto& s : env_s) {
        envp.push_back(s.data());
    }
    envp.push_back(nullptr);
    std::string arg0 = "locus_tests", arg1 = name;
    char* argv[] = {arg0.data(), arg1.data(), nullptr};
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
        ::execve(exe.c_str(), argv, envp.data());
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

// i#1121: weights written through mapped() reach the kernel, on a
// discrete GPU through the staging copy into device-local memory, also
// after a batch that recorded the copy was abandoned; and a staged
// buffer destroyed before any copy leaves nothing behind.
TEST_CASE("vulkan weights written through mapped reach the kernel",
          "[vulkan][device]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    using locus::backend::MemKind;
    VulkanContext ctx;
    INFO("device-local weights: " << ctx.device_local_weights());
    const std::uint32_t rows = 37, cols = 64;
    std::vector<float> w(std::size_t{rows} * cols), x(cols);
    for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25f;
    }
    for (std::uint32_t i = 0; i < cols; ++i) {
        x[i] = static_cast<float>(i % 5) - 2.0f;
    }
    // Never copied: destroyed while it is still staged.
    auto dropped = ctx.create_buffer(w.size() * 4, MemKind::kWeights);
    REQUIRE(ctx.mapped(dropped) != nullptr);
    ctx.destroy_buffer(dropped);

    auto wb = ctx.create_buffer(w.size() * 4, MemKind::kWeights);
    std::memcpy(ctx.mapped(wb), w.data(), w.size() * 4);
    auto xb = ctx.create_buffer(x.size() * 4);
    std::memcpy(ctx.mapped(xb), x.data(), x.size() * 4);
    auto ob = ctx.create_buffer(std::size_t{rows} * 4);
    // A batch that records the copy, then is abandoned (an exception
    // out of a forward): the copy never ran, so the next batch must
    // still make it.
    {
        ctx.begin_batch();
        const VulkanContext::Buffer bufs[] = {wb, xb, ob};
        const std::uint32_t push[] = {
            rows, cols, 0, 0, 0, 0, std::bit_cast<std::uint32_t>(1.0f)};
        ctx.dispatch(locus::backend::vk::Kernel::kMatvecF32, bufs, push,
                     VulkanContext::matvec_groups(
                         locus::backend::vk::Kernel::kMatvecF32, rows));
        ctx.abandon_batch();
    }
    // Twice: the first batch records the copy, the second reads the
    // device-local buffer alone.
    for (int pass = 0; pass < 2; ++pass) {
        CAPTURE(pass);
        std::memset(ctx.mapped(ob), 0, std::size_t{rows} * 4);
        ctx.matvec_f32(wb, rows, cols, xb, ob);
        const auto* got = static_cast<const float*>(ctx.mapped(ob));
        for (std::uint32_t r = 0; r < rows; ++r) {
            float want = 0.0f;
            for (std::uint32_t c = 0; c < cols; ++c) {
                want += w[std::size_t{r} * cols + c] * x[c];
            }
            REQUIRE(got[r] == Catch::Approx(want).margin(1e-4));
        }
    }
    for (auto b : {wb, xb, ob}) {
        ctx.destroy_buffer(b);
    }
}
