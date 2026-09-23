#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/llama.hpp"
#include "locus/pipeline/driver.hpp"
#include "locus/pipeline/net.hpp"
#include "locus/tok/tokenizer.hpp"

namespace {

std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

// Reserves a free loopback TCP port (bind to :0, read the ephemeral
// port, release it). A small race window exists before the stage
// process rebinds it; this matches the existing pipeline tests that
// reserve-then-release a port the same way.
int reserve_port() {
    int p = 0;
    const int l = locus::pipeline::listen_on("127.0.0.1", 0, &p);
    REQUIRE(l >= 0);
    ::close(l);
    return p;
}

// fork/execs the real locus-stage binary for layers [a, b), listening
// on listen_port and forwarding to downstream_port, serving one session
// then exiting. The child's stdout is discarded and its stderr is
// redirected to `stderr_path` so a failure can be diagnosed (the caller
// dumps it via INFO).
pid_t spawn_stage(std::uint32_t a, std::uint32_t b, int listen_port,
                  int downstream_port, const std::string& stderr_path) {
    const std::string model = model_path();
    const std::string layers =
        std::to_string(a) + ":" + std::to_string(b);
    const std::string listen =
        std::string("127.0.0.1:") + std::to_string(listen_port);
    const std::string down =
        std::string("127.0.0.1:") + std::to_string(downstream_port);

    const pid_t pid = ::fork();
    if (pid == 0) {
        const int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            ::dup2(devnull, 1);
        }
        const int errfd =
            ::open(stderr_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (errfd >= 0) {
            ::dup2(errfd, 2);
        }
        ::execl(LOCUS_STAGE_BIN, "locus-stage", "--model", model.c_str(),
                "--layers", layers.c_str(), "--listen", listen.c_str(),
                "--downstream", down.c_str(), "--sessions", "1",
                static_cast<char*>(nullptr));
        _exit(127);  // exec failed
    }
    REQUIRE(pid >= 0);  // parent only; the child exec'd or _exit'd above
    return pid;
}

// SIGKILLs and reaps any still-running stage children on destruction,
// so a Catch REQUIRE that aborts the test before the normal wait cannot
// leave orphaned locus-stage processes behind. release() drops a child
// that was already reaped.
struct StageReaper {
    std::vector<pid_t> pids;
    ~StageReaper() {
        for (const pid_t p : pids) {
            if (p > 0) {
                ::kill(p, SIGKILL);
                ::waitpid(p, nullptr, 0);
            }
        }
    }
    void release(pid_t p) {
        for (auto& x : pids) {
            if (x == p) {
                x = -1;
            }
        }
    }
};

// Reads a file into a string (empty if unreadable); for dumping a
// child's captured stderr into the test log on failure.
std::string slurp(const std::string& path) {
    std::string out;
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return out;
    }
    char buf[4096];
    for (;;) {
        const ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n <= 0) {
            break;
        }
        out.append(buf, static_cast<std::size_t>(n));
    }
    ::close(fd);
    return out;
}

// Waits for a child up to timeout_ms, then SIGKILLs and reaps it if it
// overran. @returns the exit code, or -1 if it had to be killed.
int wait_or_kill(pid_t pid, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    for (;;) {
        int status = 0;
        const pid_t r = ::waitpid(pid, &status, WNOHANG);
        if (r == pid) {
            return WIFEXITED(status) ? WEXITSTATUS(status) : -2;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            ::kill(pid, SIGKILL);
            ::waitpid(pid, nullptr, 0);
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

// connect_to with retry until a deadline: the stage process needs a
// moment to bind its listener and load the model.
int connect_retry(const std::string& host, int port, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    for (;;) {
        const int fd = locus::pipeline::connect_to(host, port, 200);
        if (fd >= 0) {
            return fd;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

}  // namespace

// Multi-server (i#17) increment: the lean driver (drive_generation, the
// core of locus-driver) drives a chain of REAL, separate locus-stage
// processes over loopback TCP -- the first genuine multi-process test,
// not in-process threads -- and must reproduce single-process greedy
// generation byte-for-byte. Proves the whole path: the CLI-spawned
// stages wire up (driver -> stage0 -> stage1 -> driver), the driver
// sends tokens and samples returned logits, and per-stage KV caches in
// separate address spaces still yield the exact single-process result.
TEST_CASE("locus-driver drives a multi-process stage chain",
          "[pipeline][e2e][mp]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    if (!std::filesystem::exists(LOCUS_STAGE_BIN)) {
        SKIP("locus-stage binary not built");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok_ptr = locus::tok::tokenizer_from_gguf(g);
    auto& tok = *tok_ptr;
    const std::uint32_t L = model.hparams().n_layers;
    const std::uint32_t V = model.hparams().n_vocab;
    REQUIRE(L >= 2);
    const auto prompt =
        tok.encode("Once upon a time, there was a little", true);
    constexpr std::uint32_t kGen = 12;

    // Single-process greedy reference.
    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(V);
    for (auto t : prompt) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(t, cache, seq, ws, logits);
    }
    std::vector<locus::tok::TokenId> ref;
    for (std::uint32_t i = 0; i < kGen; ++i) {
        const auto n = locus::model::argmax(logits);
        ref.push_back(n);
        if (n == tok.eos_id()) {
            break;
        }
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(n, cache, seq, ws, logits);
    }

    const std::uint32_t k = L / 2 > 0 ? L / 2 : 1;  // split point

    // The driver listens for the tail stage's logits first (so its port
    // is known before the stages start), then spawns the two stages.
    int pe = 0;
    const int le = locus::pipeline::listen_on("127.0.0.1", 0, &pe);
    REQUIRE(le >= 0);
    const int p0 = reserve_port();
    const int p1 = reserve_port();

    const std::string tmp = std::filesystem::temp_directory_path();
    const std::string err0 =
        tmp + "/locus_stage0_" + std::to_string(pe) + ".log";
    const std::string err1 =
        tmp + "/locus_stage1_" + std::to_string(pe) + ".log";

    StageReaper reaper;  // kills+reaps the children on any exit path
    const pid_t s0 = spawn_stage(0, k, p0, p1, err0);
    const pid_t s1 = spawn_stage(k, L, p1, pe, err1);
    reaper.pids = {s0, s1};

    // Connect to the head (triggers the chain to wire up), then accept
    // the tail's logits connection. Both waits are bounded so a
    // regression FAILS the test instead of hanging until the CI timeout.
    const int head_fd = connect_retry("127.0.0.1", p0, 15000);
    const int logits_fd =
        head_fd >= 0 ? locus::pipeline::accept_one(le, nullptr, 15000)
                     : -1;
    ::close(le);

    locus::pipeline::DriverResult res;
    if (head_fd >= 0 && logits_fd >= 0) {
        locus::pipeline::DriverParams params;  // greedy/argmax default
        params.max_tokens = kGen;
        res = locus::pipeline::drive_generation(head_fd, logits_fd,
                                                prompt, tok.eos_id(),
                                                params);
        ::close(head_fd);  // EOF cascades: both stages finish and exit
        ::close(logits_fd);
    }
    const int r0 = wait_or_kill(s0, 10000);
    reaper.release(s0);
    const int r1 = wait_or_kill(s1, 10000);
    reaper.release(s1);

    // On any failure below, the captured stage stderr explains why.
    INFO("stage0 stderr:\n" << slurp(err0));
    INFO("stage1 stderr:\n" << slurp(err1));
    std::filesystem::remove(err0);
    std::filesystem::remove(err1);

    REQUIRE(head_fd >= 0);        // head stage came up and accepted
    REQUIRE(logits_fd >= 0);      // tail stage connected back in time
    CHECK(r0 == 0);               // stage 0 process exited cleanly
    CHECK(r1 == 0);               // stage 1 process exited cleanly
    CHECK(res.complete);          // generation was not truncated
    REQUIRE(res.tokens == ref);   // byte-exact vs single-process greedy
}
