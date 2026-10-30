#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/video/ffmpeg.hpp"
#include "locus/video/jobs.hpp"

using locus::video::Ffmpeg;
using locus::video::VideoJob;
using locus::video::VideoJobs;
namespace fs = std::filesystem;

namespace {

fs::path temp_dir(const std::string& name) {
    const fs::path d = fs::temp_directory_path() /
                       (name + "-" + std::to_string(::getpid()));
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f),
            std::istreambuf_iterator<char>()};
}

/** Writes an executable shell script. */
void write_script(const fs::path& p, const std::string& body) {
    std::ofstream(p) << "#!/bin/sh\n" << body;
    fs::permissions(p, fs::perms::owner_all);
}

/**
 * A stand-in for ffmpeg: "-encoders" lists the given encoders; an
 * encode writes "fakemp4" to the last argument, the stdin byte count to
 * OUT.size and the arguments, one per line, to OUT.args.
 */
fs::path fake_ffmpeg(const fs::path& dir, const std::string& encoders) {
    const fs::path p = dir / "ffmpeg";
    write_script(p,
                 "if [ \"$2\" = \"-encoders\" ]; then\n"
                 "  printf '" + encoders + "'; exit 0\n"
                 "fi\n"
                 "for a; do out=$a; done\n"
                 "wc -c | tr -d ' ' > \"$out.size\"\n"
                 "printf '%s\\n' \"$@\" > \"$out.args\"\n"
                 "printf fakemp4 > \"$out\"\n");
    return p;
}

std::vector<std::vector<std::uint8_t>> frames(std::uint32_t n,
                                              std::uint32_t w,
                                              std::uint32_t h) {
    std::vector<std::vector<std::uint8_t>> f(n);
    for (std::uint32_t k = 0; k < n; ++k) {
        f[k].resize(std::size_t{w} * h * 3);
        for (std::size_t i = 0; i < f[k].size(); ++i) {
            f[k][i] = static_cast<std::uint8_t>(i * 7 + k * 31);
        }
    }
    return f;
}

}  // namespace

TEST_CASE("run_process feeds stdin and reads both outputs", "[video]") {
    // 4 MiB through cat while it writes back: neither pipe may stall.
    std::vector<std::uint8_t> big(4 << 20);
    for (std::size_t i = 0; i < big.size(); ++i) {
        big[i] = static_cast<std::uint8_t>(i * 13);
    }
    const std::span<const std::uint8_t> parts[] = {
        std::span(big).first(1000), std::span(big).subspan(1000)};
    const auto r = locus::video::run_process(
        {"/bin/sh", "-c", "cat; echo oops >&2; exit 3"}, parts);
    REQUIRE(r.exit_code == 3);
    REQUIRE(r.err == "oops\n");
    REQUIRE(r.out.size() == big.size());
    REQUIRE(std::equal(big.begin(), big.end(),
                       reinterpret_cast<const std::uint8_t*>(r.out.data())));
    // A command that is not there: 127, as a shell gives.
    REQUIRE(locus::video::run_process({"/nonexistent/locus-cmd"}, {})
                .exit_code == 127);
    // A child that stops reading early: the write fails, locus lives.
    const auto early = locus::video::run_process({"/usr/bin/true"}, parts);
    REQUIRE(early.exit_code == 0);
}

// The child keeps only 0, 1 and 2: an fd locus has open (as its
// sockets are, without close-on-exec) is closed before the exec.
TEST_CASE("run_process closes the parent's other fds in the child",
          "[video]") {
    const int fd = ::open("/dev/null", O_RDONLY);
    REQUIRE(fd >= 0);
    REQUIRE(::dup2(fd, 77) == 77);  // dup2 leaves 77 inheritable
    ::close(fd);
    // /dev/fd/77 exists in the shell only while fd 77 is open (macOS
    // and Linux; dash cannot redirect fds above 9, so no ">&77").
    const auto r = locus::video::run_process(
        {"/bin/sh", "-c",
         "if [ -e /dev/fd/77 ]; then echo leaked; else echo closed; fi"},
        {});
    ::close(77);
    REQUIRE(r.exit_code == 0);
    REQUIRE(r.out == "closed\n");
}

TEST_CASE("run_process stops a child at its deadline or on request",
          "[video]") {
    using Clock = std::chrono::steady_clock;
    // Two commands, so every shell forks sleep as a grandchild that
    // holds the pipes: the kill must reach it too.
    auto t0 = Clock::now();
    auto r = locus::video::run_process(
        {"/bin/sh", "-c", "sleep 30; echo done"}, {}, 1);
    REQUIRE(r.killed);
    REQUIRE(r.exit_code == -1);
    REQUIRE(r.out.empty());
    REQUIRE(Clock::now() - t0 < std::chrono::seconds(10));
    t0 = Clock::now();
    r = locus::video::run_process({"/bin/sh", "-c", "sleep 30"}, {}, 0,
                                  [] { return false; });
    REQUIRE(r.killed);
    REQUIRE(Clock::now() - t0 < std::chrono::seconds(10));
    // A quick child within its deadline is not touched.
    r = locus::video::run_process({"/bin/sh", "-c", "echo ok"}, {}, 30,
                                  [] { return true; });
    REQUIRE_FALSE(r.killed);
    REQUIRE(r.out == "ok\n");
}

TEST_CASE("find_executable searches PATH as a shell does", "[video]") {
    const auto sh = locus::video::find_executable("sh");
    REQUIRE(sh.size() > 3);
    REQUIRE(sh.substr(sh.size() - 3) == "/sh");
    REQUIRE(locus::video::find_executable("/bin/sh") == "/bin/sh");
    REQUIRE(locus::video::find_executable("locus-no-such-cmd").empty());
    REQUIRE(locus::video::find_executable("/nonexistent/sh").empty());
    REQUIRE(locus::video::find_executable("/tmp").empty());  // a directory
}

TEST_CASE("an mp4 goes through the ffmpeg helper", "[video]") {
    const auto d = temp_dir("locus-video-ff");
    const auto ff = locus::video::probe_ffmpeg(
        fake_ffmpeg(d, " V....D libx264  H.264\\n V.S... mpeg4  MPEG-4\\n")
            .string());
    REQUIRE(ff);
    REQUIRE(ff->libx264);
    const auto out = (d / "clip.mp4").string();
    locus::video::encode_mp4(*ff, frames(3, 4, 2), 4, 2, 30.0f, out);
    REQUIRE(read_file(out) == "fakemp4");
    REQUIRE(read_file(out + ".size") == "72\n");  // 3 frames x 4 x 2 x 3
    // "$@" has every argument but argv[0], the executable.
    const auto args_x264 = locus::video::mp4_args(*ff, 4, 2, 30.0f, out);
    REQUIRE(args_x264.front() == ff->path);
    std::string want;
    for (std::size_t i = 1; i < args_x264.size(); ++i) {
        want += args_x264[i] + "\n";
    }
    REQUIRE(read_file(out + ".args") == want);
    REQUIRE(want.find("-s\n4x2\n-r\n30\n-i\n-\n-c:v\nlibx264\n") !=
            std::string::npos);
    REQUIRE(want.find("+faststart") != std::string::npos);

    // Without libx264, mpeg4; with neither encoder, not usable.
    const auto d2 = temp_dir("locus-video-ff2");
    const auto ff2 = locus::video::probe_ffmpeg(
        fake_ffmpeg(d2, " V.S... mpeg4  MPEG-4\\n").string());
    REQUIRE(ff2);
    REQUIRE_FALSE(ff2->libx264);
    const auto args = locus::video::mp4_args(*ff2, 4, 2, 24.0f, "o.mp4");
    REQUIRE(std::find(args.begin(), args.end(), "mpeg4") != args.end());
    const auto d3 = temp_dir("locus-video-ff3");
    REQUIRE_FALSE(locus::video::probe_ffmpeg(
        fake_ffmpeg(d3, " V..... rawvideo  raw\\n").string()));
    REQUIRE_FALSE(locus::video::probe_ffmpeg("/nonexistent/ffmpeg"));

    // A failing encode reports ffmpeg's stderr; bad frames are refused.
    const fs::path bad = d3 / "bad-ffmpeg";
    write_script(bad, "echo 'Unknown encoder' >&2; exit 1\n");
    const Ffmpeg broken{bad.string(), true};
    REQUIRE_THROWS_WITH(
        locus::video::encode_mp4(broken, frames(1, 4, 2), 4, 2, 30.0f,
                                 (d3 / "x.mp4").string()),
        Catch::Matchers::ContainsSubstring("Unknown encoder"));
    REQUIRE_THROWS_AS(locus::video::encode_mp4(*ff, frames(1, 4, 3), 4, 2,
                                               30.0f, out),
                      std::invalid_argument);
    fs::remove_all(d);
    fs::remove_all(d2);
    fs::remove_all(d3);
}

// The real ffmpeg, when installed: a 9-frame 64 x 64 clip is an MP4
// (an ftyp box first).
TEST_CASE("ffmpeg writes a real mp4", "[video]") {
    const auto ff = locus::video::probe_ffmpeg("ffmpeg");
    if (!ff) SKIP("no ffmpeg on PATH");
    const auto d = temp_dir("locus-video-real");
    const auto out = (d / "clip.mp4").string();
    locus::video::encode_mp4(*ff, frames(9, 64, 64), 64, 64, 30.0f, out);
    const auto bytes = read_file(out);
    REQUIRE(bytes.size() > 100);
    REQUIRE(bytes.substr(4, 4) == "ftyp");
    fs::remove_all(d);
}

TEST_CASE("video jobs run in order and expire", "[video]") {
    const auto d = temp_dir("locus-video-jobs");
    std::atomic<std::int64_t> clock{1000};
    VideoJobs jobs(d.string(), 2, 3600, [&] { return clock.load(); });
    VideoJob meta;
    meta.prompt = "a fox";
    meta.size = "64x64";
    const auto a = jobs.submit(meta, [](const std::string& out,
                                        const std::function<void(int)>& p,
                                        const std::function<bool()>&) {
        p(50);
        std::ofstream(out) << "clip";
    });
    REQUIRE(a);
    REQUIRE(a->status == "queued");
    REQUIRE(a->id.rfind("video_", 0) == 0);
    REQUIRE(a->created_at == 1000);
    const auto b = jobs.submit(meta, [](const std::string&,
                                        const std::function<void(int)>&,
                                        const std::function<bool()>&) {
        throw std::runtime_error("out of memory");
    });
    REQUIRE(b);
    jobs.wait_idle();
    const auto ja = jobs.get(a->id);
    REQUIRE(ja);
    REQUIRE(ja->status == "completed");
    REQUIRE(ja->progress == 100);
    REQUIRE(ja->expires_at == 1000 + 3600);
    REQUIRE(read_file(ja->file) == "clip");
    REQUIRE(ja->prompt == "a fox");
    const auto jb = jobs.get(b->id);
    REQUIRE(jb);
    REQUIRE(jb->status == "failed");
    REQUIRE(jb->error == "out of memory");
    REQUIRE_FALSE(jobs.get("video_nope"));
    // Past the expiry the job and its file are gone.
    clock = 1000 + 3600;
    REQUIRE_FALSE(jobs.get(a->id));
    REQUIRE_FALSE(fs::exists(ja->file));
    fs::remove_all(d);
}

TEST_CASE("video jobs cap the queue and stop a running job",
          "[video]") {
    const auto d = temp_dir("locus-video-cap");
    std::promise<void> started;
    std::atomic<bool> stopped{false};
    {
        VideoJobs jobs(d.string(), 1, 60);
        const auto a = jobs.submit(
            {}, [&](const std::string&, const std::function<void(int)>& p,
                    const std::function<bool()>& alive) {
                p(10);
                started.set_value();
                while (alive()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                stopped = true;
            });
        REQUIRE(a);
        started.get_future().wait();
        const auto running = jobs.get(a->id);
        REQUIRE(running->status == "in_progress");
        REQUIRE(running->progress == 10);
        // One pending job is the cap: the next is refused.
        REQUIRE_FALSE(jobs.submit({}, [](const std::string&,
                                         const std::function<void(int)>&,
                                         const std::function<bool()>&) {}));
    }  // the destructor stops the running job
    REQUIRE(stopped);
    fs::remove_all(d);
}
