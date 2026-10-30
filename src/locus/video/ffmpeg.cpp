#include "locus/video/ffmpeg.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <stdexcept>

namespace locus::video {

namespace {

constexpr std::size_t kMaxErr = 64 * 1024;

/** SIGPIPE kills a process that writes to a pipe whose reader is gone;
 *  ignore it once, and handle EPIPE on write() instead. */
void ignore_sigpipe_once() {
    static std::once_flag once;
    std::call_once(once, [] { std::signal(SIGPIPE, SIG_IGN); });
}

void close_fd(int& fd) {
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

/** A pipe whose two ends close on exec (the child dup2s the ones it
 *  keeps onto 0, 1 and 2). */
void make_pipe(int fds[2]) {
    if (::pipe(fds) != 0) {
        throw std::runtime_error("video: pipe failed");
    }
    ::fcntl(fds[0], F_SETFD, FD_CLOEXEC);
    ::fcntl(fds[1], F_SETFD, FD_CLOEXEC);
}

void set_nonblocking(int fd) {
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags >= 0) ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

/** Reads what is ready on fd into s (up to cap bytes kept); @returns
 *  false at end of file or on an error. */
bool drain(int fd, std::string& s, std::size_t cap) {
    char buf[16384];
    for (;;) {
        const ssize_t n = ::read(fd, buf, sizeof buf);
        if (n > 0) {
            const std::size_t keep =
                s.size() < cap ? std::min(cap - s.size(),
                                          static_cast<std::size_t>(n))
                               : 0;
            s.append(buf, keep);
            continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return true;
        if (n < 0 && errno == EINTR) continue;
        return false;
    }
}

}  // namespace

ProcessResult run_process(
    const std::vector<std::string>& argv,
    std::span<const std::span<const std::uint8_t>> input) {
    if (argv.empty()) throw std::invalid_argument("video: empty argv");
    ignore_sigpipe_once();
    // The exec arguments are built before the fork: the child only
    // calls async-signal-safe functions.
    std::vector<char*> args;
    for (const auto& a : argv) args.push_back(const_cast<char*>(a.c_str()));
    args.push_back(nullptr);

    int in[2] = {-1, -1}, out[2] = {-1, -1}, err[2] = {-1, -1};
    try {
        make_pipe(in);
        make_pipe(out);
        make_pipe(err);
    } catch (...) {
        for (int* p : {in, out, err}) {
            close_fd(p[0]);
            close_fd(p[1]);
        }
        throw;
    }
    const pid_t pid = ::fork();
    if (pid < 0) {
        for (int* p : {in, out, err}) {
            close_fd(p[0]);
            close_fd(p[1]);
        }
        throw std::runtime_error("video: fork failed");
    }
    if (pid == 0) {
        ::dup2(in[0], 0);
        ::dup2(out[1], 1);
        ::dup2(err[1], 2);
        ::execvp(args[0], args.data());
        ::_exit(127);
    }
    close_fd(in[0]);
    close_fd(out[1]);
    close_fd(err[1]);
    set_nonblocking(in[1]);
    set_nonblocking(out[0]);
    set_nonblocking(err[0]);

    ProcessResult r;
    std::size_t chunk = 0, off = 0;
    const auto skip_empty = [&] {
        while (chunk < input.size() && off >= input[chunk].size()) {
            ++chunk;
            off = 0;
        }
    };
    skip_empty();
    if (chunk == input.size()) close_fd(in[1]);
    while (in[1] >= 0 || out[0] >= 0 || err[0] >= 0) {
        pollfd p[3];
        nfds_t n = 0;
        if (in[1] >= 0) p[n++] = {in[1], POLLOUT, 0};
        if (out[0] >= 0) p[n++] = {out[0], POLLIN, 0};
        if (err[0] >= 0) p[n++] = {err[0], POLLIN, 0};
        if (::poll(p, n, -1) < 0) {
            if (errno == EINTR) continue;
            break;
        }
        for (nfds_t k = 0; k < n; ++k) {
            if (p[k].revents == 0) continue;
            if (p[k].fd == in[1]) {
                const auto& c = input[chunk];
                const ssize_t w = ::write(in[1], c.data() + off, c.size() - off);
                if (w > 0) {
                    off += static_cast<std::size_t>(w);
                    skip_empty();
                    if (chunk == input.size()) close_fd(in[1]);
                } else if (w < 0 && errno != EAGAIN && errno != EINTR) {
                    close_fd(in[1]);  // EPIPE: the child stopped reading
                }
            } else if (p[k].fd == out[0]) {
                if (!drain(out[0], r.out, SIZE_MAX)) close_fd(out[0]);
            } else if (p[k].fd == err[0]) {
                if (!drain(err[0], r.err, kMaxErr)) close_fd(err[0]);
            }
        }
    }
    close_fd(in[1]);
    close_fd(out[0]);
    close_fd(err[0]);
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    r.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return r;
}

std::optional<Ffmpeg> probe_ffmpeg(const std::string& path) {
    if (path.empty()) return std::nullopt;
    const auto r = run_process({path, "-hide_banner", "-encoders"}, {});
    if (r.exit_code != 0) return std::nullopt;
    Ffmpeg ff{path, r.out.find(" libx264 ") != std::string::npos};
    if (!ff.libx264 && r.out.find(" mpeg4 ") == std::string::npos) {
        return std::nullopt;  // neither encoder: not an ffmpeg we can use
    }
    return ff;
}

std::vector<std::string> mp4_args(const Ffmpeg& ff, std::uint32_t width,
                                  std::uint32_t height, float fps,
                                  const std::string& out_path) {
    char rate[32];
    std::snprintf(rate, sizeof rate, "%g", static_cast<double>(fps));
    std::vector<std::string> a = {
        ff.path, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", std::to_string(width) + "x" + std::to_string(height),
        "-r", rate, "-i", "-"};
    if (ff.libx264) {
        a.insert(a.end(), {"-c:v", "libx264", "-preset", "veryfast"});
    } else {
        a.insert(a.end(), {"-c:v", "mpeg4", "-q:v", "3"});
    }
    a.insert(a.end(), {"-pix_fmt", "yuv420p", "-movflags", "+faststart",
                       "-threads", "2", out_path});
    return a;
}

void encode_mp4(const Ffmpeg& ff,
                const std::vector<std::vector<std::uint8_t>>& frames,
                std::uint32_t width, std::uint32_t height, float fps,
                const std::string& out_path) {
    if (frames.empty()) throw std::invalid_argument("video: no frames");
    const std::size_t bytes = std::size_t{width} * height * 3;
    std::vector<std::span<const std::uint8_t>> chunks;
    for (const auto& f : frames) {
        if (f.size() != bytes) {
            throw std::invalid_argument("video: frame is not width x height");
        }
        chunks.emplace_back(f);
    }
    const auto r = run_process(mp4_args(ff, width, height, fps, out_path),
                               chunks);
    std::error_code ec;
    if (r.exit_code != 0 || std::filesystem::file_size(out_path, ec) == 0 ||
        ec) {
        throw std::runtime_error("ffmpeg failed (exit " +
                                 std::to_string(r.exit_code) + "): " + r.err);
    }
}

}  // namespace locus::video
