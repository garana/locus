#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace locus::video {

/** What a child process left behind. */
struct ProcessResult {
    int exit_code = -1;  ///< the exit status, or -1 when killed or not run
    bool killed = false; ///< stopped by the deadline or by alive()
    std::string out;     ///< its stdout
    std::string err;     ///< its stderr, at most 64 KiB
};

/**
 * The executable a command name runs: a name with a '/' as it is, else
 * the first match on PATH (/usr/bin:/bin when PATH is unset), as a
 * shell finds it.
 *
 * @returns The path, or empty when nothing executable is found.
 */
std::string find_executable(const std::string& name);

/**
 * Runs argv (no shell; argv[0] resolved by find_executable in the
 * parent, then execv) with `input` written to its stdin, one chunk
 * after another, then stdin closed; reads its stdout and stderr while
 * it writes, so neither side blocks on a full pipe. The child keeps only
 * fds 0, 1 and 2: every other fd of locus (its listening and client
 * sockets) is closed before the exec. SIGPIPE is ignored process-wide
 * (once), so a child that exits early makes the write fail with EPIPE
 * instead of killing locus.
 *
 * @param argv The command and its arguments; must be non-empty.
 * @param input The stdin bytes, in chunks (frames, for an encoder).
 * @param timeout_s Kills the child (SIGKILL) after this many seconds;
 *     0 waits for it.
 * @param alive Polled about every 250 ms when set; false kills the
 *     child.
 * @returns The exit code and the captured output. A command that is
 *     not found gives exit code 127 (as a shell would).
 * @throws std::runtime_error when the pipes or the fork fail.
 */
ProcessResult run_process(const std::vector<std::string>& argv,
                          std::span<const std::span<const std::uint8_t>> input,
                          int timeout_s = 0,
                          const std::function<bool()>& alive = {});

/** An ffmpeg executable and the H.264 encoder it offers. */
struct Ffmpeg {
    std::string path;      ///< resolved (find_executable)
    bool libx264 = false;  ///< else the built-in mpeg4 encoder
};

/** How long an MP4 encode may take before ffmpeg is killed. */
constexpr int kEncodeTimeoutS = 600;

/**
 * Checks that `path` runs as ffmpeg ("-hide_banner -encoders") and
 * whether it has libx264; the result holds the resolved path.
 *
 * @returns The executable, or nothing when it does not run or exits
 *     with an error.
 */
std::optional<Ffmpeg> probe_ffmpeg(const std::string& path);

/**
 * The ffmpeg arguments that encode raw RGB frames from stdin into an
 * MP4 file: rawvideo rgb24 at width x height and fps, then libx264
 * (or mpeg4) in yuv420p, faststart (the index first, for streaming
 * players), at most two encoder threads.
 */
std::vector<std::string> mp4_args(const Ffmpeg& ff, std::uint32_t width,
                                  std::uint32_t height, float fps,
                                  const std::string& out_path);

/**
 * Encodes frames (each height * width * 3 RGB bytes, row-major) into
 * an MP4 file at out_path with ffmpeg, which is killed after
 * kEncodeTimeoutS seconds or when alive() turns false.
 *
 * @throws std::invalid_argument on no frames or a wrong frame size.
 * @throws std::runtime_error when ffmpeg fails (with its stderr), times
 *     out or is stopped.
 */
void encode_mp4(const Ffmpeg& ff,
                const std::vector<std::vector<std::uint8_t>>& frames,
                std::uint32_t width, std::uint32_t height, float fps,
                const std::string& out_path,
                const std::function<bool()>& alive = {});

}  // namespace locus::video
