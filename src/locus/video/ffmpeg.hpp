#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace locus::video {

/** What a child process left behind. */
struct ProcessResult {
    int exit_code = -1;  ///< the exit status, or -1 when killed or not run
    std::string out;     ///< its stdout
    std::string err;     ///< its stderr, at most 64 KiB
};

/**
 * Runs argv (argv[0] looked up on PATH, no shell) with `input` written
 * to its stdin, one chunk after another, then stdin closed; reads its
 * stdout and stderr while it writes, so neither side blocks on a full
 * pipe. SIGPIPE is ignored process-wide (once), so a child that exits
 * early makes the write fail with EPIPE instead of killing locus.
 *
 * @param argv The command and its arguments; must be non-empty.
 * @param input The stdin bytes, in chunks (frames, for an encoder).
 * @returns The exit code and the captured output. A command that
 *     cannot be run gives exit code 127 (as a shell would).
 * @throws std::runtime_error when the pipes or the fork fail.
 */
ProcessResult run_process(const std::vector<std::string>& argv,
                          std::span<const std::span<const std::uint8_t>> input);

/** An ffmpeg executable and the H.264 encoder it offers. */
struct Ffmpeg {
    std::string path;
    bool libx264 = false;  ///< else the built-in mpeg4 encoder
};

/**
 * Checks that `path` runs as ffmpeg ("-hide_banner -encoders") and
 * whether it has libx264.
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
 * an MP4 file at out_path with ffmpeg.
 *
 * @throws std::invalid_argument on no frames or a wrong frame size.
 * @throws std::runtime_error when ffmpeg fails, with its stderr.
 */
void encode_mp4(const Ffmpeg& ff,
                const std::vector<std::vector<std::uint8_t>>& frames,
                std::uint32_t width, std::uint32_t height, float fps,
                const std::string& out_path);

}  // namespace locus::video
