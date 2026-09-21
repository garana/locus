#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace locus::pipeline {

/**
 * Wire protocol for pipeline-parallel inference across hosts
 * (multi-server, DESIGN.md "R15+"). One pipeline stage owns a
 * contiguous range of layers and hands the residual stream (the
 * between-layers hidden state) to the next stage as an Activation
 * message. This module is the transport primitive: a binary,
 * length-prefixed codec plus blocking framed read/write over a socket
 * fd. Connection setup (TCP listen/connect) and the stage run loop
 * come in later increments.
 */

/** Message kinds carried on a pipeline connection. */
enum class MsgType : std::uint16_t {
    kActivation = 1,  /**< A stage's output residual stream, one token. */
    kResult = 2,      /**< Final stage's sampled token back to the entry
                       *   host (reserved; not encoded yet). */
};

/**
 * One token's residual stream handed from one stage to the next,
 * tagged with the sequence and token position so the receiving stage
 * writes its KV at the matching slot (forward_layers uses `position`
 * as `pos`).
 */
struct Activation {
    std::uint64_t request_id = 0;  /**< Sequence this token belongs to. */
    std::uint32_t position = 0;    /**< Token position in the sequence. */
    std::vector<float> hidden;     /**< n_embd floats (residual stream). */
};

/** Outcome of a buffer decode attempt (mirrors auth::HelperDecode). */
enum class Decode {
    kIncomplete,  /**< Need more bytes; `buf` left untouched. */
    kComplete,    /**< One message decoded and consumed from `buf`. */
    kError,       /**< Malformed frame; the connection should close. */
};

/**
 * Encodes `a` as a length-prefixed binary frame appended to `out`:
 *
 *     u32 frame_len | u16 type | u16 flags | u64 request_id |
 *     u32 position  | u32 n_floats | f32 hidden[n_floats]
 *
 * frame_len counts every byte after itself. All integers are
 * little-endian; the hidden floats are copied raw (IEEE-754). locus
 * targets a homogeneous little-endian cluster (x86-64 / arm64), so the
 * payload floats are not byte-swapped.
 */
void encode(const Activation& a, std::string& out);

/**
 * Decodes one frame from the front of `buf`. On kComplete the consumed
 * bytes are erased from `buf` and `out` is filled; on kIncomplete `buf`
 * is left intact; on kError `err` describes the fault and the caller
 * should close the connection.
 */
Decode decode(std::string& buf, Activation& out, std::string& err);

/** @returns The largest frame length accepted by decode/read_message;
 * guards against a desynced or hostile peer. */
std::uint32_t max_frame_bytes();

// ---- blocking framed transport over a socket fd (functional) ----
// No timeouts yet: functional correctness first, per the multi-server
// plan. Latency handling and non-blocking IO come later.

/**
 * Writes all of `bytes` to `fd`, looping on partial writes and
 * retrying EINTR. Uses MSG_NOSIGNAL where available so a peer that
 * closed does not raise SIGPIPE.
 *
 * @returns false on a write error or a zero-length write (EOF).
 */
bool write_all(int fd, std::span<const char> bytes);

/** Encodes `a` and writes the whole frame to `fd`. @returns false on a
 * write error. */
bool write_message(int fd, const Activation& a);

/** Outcome of read_message. */
enum class ReadResult {
    kOk,     /**< A complete message was read into `out`. */
    kEof,    /**< Peer closed cleanly at a frame boundary. */
    kError,  /**< Truncated frame, oversized frame, or bad data. */
};

/**
 * Reads exactly one frame from `fd` (blocking): the length prefix, then
 * that many payload bytes, so nothing is buffered between calls.
 *
 * @param out Filled on kOk.
 * @returns kEof only when the peer closes before any byte of a new
 *     frame; a close mid-frame is kError.
 */
ReadResult read_message(int fd, Activation& out);

}  // namespace locus::pipeline
