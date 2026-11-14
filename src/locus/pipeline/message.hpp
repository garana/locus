#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "locus/tok/tokenizer.hpp"

namespace locus::pipeline {

/**
 * Wire protocol for pipeline-parallel inference across hosts
 * (multi-server, DESIGN.md "R15+"). A pipeline stage owns a contiguous
 * range of layers. Three message kinds cross a connection:
 *
 *   kToken       entry -> first stage: the input token to embed.
 *   kActivation  stage -> next stage: the residual stream (n_embd
 *                floats) handed on after a stage's layers.
 *   kLogits      last stage -> entry: the output logits (n_vocab
 *                floats) for sampling.
 *
 * This module is the transport: a binary, length-prefixed codec plus
 * blocking framed read/write over a socket fd. Connection setup (TCP
 * listen/connect) and the stage run loop build on it.
 */
enum class MsgType : std::uint16_t {
    kToken = 1,       /**< Input token for the first stage. */
    kActivation = 2,  /**< Residual stream between stages. */
    kLogits = 3,      /**< Output logits from the last stage. */
    kSessionEnd = 4,  /**< End of a sequence on a KEPT-OPEN connection:
                       *   the receiver frees that request_id's session
                       *   (releases its KV) but does NOT close the fd, so
                       *   the next request_id reuses the connection
                       *   (issue 76 / the downstream pool). No payload. */
    kExpertCall = 5,  /**< Expert parallelism (#781): run one MoE expert
                       *   on the hidden state. position = layer, token =
                       *   expert id, data = xb (n_embd floats). */
    kExpertOut = 6,   /**< The reply to a kExpertCall, with its tags:
                       *   data = the expert's output d (n_embd floats),
                       *   before the router weight. */
};

/**
 * One pipeline message, tagged by kind. `request_id` and `position`
 * identify the sequence and token position (position feeds
 * forward_layers' pos on the receiving stage). `token` is meaningful
 * only for kToken; `data` carries the floats for kActivation (the
 * residual stream, n_embd) and kLogits (n_vocab), and is empty for
 * kToken.
 */
struct Message {
    MsgType type = MsgType::kActivation;
    std::uint64_t request_id = 0;
    std::uint32_t position = 0;
    tok::TokenId token = 0;      /**< kToken only. */
    std::vector<float> data;     /**< kActivation / kLogits payload. */
};

/** @returns A kToken message. */
Message make_token(std::uint64_t request_id, std::uint32_t position,
                   tok::TokenId token);
/** @returns A kActivation message carrying `hidden` (n_embd floats). */
Message make_activation(std::uint64_t request_id,
                        std::uint32_t position,
                        std::vector<float> hidden);
/** @returns A kLogits message carrying `logits` (n_vocab floats). */
Message make_logits(std::uint64_t request_id, std::uint32_t position,
                    std::vector<float> logits);
/** @returns A kSessionEnd message for `request_id` (no payload); the
 * receiver frees that session but keeps the connection open for reuse. */
Message make_session_end(std::uint64_t request_id);
/**
 * @returns A kExpertCall: run expert `expert` of layer `layer` on `xb`
 *     (n_embd floats) for `request_id`.
 */
Message make_expert_call(std::uint64_t request_id, std::uint32_t layer,
                         std::uint32_t expert, std::vector<float> xb);
/**
 * @returns The kExpertOut reply to that call: `d` (n_embd floats), the
 *     expert's output before the router weight.
 */
Message make_expert_out(std::uint64_t request_id, std::uint32_t layer,
                        std::uint32_t expert, std::vector<float> d);

/** Outcome of a buffer decode attempt (mirrors auth::HelperDecode). */
enum class Decode {
    kIncomplete,  /**< Need more bytes; `buf` left untouched. */
    kComplete,    /**< One message decoded and consumed from `buf`. */
    kError,       /**< Malformed frame; the connection should close. */
};

/**
 * Encodes `m` as a length-prefixed binary frame appended to `out`:
 *
 *     u32 frame_len | u16 type | u16 flags | u64 request_id |
 *     u32 position  | i32 token | u32 n_floats | f32 data[n_floats]
 *
 * frame_len counts every byte after itself. All integers are
 * little-endian; the floats are copied raw (IEEE-754). locus targets a
 * homogeneous little-endian cluster (x86-64 / arm64), so the payload
 * floats are not byte-swapped.
 */
void encode(const Message& m, std::string& out);

/**
 * Decodes one frame from the front of `buf`. On kComplete the consumed
 * bytes are erased and `out` is filled; on kIncomplete `buf` is left
 * intact; on kError `err` describes the fault.
 */
Decode decode(std::string& buf, Message& out, std::string& err);

/**
 * As decode, but a frame whose length prefix leaves room for more than
 * `max_floats` floats is kError as soon as the prefix is in, before its
 * payload is buffered (i#854).
 */
Decode decode(std::string& buf, Message& out, std::string& err,
              std::uint32_t max_floats);

/** @returns The largest frame length accepted by decode/read_message;
 * guards against a desynced or hostile peer. */
std::uint32_t max_frame_bytes();

// ---- blocking framed transport over a socket fd (functional) ----
// No timeouts yet: functional correctness first, per the multi-server
// plan. Latency handling and non-blocking IO come later.

/**
 * Writes all of `bytes` to `fd`, looping on partial writes and
 * retrying EINTR. Uses MSG_NOSIGNAL where available so a peer that
 * closed does not raise SIGPIPE. @returns false on a write error.
 */
bool write_all(int fd, std::span<const char> bytes);

/** Encodes `m` and writes the whole frame to `fd`. @returns false on a
 * write error. */
bool write_message(int fd, const Message& m);

/** Outcome of read_message. */
enum class ReadResult {
    kOk,      /**< A complete message was read into `out`. */
    kEof,     /**< Peer closed cleanly at a frame boundary. */
    kError,   /**< Truncated frame, oversized frame, or bad data. */
    kTimeout, /**< Recv timeout (see set_recv_timeout) elapsed before a
               *   complete frame arrived. TERMINAL for the connection,
               *   like kError: a partial frame may already have been
               *   consumed, so a caller must reconnect, not re-read. */
};

/**
 * Reads exactly one frame from `fd` (blocking): the length prefix, then
 * that many payload bytes, so nothing is buffered between calls.
 *
 * @returns kEof only when the peer closes before any byte of a new
 *     frame; a close mid-frame is kError.
 */
ReadResult read_message(int fd, Message& out);

/**
 * As read_message, but a frame whose length leaves room for more than
 * `max_floats` floats is refused (kError) before its payload is read,
 * so a peer cannot make the reader allocate more than one message of
 * the model's size (i#850).
 *
 * @param max_floats The most floats one frame may carry (n_embd for an
 *     expert call or its answer).
 */
ReadResult read_message(int fd, Message& out, std::uint32_t max_floats);

}  // namespace locus::pipeline
