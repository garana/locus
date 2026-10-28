#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "locus/auth/auth_client.hpp"
#include "locus/chat/template.hpp"
#include "locus/engine/engine.hpp"
#include "locus/http/server.hpp"
#include "locus/model/transformer.hpp"
#include "locus/server/engine_loop.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::audio {
class SnacDecoder;
}
namespace locus::image {
class JanusImageGen;
class JanusVision;
class Vq16Decoder;
}
namespace locus::server {
struct VisionChat;
}

namespace locus::server {

/**
 * OpenAI- and Anthropic-compatible HTTP front end (DESIGN.md 4.5):
 *
 *   POST /v1/completions        {prompt, max_tokens, stream}
 *   POST /v1/chat/completions   {messages, max_tokens, stream}
 *   POST /v1/messages           Anthropic Messages API
 *   GET  /health
 *
 * The OpenAI endpoints stream as text/event-stream with OpenAI-style
 * `data: {...}` chunks and a final `data: [DONE]`. /v1/messages
 * speaks the Anthropic shape -- {system, messages, max_tokens,
 * stream}, a {type:"message", content:[{type:"text"}]} response, and
 * the message_start / content_block_delta / message_stop SSE event
 * sequence -- so Claude Code (ANTHROPIC_BASE_URL) and the Anthropic
 * SDK can target locus directly. Chat prompts are built via the
 * configured chat template.
 *
 * Tool-calling output: /v1/chat/completions and /v1/messages accept
 * a `tools` list (OpenAI or Anthropic shape). It is rendered into a
 * system instruction, and if the (non-streaming) completion parses
 * as a tool call the response carries OpenAI `tool_calls` /
 * finish_reason "tool_calls", or Anthropic `tool_use` /
 * stop_reason "tool_use". locus only expresses the call -- it never
 * runs the tool (execution is the client's job). See server/tools.
 * Incoming image/tool_result content blocks are still ignored.
 */
class OpenAiServer {
  public:
    struct Options {
        /** Reported in responses' "model" field. */
        std::string model_name = "locus";
        /** The GGUF's general.architecture, for the /metrics
         * locus_info series (empty when unknown). */
        std::string model_arch;
        engine::Engine::Config engine;
        /** Hard cap applied to requested max_tokens. */
        std::uint32_t max_tokens_cap = 1024;
        /** Max HTTP request body accepted before the handler runs;
         * httplib buffers the whole body into memory, so an
         * unbounded Content-Length is an unauthenticated OOM. Bodies
         * over this get 413. Default 16 MiB. */
        std::size_t max_body_bytes = 16u * 1024 * 1024;
        /** Max number of inputs one /v1/embeddings call may batch;
         * each runs a serialized model forward, so an unbounded array
         * is a CPU-exhaustion vector. Over this gets 400. */
        std::size_t max_embeddings_inputs = 256;
        /** Path the Prometheus metrics endpoint is served at
         * (--metrics-path); default /metrics. */
        std::string metrics_path = "/metrics";
        /** Path to the neural-codec decoder GGUF (arch "snac") that
         * enables POST /v1/audio/speech (--audio-decoder). Empty =
         * the endpoint is off and returns 503. Loaded at construction;
         * the loaded model must be an Orpheus-TTS checkpoint for the
         * endpoint to produce audio. */
        std::string audio_decoder_path;
        /** Per-request cap on generated audio tokens for
         * /v1/audio/speech (--audio-max-tokens); a request's own
         * max_tokens is clamped to it. 7 tokens = one 85 ms frame. */
        std::uint32_t audio_max_tokens = 1024;
        /** Path to the VQ-16 image decoder GGUF (arch "vq16") that
         * enables POST /v1/images/generations (--image-decoder). Empty
         * = the endpoint is off and returns 503. The loaded model must
         * be a Janus-Pro checkpoint, whose janus.* tensors are read
         * from model_path. */
        std::string image_decoder_path;
        /** Path of the loaded model's GGUF; the image endpoint reads
         * the janus.* tensors from it. */
        std::string model_path;
        /** Formats /v1/chat/completions prompts (default plain).
         * Use chat::ChatTemplate::from_gguf for real chat
         * models. */
        chat::ChatTemplate chat_template;
        /** Optional auth (DESIGN R15): when auth_helper_argv is
         * non-empty, a request to a protected route must carry a
         * credential (Authorization: Bearer <t>, or x-api-key: <t>)
         * that a helper resolves; the same sockets also receive
         * request-lifecycle event pushes. Empty = auth OFF (open,
         * the default). auth_helpers helper processes are spawned. */
        std::vector<std::string> auth_helper_argv;
        std::size_t auth_helpers = 1;
        auth::AuthConfig auth_cache;
        /** Prompt caching (KV prefix reuse) -- on by default. A
         * request sharing a block-aligned prompt prefix with a recent
         * finished request adopts its pinned KV and skips reprefill;
         * byte-exact, reported via the API usage cache fields. Enables
         * engine.prefix_cache. */
        bool prompt_cache = true;
    };

    OpenAiServer(const model::TransformerModel& m,
                 const tok::Tokenizer& tok, Options opt);
    ~OpenAiServer();

    /** Serves forever on host:port. @returns false on bind error. */
    bool listen(const std::string& host, int port);

    /**
     * Binds to an OS-assigned port (for tests).
     *
     * @returns The port, or -1 on failure. Serve with
     *     listen_after_bind() from another thread.
     */
    int bind_any_port(const std::string& host);

    /** Blocking companion to bind_any_port(). */
    bool listen_after_bind();

    /** Stops a running listen loop (thread-safe). */
    void stop();

  private:
    /** Engine config with prompt caching resolved from Options:
     * prefix_cache = engine.prefix_cache || prompt_cache. */
    static engine::Engine::Config resolve_engine_cfg(const Options& o);
    /** Which endpoint a (method, path) hits. One home for routing, used
     * by both the dispatch handler and the auth head stage. */
    enum class Route {
        kHealth, kModels, kModelById, kMetrics,
        kCompletions, kChat, kMessages, kEmbeddings, kAudioSpeech,
        kImages, kUnknown
    };
    /** Classifies a request. GET and HEAD both match the GET routes;
     * /v1/models/{id} matches by prefix (the regex route retired). */
    Route route_of(const std::string& method,
                   const std::string& path) const;
    /** Builds the dispatch + auth-head handlers into route_handler_ /
     * auth_head_ (called once, from the ctor). */
    void install_routes();
    /** Constructs http_ bound to host:port and starts it; @returns the
     * bound port (>0) or -1. */
    int build_and_start(const std::string& host, int port);
    /** Lazily builds (once) and returns the token -> decoded-bytes
     * table used by constrained decoding. */
    std::shared_ptr<const std::vector<std::string>> json_pieces();
    /** L2-normalized last-token embedding of `text` (serialized). */
    std::vector<float> embed_text(const std::string& text);
    /**
     * Authorizes a request. When auth is off, allows (identity "").
     * Otherwise reads the bearer/x-api-key credential and resolves
     * it; on deny sets a 401 response and returns false.
     */
    bool authorize(const http::Request& req,
                   http::ServerResponse& res, std::string& identity);
    /** Fire-and-forget request-lifecycle event to the auth helper
     * (no-op when auth is off). */
    void auth_event(const char* event, const std::string& kind,
                    const std::string& identity);
    /**
     * Answers a chat (OpenAI) or messages (Anthropic) request whose
     * messages carry images (i#241): the Janus vision encoder turns
     * each image into its placeholder embeddings, then the reply is
     * decoded on a dedicated cache, serialized under vision_mu_.
     * Non-streaming.
     *
     * @param anthropic Answer in the /v1/messages shape.
     * @param chat The parsed request (turns, images, max_tokens,
     *     sampling).
     */
    void vision_chat(bool anthropic, const VisionChat& chat,
                     http::ServerResponse& res,
                     const http::RequestContext& ctx);

    const model::TransformerModel& model_;
    const tok::Tokenizer& tok_;
    Options opt_;
    EngineLoop loop_;
    http::Handler route_handler_;    /**< dispatch (built in ctor). */
    http::HeadHandler auth_head_;    /**< auth head stage (built in ctor). */
    std::unique_ptr<http::Server> http_;
    std::uint32_t n_vocab_ = 0;

    // Cumulative counters exposed at GET /metrics (Prometheus text).
    std::atomic<std::uint64_t> requests_total_{0};
    std::atomic<std::uint64_t> prompt_tokens_total_{0};
    std::atomic<std::uint64_t> completion_tokens_total_{0};

    // Optional auth (built from Options.auth_helper_argv; null=off).
    auth::WallClock auth_clock_;
    std::unique_ptr<auth::AuthClient> auth_;

    std::once_flag pieces_once_;
    std::shared_ptr<const std::vector<std::string>> pieces_;

    // Embeddings run on a dedicated cache/workspace, serialized.
    std::mutex embed_mu_;
    std::once_flag embed_once_;
    std::unique_ptr<kv::PagedKvCache> embed_cache_;
    model::TransformerModel::Workspace embed_ws_;

    // /v1/audio/speech: the SNAC decoder (null when --audio-decoder was
    // not given) and the Orpheus audio band base resolved from the
    // vocab (-1 when the loaded model is not a TTS checkpoint). The
    // decode runs on a serialized dedicated cache, like embeddings.
    std::unique_ptr<audio::SnacDecoder> snac_;
    tok::TokenId audio_band_base_ = -1;
    /** Head rows speech samples from (Orpheus's own tokens); all rows
     * when the vocab is not laid out as expected. */
    model::LogitRows audio_rows_;
    std::mutex audio_mu_;
    std::unique_ptr<kv::PagedKvCache> audio_cache_;
    model::TransformerModel::Workspace audio_ws_;

    // /v1/images/generations: the VQ-16 decoder (null when
    // --image-decoder was not given) and the Janus image heads (null
    // when the model has no janus.* tensors). One image at a time, under
    // image_mu_: a generation holds its own KV cache for two sequences.
    // The heads point into janus_gguf_'s mapping (declared first, so it
    // is destroyed last).
    std::unique_ptr<image::Vq16Decoder> vq16_;
    std::unique_ptr<gguf::GgufFile> janus_gguf_;
    std::unique_ptr<image::JanusImageGen> janus_;
    std::mutex image_mu_;

    // Image input (i#241): the Janus vision encoder, loaded when the
    // model's GGUF has vit.* tensors (null otherwise: a request with an
    // image then gets a 400). Its matrices point into vision_gguf_'s
    // mapping. Replies run one at a time on a dedicated cache.
    std::unique_ptr<gguf::GgufFile> vision_gguf_;
    std::unique_ptr<image::JanusVision> vision_;
    std::mutex vision_mu_;
    std::unique_ptr<kv::PagedKvCache> vision_cache_;
    model::TransformerModel::Workspace vision_ws_;
};

}  // namespace locus::server
