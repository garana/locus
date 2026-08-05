#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "locus/auth/auth_client.hpp"
#include "locus/chat/template.hpp"
#include "locus/engine/engine.hpp"
#include "locus/model/llama.hpp"
#include "locus/server/engine_loop.hpp"
#include "locus/tok/tokenizer.hpp"

namespace httplib {
class Server;
struct Request;
struct Response;
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
        engine::Engine::Config engine;
        /** Hard cap applied to requested max_tokens. */
        std::uint32_t max_tokens_cap = 1024;
        /** Path the Prometheus metrics endpoint is served at
         * (--metrics-path); default /metrics. */
        std::string metrics_path = "/metrics";
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
    };

    OpenAiServer(const model::LlamaModel& m,
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
    void install_routes();
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
    bool authorize(const httplib::Request& req,
                   httplib::Response& res, std::string& identity);
    /** Fire-and-forget request-lifecycle event to the auth helper
     * (no-op when auth is off). */
    void auth_event(const char* event, const std::string& kind,
                    const std::string& identity);

    const model::LlamaModel& model_;
    const tok::Tokenizer& tok_;
    Options opt_;
    EngineLoop loop_;
    std::unique_ptr<httplib::Server> http_;
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
    model::LlamaModel::Workspace embed_ws_;
};

}  // namespace locus::server
