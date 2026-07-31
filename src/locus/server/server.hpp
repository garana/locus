#pragma once

#include <memory>
#include <string>

#include "locus/chat/template.hpp"
#include "locus/engine/engine.hpp"
#include "locus/model/llama.hpp"
#include "locus/server/engine_loop.hpp"
#include "locus/tok/tokenizer.hpp"

namespace httplib {
class Server;
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
 * configured chat template. Text-only: image and tool_use/
 * tool_result content blocks are ignored (no tool support yet).
 */
class OpenAiServer {
  public:
    struct Options {
        /** Reported in responses' "model" field. */
        std::string model_name = "locus";
        engine::Engine::Config engine;
        /** Hard cap applied to requested max_tokens. */
        std::uint32_t max_tokens_cap = 1024;
        /** Formats /v1/chat/completions prompts (default plain).
         * Use chat::ChatTemplate::from_gguf for real chat
         * models. */
        chat::ChatTemplate chat_template;
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

    const tok::Tokenizer& tok_;
    Options opt_;
    EngineLoop loop_;
    std::unique_ptr<httplib::Server> http_;
};

}  // namespace locus::server
