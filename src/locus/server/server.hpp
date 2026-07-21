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
 * OpenAI-compatible HTTP front end (DESIGN.md 4.5):
 *
 *   POST /v1/completions        {prompt, max_tokens, stream}
 *   POST /v1/chat/completions   {messages, max_tokens, stream}
 *   GET  /health
 *
 * stream=true answers as text/event-stream with OpenAI-style
 * `data: {...}` chunks and a final `data: [DONE]`. Chat messages
 * are flattened to plain text (base models have no chat template;
 * documented limitation for the MVP).
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
