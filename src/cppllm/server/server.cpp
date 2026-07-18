#include "cppllm/server/server.hpp"

#include <cstdint>
#include <vector>

#include "httplib.h"
#include "json.hpp"

namespace cppllm::server {

namespace {

using nlohmann::json;

json make_error(const std::string& message) {
    return json{{"error", {{"message", message},
                           {"type", "invalid_request_error"}}}};
}

/** Flattens chat messages to a plain-text prompt (MVP: no chat
 * template; base models complete text). */
std::string flatten_messages(const json& messages) {
    std::string prompt;
    for (const auto& m : messages) {
        if (!m.contains("content") ||
            !m["content"].is_string()) {
            throw std::invalid_argument(
                "message content must be a string");
        }
        if (!prompt.empty()) {
            prompt += "\n";
        }
        prompt += m["content"].get<std::string>();
    }
    return prompt;
}

}  // namespace

OpenAiServer::OpenAiServer(const model::LlamaModel& m,
                           const tok::SpmTokenizer& tok,
                           Options opt)
    : tok_(tok),
      opt_(std::move(opt)),
      loop_(m, tok.eos_id(), opt_.engine),
      http_(std::make_unique<httplib::Server>()) {
    install_routes();
}

OpenAiServer::~OpenAiServer() { http_->stop(); }

bool OpenAiServer::listen(const std::string& host, int port) {
    return http_->listen(host, port);
}

int OpenAiServer::bind_any_port(const std::string& host) {
    return http_->bind_to_any_port(host);
}

bool OpenAiServer::listen_after_bind() {
    return http_->listen_after_bind();
}

void OpenAiServer::stop() { http_->stop(); }

void OpenAiServer::install_routes() {
    http_->Get("/health",
               [](const httplib::Request&, httplib::Response& res) {
                   res.set_content(json{{"status", "ok"}}.dump(),
                                   "application/json");
               });

    // Both endpoints share one implementation; `chat` only changes
    // how the prompt is built and how the response is shaped.
    auto handle = [this](const httplib::Request& req,
                         httplib::Response& res, bool chat) {
        json body;
        std::string prompt;
        std::uint32_t max_tokens = 16;
        bool stream = false;
        try {
            body = json::parse(req.body);
            if (chat) {
                if (!body.contains("messages") ||
                    !body["messages"].is_array() ||
                    body["messages"].empty()) {
                    throw std::invalid_argument(
                        "messages must be a non-empty array");
                }
                prompt = flatten_messages(body["messages"]);
            } else {
                if (!body.contains("prompt") ||
                    !body["prompt"].is_string()) {
                    throw std::invalid_argument(
                        "prompt must be a string");
                }
                prompt = body["prompt"].get<std::string>();
            }
            if (body.contains("max_tokens")) {
                const auto v = body["max_tokens"];
                if (!v.is_number_unsigned() ||
                    v.get<std::uint64_t>() == 0) {
                    throw std::invalid_argument(
                        "max_tokens must be a positive integer");
                }
                max_tokens = static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(
                        v.get<std::uint64_t>(),
                        opt_.max_tokens_cap));
            }
            stream = body.value("stream", false);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(make_error(e.what()).dump(),
                            "application/json");
            return;
        }

        const std::uint64_t id =
            loop_.submit(tok_.encode(prompt, true), max_tokens);
        const std::string object =
            chat ? "chat.completion" : "text_completion";
        const std::string sse_object =
            chat ? "chat.completion.chunk" : "text_completion";
        const std::string rid =
            "cppllm-" + std::to_string(id);

        if (!stream) {
            auto v = loop_.wait_done(id);
            if (v.status == engine::Status::kFailed) {
                res.status = 500;
                res.set_content(make_error(v.error).dump(),
                                "application/json");
                return;
            }
            std::string text = tok_.decode(v.generated);
            const bool hit_eos =
                !v.generated.empty() &&
                v.generated.back() == tok_.eos_id();
            json choice =
                chat ? json{{"index", 0},
                            {"message",
                             {{"role", "assistant"},
                              {"content", text}}},
                            {"finish_reason",
                             hit_eos ? "stop" : "length"}}
                     : json{{"index", 0},
                            {"text", text},
                            {"finish_reason",
                             hit_eos ? "stop" : "length"}};
            json out{{"id", rid},
                     {"object", object},
                     {"model", opt_.model_name},
                     {"choices", json::array({choice})}};
            res.set_content(out.dump(), "application/json");
            return;
        }

        // SSE streaming: one chunk per token, then [DONE].
        auto state = std::make_shared<std::size_t>(0);
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, id, chat, rid, sse_object, state](
                std::size_t, httplib::DataSink& sink) {
                auto v = loop_.wait_progress(id, *state);
                std::string payload;
                for (std::size_t i = *state;
                     i < v.generated.size(); ++i) {
                    const tok::TokenId t = v.generated[i];
                    if (t == tok_.eos_id()) {
                        continue;
                    }
                    const std::string piece =
                        tok_.decode({&t, 1});
                    json delta =
                        chat ? json{{"index", 0},
                                    {"delta",
                                     {{"content", piece}}}}
                             : json{{"index", 0},
                                    {"text", piece}};
                    json chunk{{"id", rid},
                               {"object", sse_object},
                               {"choices",
                                json::array({delta})}};
                    payload +=
                        "data: " + chunk.dump() + "\n\n";
                }
                *state = v.generated.size();
                const bool terminal =
                    v.status == engine::Status::kDone ||
                    v.status == engine::Status::kFailed;
                if (terminal) {
                    payload += "data: [DONE]\n\n";
                }
                if (!payload.empty() &&
                    !sink.write(payload.data(),
                                payload.size())) {
                    return false;
                }
                if (terminal) {
                    sink.done();
                    return false;
                }
                return true;
            });
    };

    http_->Post("/v1/completions",
                [handle](const httplib::Request& req,
                         httplib::Response& res) {
                    handle(req, res, false);
                });
    http_->Post("/v1/chat/completions",
                [handle](const httplib::Request& req,
                         httplib::Response& res) {
                    handle(req, res, true);
                });
}

}  // namespace cppllm::server
