#include "locus/server/server.hpp"

#include <cstdint>
#include <vector>

#include "httplib.h"
#include "json.hpp"

namespace locus::server {

namespace {

using nlohmann::json;

json make_error(const std::string& message) {
    return json{{"error", {{"message", message},
                           {"type", "invalid_request_error"}}}};
}

/** Parses and validates the OpenAI messages array. */
std::vector<chat::Message> parse_messages(const json& messages) {
    std::vector<chat::Message> out;
    for (const auto& m : messages) {
        if (!m.contains("content") ||
            !m["content"].is_string()) {
            throw std::invalid_argument(
                "message content must be a string");
        }
        chat::Message msg;
        msg.role = m.value("role", "user");
        if (msg.role != "system" && msg.role != "user" &&
            msg.role != "assistant") {
            throw std::invalid_argument(
                "unsupported message role: " + msg.role);
        }
        msg.content = m["content"].get<std::string>();
        out.push_back(std::move(msg));
    }
    return out;
}

/**
 * Anthropic content is either a plain string or an array of content
 * blocks; we take the text of every {"type":"text"} block and
 * ignore the rest (images, tool_use/tool_result -- no tool support
 * yet). @returns the concatenated text.
 */
std::string anthropic_text(const json& content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (content.is_array()) {
        std::string out;
        for (const auto& block : content) {
            if (block.value("type", "") == "text" &&
                block.contains("text") &&
                block["text"].is_string()) {
                out += block["text"].get<std::string>();
            }
        }
        return out;
    }
    throw std::invalid_argument(
        "content must be a string or an array of blocks");
}

/**
 * Parses an Anthropic /v1/messages body into chat::Message list:
 * the top-level `system` (string or blocks) becomes a leading
 * system message, then each messages[] entry (user/assistant).
 */
std::vector<chat::Message> parse_anthropic_messages(
    const json& body) {
    std::vector<chat::Message> out;
    if (body.contains("system") && !body["system"].is_null()) {
        chat::Message sys;
        sys.role = "system";
        sys.content = anthropic_text(body["system"]);
        if (!sys.content.empty()) {
            out.push_back(std::move(sys));
        }
    }
    for (const auto& m : body.at("messages")) {
        chat::Message msg;
        msg.role = m.value("role", "user");
        if (msg.role != "user" && msg.role != "assistant") {
            throw std::invalid_argument(
                "unsupported message role: " + msg.role);
        }
        if (!m.contains("content")) {
            throw std::invalid_argument(
                "message must have content");
        }
        msg.content = anthropic_text(m["content"]);
        out.push_back(std::move(msg));
    }
    return out;
}

}  // namespace

OpenAiServer::OpenAiServer(const model::LlamaModel& m,
                           const tok::Tokenizer& tok,
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
                prompt = opt_.chat_template.apply(
                    parse_messages(body["messages"]));
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
            "locus-" + std::to_string(id);

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

    // Anthropic Messages API (POST /v1/messages): same engine, its
    // own request/response shape and SSE event sequence, so Claude
    // Code and the Anthropic SDK can target locus directly.
    auto handle_messages = [this](const httplib::Request& req,
                                  httplib::Response& res) {
        std::string prompt;
        std::uint32_t max_tokens = 16;
        std::uint32_t input_tokens = 0;
        bool stream = false;
        std::vector<tok::TokenId> ids;
        try {
            const json body = json::parse(req.body);
            if (!body.contains("messages") ||
                !body["messages"].is_array() ||
                body["messages"].empty()) {
                throw std::invalid_argument(
                    "messages must be a non-empty array");
            }
            prompt = opt_.chat_template.apply(
                parse_anthropic_messages(body));
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
            ids = tok_.encode(prompt, true);
            input_tokens = static_cast<std::uint32_t>(ids.size());
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(
                json{{"type", "error"},
                     {"error",
                      {{"type", "invalid_request_error"},
                       {"message", e.what()}}}}
                    .dump(),
                "application/json");
            return;
        }

        const std::uint64_t id = loop_.submit(std::move(ids),
                                              max_tokens);
        const std::string rid = "msg_locus-" + std::to_string(id);

        if (!stream) {
            auto v = loop_.wait_done(id);
            if (v.status == engine::Status::kFailed) {
                res.status = 500;
                res.set_content(
                    json{{"type", "error"},
                         {"error",
                          {{"type", "api_error"},
                           {"message", v.error}}}}
                        .dump(),
                    "application/json");
                return;
            }
            const std::string text = tok_.decode(v.generated);
            const bool hit_eos =
                !v.generated.empty() &&
                v.generated.back() == tok_.eos_id();
            json out{
                {"id", rid},
                {"type", "message"},
                {"role", "assistant"},
                {"model", opt_.model_name},
                {"content", json::array({{{"type", "text"},
                                          {"text", text}}})},
                {"stop_reason", hit_eos ? "end_turn" : "max_tokens"},
                {"stop_sequence", nullptr},
                {"usage",
                 {{"input_tokens", input_tokens},
                  {"output_tokens",
                   static_cast<std::uint32_t>(
                       v.generated.size())}}}};
            res.set_content(out.dump(), "application/json");
            return;
        }

        // Anthropic SSE: message_start -> content_block_start ->
        // content_block_delta* -> content_block_stop ->
        // message_delta -> message_stop.
        struct St {
            std::size_t emitted = 0;
            bool started = false;
        };
        auto st = std::make_shared<St>();
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, id, rid, input_tokens, st](
                std::size_t, httplib::DataSink& sink) {
                std::string payload;
                if (!st->started) {
                    json start{
                        {"type", "message_start"},
                        {"message",
                         {{"id", rid},
                          {"type", "message"},
                          {"role", "assistant"},
                          {"model", opt_.model_name},
                          {"content", json::array()},
                          {"stop_reason", nullptr},
                          {"stop_sequence", nullptr},
                          {"usage",
                           {{"input_tokens", input_tokens},
                            {"output_tokens", 0}}}}}};
                    payload += "event: message_start\ndata: " +
                               start.dump() + "\n\n";
                    payload +=
                        "event: content_block_start\ndata: " +
                        json{{"type", "content_block_start"},
                             {"index", 0},
                             {"content_block",
                              {{"type", "text"},
                               {"text", ""}}}}
                            .dump() +
                        "\n\n";
                    st->started = true;
                }
                auto v = loop_.wait_progress(id, st->emitted);
                for (std::size_t i = st->emitted;
                     i < v.generated.size(); ++i) {
                    const tok::TokenId t = v.generated[i];
                    if (t == tok_.eos_id()) {
                        continue;
                    }
                    payload +=
                        "event: content_block_delta\ndata: " +
                        json{{"type", "content_block_delta"},
                             {"index", 0},
                             {"delta",
                              {{"type", "text_delta"},
                               {"text", tok_.decode({&t, 1})}}}}
                            .dump() +
                        "\n\n";
                }
                st->emitted = v.generated.size();
                const bool terminal =
                    v.status == engine::Status::kDone ||
                    v.status == engine::Status::kFailed;
                if (terminal) {
                    const bool hit_eos =
                        !v.generated.empty() &&
                        v.generated.back() == tok_.eos_id();
                    payload +=
                        "event: content_block_stop\ndata: " +
                        json{{"type", "content_block_stop"},
                             {"index", 0}}
                            .dump() +
                        "\n\n";
                    payload +=
                        "event: message_delta\ndata: " +
                        json{{"type", "message_delta"},
                             {"delta",
                              {{"stop_reason",
                                hit_eos ? "end_turn"
                                        : "max_tokens"},
                               {"stop_sequence", nullptr}}},
                             {"usage",
                              {{"output_tokens",
                                static_cast<std::uint32_t>(
                                    st->emitted)}}}}
                            .dump() +
                        "\n\n";
                    payload += "event: message_stop\ndata: " +
                               json{{"type", "message_stop"}}
                                   .dump() +
                               "\n\n";
                }
                if (!payload.empty() &&
                    !sink.write(payload.data(), payload.size())) {
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
    http_->Post("/v1/messages", handle_messages);
}

}  // namespace locus::server
