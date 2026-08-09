#include "locus/server/server.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "httplib.h"
#include "json.hpp"
#include "locus/auth/helper_spawn.hpp"
#include "locus/model/grammar.hpp"
#include "locus/server/tools.hpp"

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

/**
 * Reads sampling parameters common to both APIs from the request.
 * Unspecified -> greedy default (temperature 0), which keeps
 * responses deterministic unless the caller opts into sampling.
 * @returns {params, seed} (seed 0 = nondeterministic).
 */
std::pair<model::SamplingParams, std::uint64_t> parse_sampling(
    const json& body) {
    model::SamplingParams p;
    const auto num = [&](const char* k, float& dst) {
        if (body.contains(k) && body[k].is_number()) {
            dst = body[k].get<float>();
        }
    };
    num("temperature", p.temperature);
    num("top_p", p.top_p);
    num("min_p", p.min_p);
    num("repeat_penalty", p.repeat_penalty);
    num("frequency_penalty", p.frequency_penalty);
    num("presence_penalty", p.presence_penalty);
    if (body.contains("top_k") && body["top_k"].is_number_unsigned()) {
        p.top_k = body["top_k"].get<std::uint32_t>();
    }
    std::uint64_t seed = 0;
    if (body.contains("seed") && body["seed"].is_number_unsigned()) {
        seed = body["seed"].get<std::uint64_t>();
    }
    return {p, seed};
}

/** True when the caller asked for JSON output (OpenAI
 * response_format:{type:"json_object"} or Anthropic-style). */
bool wants_json(const json& body) {
    if (body.contains("response_format") &&
        body["response_format"].is_object()) {
        return body["response_format"].value("type", "") ==
               "json_object";
    }
    return false;
}

/**
 * Constrained-decoding filter that forces valid JSON output. Holds
 * the shared token->bytes table plus a live JsonGrammar; a candidate
 * token is allowed iff feeding its bytes keeps the JSON valid.
 */
class JsonTokenConstraint : public model::TokenConstraint {
  public:
    explicit JsonTokenConstraint(
        std::shared_ptr<const std::vector<std::string>> pieces)
        : pieces_(std::move(pieces)) {}

    bool allows(tok::TokenId t) const override {
        if (static_cast<std::size_t>(t) >= pieces_->size()) {
            return true;  // special/oob token (e.g. eos): allow
        }
        model::JsonGrammar g = grammar_;
        for (unsigned char c : (*pieces_)[t]) {
            if (!g.feed(c)) {
                return false;
            }
        }
        return true;
    }

    void commit(tok::TokenId t) override {
        if (static_cast<std::size_t>(t) >= pieces_->size()) {
            return;
        }
        for (unsigned char c : (*pieces_)[t]) {
            grammar_.feed(c);
        }
    }

  private:
    std::shared_ptr<const std::vector<std::string>> pieces_;
    model::JsonGrammar grammar_;
};

}  // namespace

OpenAiServer::OpenAiServer(const model::LlamaModel& m,
                           const tok::Tokenizer& tok,
                           Options opt)
    : model_(m),
      tok_(tok),
      opt_(std::move(opt)),
      loop_(m, tok.eos_id(), opt_.engine),
      http_(std::make_unique<httplib::Server>()),
      n_vocab_(m.hparams().n_vocab) {
    if (!opt_.auth_helper_argv.empty()) {
        std::vector<std::unique_ptr<auth::HelperConnection>> conns;
        for (std::size_t i = 0;
             i < std::max<std::size_t>(1, opt_.auth_helpers); ++i) {
            std::string err;
            auto c = auth::spawn_helper(opt_.auth_helper_argv, &err);
            if (!c) {
                throw std::runtime_error("auth helper spawn: " + err);
            }
            conns.push_back(std::move(c));
        }
        auth_ = std::make_unique<auth::AuthClient>(
            std::move(conns), auth_clock_, opt_.auth_cache);
    }
    // Bound the request body httplib buffers into memory before any
    // handler (and thus before authorize()) runs; without this the
    // default is SIZE_MAX -- an unauthenticated OOM.
    http_->set_payload_max_length(opt_.max_body_bytes);
    install_routes();
}

bool OpenAiServer::authorize(const httplib::Request& req,
                             httplib::Response& res,
                             std::string& identity) {
    if (!auth_) {
        identity.clear();
        return true;  // auth disabled -> open
    }
    // Accept "Authorization: Bearer <token>" or "x-api-key: <token>".
    std::string cred;
    if (req.has_header("Authorization")) {
        const std::string h = req.get_header_value("Authorization");
        const std::string kw = "Bearer ";
        cred = (h.rfind(kw, 0) == 0) ? h.substr(kw.size()) : h;
    } else if (req.has_header("x-api-key")) {
        cred = req.get_header_value("x-api-key");
    }
    auth::AuthClient::Result r =
        cred.empty() ? auth::AuthClient::Result{}
                     : auth_->resolve(cred);
    if (!r.ok) {
        res.status = 401;
        res.set_content(
            json{{"error",
                  {{"type", "authentication_error"},
                   {"message", "invalid or missing credential"}}}}
                .dump(),
            "application/json");
        return false;
    }
    identity = std::move(r.identity);
    return true;
}

void OpenAiServer::auth_event(const char* event,
                              const std::string& kind,
                              const std::string& identity) {
    if (!auth_) {
        return;
    }
    auth::HelperMessage ev;
    ev.type = "request";
    ev.set("event", event);
    ev.set("kind", kind);
    ev.set("model", opt_.model_name);
    if (!identity.empty()) {
        ev.set("identity", identity);
    }
    auth_->report_event(std::move(ev));
}

std::vector<float> OpenAiServer::embed_text(const std::string& text) {
    std::call_once(embed_once_, [this] {
        embed_cache_ = std::make_unique<kv::PagedKvCache>(
            model_.make_cache());
        embed_ws_ = model_.make_workspace();
    });
    const auto ids = tok_.encode(text, true);
    std::vector<float> emb(model_.hparams().n_embd);
    {
        std::lock_guard<std::mutex> lk(embed_mu_);
        kv::PagedKvCache::Seq seq;
        model_.embed(ids, *embed_cache_, seq, embed_ws_, emb);
        embed_cache_->release(seq);
    }
    double norm = 0.0;
    for (float v : emb) {
        norm += static_cast<double>(v) * v;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0) {
        for (float& v : emb) {
            v = static_cast<float>(v / norm);
        }
    }
    return emb;
}

std::shared_ptr<const std::vector<std::string>>
OpenAiServer::json_pieces() {
    std::call_once(pieces_once_, [this] {
        auto pieces = std::make_shared<std::vector<std::string>>();
        pieces->reserve(n_vocab_);
        for (std::uint32_t t = 0; t < n_vocab_; ++t) {
            const tok::TokenId id = static_cast<tok::TokenId>(t);
            pieces->push_back(tok_.decode({&id, 1}));
        }
        pieces_ = std::move(pieces);
    });
    return pieces_;
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

    // OpenAI model listing: locus serves exactly one model.
    http_->Get(
        "/v1/models",
        [this](const httplib::Request& req, httplib::Response& res) {
            // Gated when auth is on: the model id is deployment info.
            // (authorize() is a no-op / returns true when auth off.)
            std::string identity;
            if (!authorize(req, res, identity)) {
                return;
            }
            json entry{{"id", opt_.model_name},
                       {"object", "model"},
                       {"created", 0},
                       {"owned_by", "locus"}};
            res.set_content(
                json{{"object", "list"},
                     {"data", json::array({entry})}}
                    .dump(),
                "application/json");
        });
    // Retrieve a single model by id (OpenAI GET /v1/models/{id}).
    http_->Get(
        R"(/v1/models/(.+))",
        [this](const httplib::Request& req, httplib::Response& res) {
            std::string identity;
            if (!authorize(req, res, identity)) {
                return;
            }
            if (req.matches[1] != opt_.model_name) {
                res.status = 404;
                res.set_content(
                    make_error("model not found").dump(),
                    "application/json");
                return;
            }
            res.set_content(json{{"id", opt_.model_name},
                                 {"object", "model"},
                                 {"created", 0},
                                 {"owned_by", "locus"}}
                                .dump(),
                            "application/json");
        });

    // Prometheus text-format metrics: request/token counters plus
    // engine gauges (KV pool, prefix reuse, speculative accepts).
    http_->Get(
        opt_.metrics_path,
        [this](const httplib::Request& req, httplib::Response& res) {
            // Gated when auth is on: counters/gauges disclose traffic
            // and capacity. Open (no-op) in the default no-auth mode,
            // so unauthenticated scrapers keep working there.
            std::string identity;
            if (!authorize(req, res, identity)) {
                return;
            }
            const auto s = loop_.stats();
            std::string b;
            auto counter = [&](const char* name, const char* help,
                               std::uint64_t v) {
                b += "# HELP ";
                b += name;
                b += ' ';
                b += help;
                b += "\n# TYPE ";
                b += name;
                b += " counter\n";
                b += name;
                b += ' ';
                b += std::to_string(v);
                b += '\n';
            };
            auto gauge = [&](const char* name, const char* help,
                             std::uint64_t v) {
                b += "# HELP ";
                b += name;
                b += ' ';
                b += help;
                b += "\n# TYPE ";
                b += name;
                b += " gauge\n";
                b += name;
                b += ' ';
                b += std::to_string(v);
                b += '\n';
            };
            counter("locus_requests_total",
                    "Completion requests accepted.",
                    requests_total_.load());
            counter("locus_prompt_tokens_total",
                    "Prompt tokens submitted.",
                    prompt_tokens_total_.load());
            counter("locus_completion_tokens_total",
                    "Tokens generated.",
                    completion_tokens_total_.load());
            counter("locus_prefix_reused_tokens_total",
                    "Prompt tokens served from the prefix cache.",
                    s.prefix_reused_tokens);
            counter("locus_spec_accepted_tokens_total",
                    "Speculative draft tokens accepted.",
                    s.spec_accepted_tokens);
            counter("locus_spec_steps_total",
                    "Speculative verify forwards run.",
                    s.spec_steps);
            gauge("locus_kv_blocks_total", "KV cache pool size.",
                  s.total_blocks);
            gauge("locus_kv_blocks_free", "Free KV cache blocks.",
                  s.free_blocks);
            res.set_content(b, "text/plain; version=0.0.4");
        });

    // Both endpoints share one implementation; `chat` only changes
    // how the prompt is built and how the response is shaped.
    auto handle = [this](const httplib::Request& req,
                         httplib::Response& res, bool chat) {
        std::string identity;
        if (!authorize(req, res, identity)) {
            return;  // 401 already set
        }
        auth_event("create", chat ? "chat" : "completion", identity);
        json body;
        std::string prompt;
        std::uint32_t max_tokens = 16;
        bool stream = false;
        std::vector<ToolSpec> tools;
        try {
            body = json::parse(req.body);
            if (chat) {
                if (!body.contains("messages") ||
                    !body["messages"].is_array() ||
                    body["messages"].empty()) {
                    throw std::invalid_argument(
                        "messages must be a non-empty array");
                }
                auto msgs = parse_messages(body["messages"]);
                tools = parse_tools(body, /*anthropic=*/false);
                if (!tools.empty()) {
                    msgs.insert(msgs.begin(),
                                {"system",
                                 render_tools_system(tools)});
                }
                prompt = opt_.chat_template.apply(msgs);
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

        auto [sampling, seed] = parse_sampling(body);
        std::unique_ptr<model::TokenConstraint> constraint;
        if (wants_json(body)) {
            constraint = std::make_unique<JsonTokenConstraint>(
                json_pieces());
        }
        // logprobs: /v1/completions takes an integer count; /v1/chat/
        // completions takes a bool `logprobs` + integer `top_logprobs`.
        engine::LogprobsOpt lp;
        if (chat) {
            if (body.value("logprobs", false)) {
                lp.enabled = true;
                lp.top = body.value("top_logprobs", 0);
            }
        } else if (body.contains("logprobs") &&
                   body["logprobs"].is_number_unsigned()) {
            lp.enabled = true;
            lp.top = body["logprobs"].get<std::uint32_t>();
        }
        auto prompt_ids = tok_.encode(prompt, true);
        const std::uint32_t n_prompt =
            static_cast<std::uint32_t>(prompt_ids.size());
        // Reject a prompt the KV pool can never hold BEFORE it enters
        // the wait queue; an un-admissible request left waiting would
        // otherwise starve the queue (and, pre-fix, wedge the loop).
        if (const std::string why = loop_.admit_error(n_prompt);
            !why.empty()) {
            res.status = 400;
            res.set_content(make_error(why).dump(), "application/json");
            return;
        }
        requests_total_.fetch_add(1, std::memory_order_relaxed);
        prompt_tokens_total_.fetch_add(n_prompt,
                                       std::memory_order_relaxed);
        const std::uint64_t id = loop_.submit(
            std::move(prompt_ids), max_tokens, sampling, seed,
            std::move(constraint), lp);
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
            json choice;
            if (auto tc = detect_tool_call(text, tools)) {
                json call{{"id", "call_" + std::to_string(id)},
                          {"type", "function"},
                          {"function",
                           {{"name", tc->name},
                            {"arguments", tc->arguments.dump()}}}};
                choice = {{"index", 0},
                          {"message",
                           {{"role", "assistant"},
                            {"content", nullptr},
                            {"tool_calls", json::array({call})}}},
                          {"finish_reason", "tool_calls"}};
            } else if (chat) {
                choice = {{"index", 0},
                          {"message",
                           {{"role", "assistant"},
                            {"content", text}}},
                          {"finish_reason",
                           hit_eos ? "stop" : "length"}};
            } else {
                choice = {{"index", 0},
                          {"text", text},
                          {"finish_reason",
                           hit_eos ? "stop" : "length"}};
            }
            if (lp.enabled &&
                choice.value("finish_reason", "") != "tool_calls") {
                auto piece = [this](tok::TokenId t) {
                    return tok_.decode({&t, 1});
                };
                if (chat) {
                    // Chat shape: logprobs.content[] with per-token
                    // {token, logprob, top_logprobs[]}.
                    json content = json::array();
                    for (const auto& e : v.logprobs) {
                        json tops = json::array();
                        for (const auto& a : e.top) {
                            tops.push_back(
                                {{"token", piece(a.token)},
                                 {"logprob", a.logprob}});
                        }
                        content.push_back(
                            {{"token", piece(e.token)},
                             {"logprob", e.logprob},
                             {"top_logprobs", tops}});
                    }
                    choice["logprobs"] = {{"content", content}};
                } else {
                    // Completions shape: parallel arrays.
                    json toks = json::array();
                    json tlp = json::array();
                    json tops = json::array();
                    for (const auto& e : v.logprobs) {
                        toks.push_back(piece(e.token));
                        tlp.push_back(e.logprob);
                        json m = json::object();
                        for (const auto& a : e.top) {
                            m[piece(a.token)] = a.logprob;
                        }
                        tops.push_back(m);
                    }
                    choice["logprobs"] = {{"tokens", toks},
                                          {"token_logprobs", tlp},
                                          {"top_logprobs", tops}};
                }
            }
            const auto n_completion =
                static_cast<std::uint32_t>(v.generated.size());
            completion_tokens_total_.fetch_add(
                n_completion, std::memory_order_relaxed);
            json out{{"id", rid},
                     {"object", object},
                     {"model", opt_.model_name},
                     {"choices", json::array({choice})},
                     {"usage",
                      {{"prompt_tokens", n_prompt},
                       {"completion_tokens", n_completion},
                       {"total_tokens", n_prompt + n_completion}}}};
            res.set_content(out.dump(), "application/json");
            return;
        }

        // SSE streaming: one chunk per token, then [DONE]. When
        // tools are supplied (chat only) locus cannot recognize a
        // tool call until the whole completion is parsed, so it
        // BUFFERS -- emitting nothing token-by-token and, at the end,
        // either a tool_calls delta or the buffered text -- so a
        // streaming agent (Copilot CLI, etc.) never sees raw tool
        // JSON as content.
        auto state = std::make_shared<std::size_t>(0);
        auto tools_sp =
            std::make_shared<std::vector<ToolSpec>>(std::move(tools));
        const bool buffer_tools = chat && !tools_sp->empty();
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, id, chat, rid, sse_object, state, tools_sp,
             buffer_tools](std::size_t, httplib::DataSink& sink) {
                auto v = loop_.wait_progress(id, *state);
                std::string payload;
                if (!buffer_tools) {
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
                }
                *state = v.generated.size();
                const bool terminal =
                    v.status == engine::Status::kDone ||
                    v.status == engine::Status::kFailed;
                if (terminal) {
                    completion_tokens_total_.fetch_add(
                        v.generated.size(),
                        std::memory_order_relaxed);
                    if (buffer_tools) {
                        // Emit the final chat chunk: a tool_calls
                        // delta if the buffered completion parsed as
                        // a call, else the buffered text as content.
                        const bool hit_eos =
                            !v.generated.empty() &&
                            v.generated.back() == tok_.eos_id();
                        const std::string text =
                            tok_.decode(v.generated);
                        json delta;
                        std::string finish =
                            hit_eos ? "stop" : "length";
                        if (auto tc =
                                detect_tool_call(text, *tools_sp)) {
                            json call{
                                {"index", 0},
                                {"id",
                                 "call_" + std::to_string(id)},
                                {"type", "function"},
                                {"function",
                                 {{"name", tc->name},
                                  {"arguments",
                                   tc->arguments.dump()}}}};
                            delta = {{"role", "assistant"},
                                     {"tool_calls",
                                      json::array({call})}};
                            finish = "tool_calls";
                        } else {
                            delta = {{"content", text}};
                        }
                        json chunk{
                            {"id", rid},
                            {"object", sse_object},
                            {"choices",
                             json::array({{{"index", 0},
                                           {"delta", delta},
                                           {"finish_reason",
                                            finish}}})}};
                        payload += "data: " + chunk.dump() + "\n\n";
                    }
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
        std::string identity;
        if (!authorize(req, res, identity)) {
            return;
        }
        auth_event("create", "messages", identity);
        std::string prompt;
        std::uint32_t max_tokens = 16;
        std::uint32_t input_tokens = 0;
        bool stream = false;
        std::vector<tok::TokenId> ids;
        std::vector<ToolSpec> tools;
        model::SamplingParams sampling;
        std::uint64_t seed = 0;
        std::unique_ptr<model::TokenConstraint> constraint;
        try {
            const json body = json::parse(req.body);
            if (!body.contains("messages") ||
                !body["messages"].is_array() ||
                body["messages"].empty()) {
                throw std::invalid_argument(
                    "messages must be a non-empty array");
            }
            if (wants_json(body)) {
                constraint = std::make_unique<JsonTokenConstraint>(
                    json_pieces());
            }
            auto msgs = parse_anthropic_messages(body);
            tools = parse_tools(body, /*anthropic=*/true);
            if (!tools.empty()) {
                msgs.insert(msgs.begin(),
                            {"system", render_tools_system(tools)});
            }
            prompt = opt_.chat_template.apply(msgs);
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
            std::tie(sampling, seed) = parse_sampling(body);
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

        requests_total_.fetch_add(1, std::memory_order_relaxed);
        prompt_tokens_total_.fetch_add(input_tokens,
                                       std::memory_order_relaxed);
        const std::uint64_t id =
            loop_.submit(std::move(ids), max_tokens, sampling, seed,
                         std::move(constraint));
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
            json content;
            std::string stop_reason;
            if (auto tc = detect_tool_call(text, tools)) {
                content = json::array(
                    {{{"type", "tool_use"},
                      {"id", "toolu_locus-" + std::to_string(id)},
                      {"name", tc->name},
                      {"input", tc->arguments}}});
                stop_reason = "tool_use";
            } else {
                content = json::array(
                    {{{"type", "text"}, {"text", text}}});
                stop_reason = hit_eos ? "end_turn" : "max_tokens";
            }
            json out{
                {"id", rid},
                {"type", "message"},
                {"role", "assistant"},
                {"model", opt_.model_name},
                {"content", content},
                {"stop_reason", stop_reason},
                {"stop_sequence", nullptr},
                {"usage",
                 {{"input_tokens", input_tokens},
                  {"output_tokens",
                   static_cast<std::uint32_t>(
                       v.generated.size())}}}};
            completion_tokens_total_.fetch_add(
                v.generated.size(), std::memory_order_relaxed);
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
        auto tools_sp =
            std::make_shared<std::vector<ToolSpec>>(std::move(tools));
        // With tools, buffer until the whole completion is parsed:
        // defer the content_block_start until we know whether it is a
        // text or tool_use block (Anthropic streaming shape).
        const bool buffer_tools = !tools_sp->empty();
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, id, rid, input_tokens, st, tools_sp, buffer_tools](
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
                    if (!buffer_tools) {
                        payload +=
                            "event: content_block_start\ndata: " +
                            json{{"type", "content_block_start"},
                                 {"index", 0},
                                 {"content_block",
                                  {{"type", "text"},
                                   {"text", ""}}}}
                                .dump() +
                            "\n\n";
                    }
                    st->started = true;
                }
                auto v = loop_.wait_progress(id, st->emitted);
                if (!buffer_tools) {
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
                }
                st->emitted = v.generated.size();
                const bool terminal =
                    v.status == engine::Status::kDone ||
                    v.status == engine::Status::kFailed;
                if (terminal) {
                    const bool hit_eos =
                        !v.generated.empty() &&
                        v.generated.back() == tok_.eos_id();
                    std::string stop_reason =
                        hit_eos ? "end_turn" : "max_tokens";
                    if (buffer_tools) {
                        // Emit the deferred block now that the type
                        // is known: tool_use if the buffered text
                        // parsed as a call, else a text block.
                        const std::string text =
                            tok_.decode(v.generated);
                        if (auto tc =
                                detect_tool_call(text, *tools_sp)) {
                            payload +=
                                "event: content_block_start\n"
                                "data: " +
                                json{{"type", "content_block_start"},
                                     {"index", 0},
                                     {"content_block",
                                      {{"type", "tool_use"},
                                       {"id",
                                        "toolu_locus-" +
                                            std::to_string(id)},
                                       {"name", tc->name},
                                       {"input", json::object()}}}}
                                    .dump() +
                                "\n\n";
                            payload +=
                                "event: content_block_delta\n"
                                "data: " +
                                json{{"type", "content_block_delta"},
                                     {"index", 0},
                                     {"delta",
                                      {{"type", "input_json_delta"},
                                       {"partial_json",
                                        tc->arguments.dump()}}}}
                                    .dump() +
                                "\n\n";
                            stop_reason = "tool_use";
                        } else {
                            payload +=
                                "event: content_block_start\n"
                                "data: " +
                                json{{"type", "content_block_start"},
                                     {"index", 0},
                                     {"content_block",
                                      {{"type", "text"},
                                       {"text", ""}}}}
                                    .dump() +
                                "\n\n";
                            payload +=
                                "event: content_block_delta\n"
                                "data: " +
                                json{{"type", "content_block_delta"},
                                     {"index", 0},
                                     {"delta",
                                      {{"type", "text_delta"},
                                       {"text", text}}}}
                                    .dump() +
                                "\n\n";
                        }
                    }
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
                              {{"stop_reason", stop_reason},
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
                    completion_tokens_total_.fetch_add(
                        st->emitted, std::memory_order_relaxed);
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

    // OpenAI embeddings: pooled hidden state, L2-normalized. `input`
    // is a string or an array of strings.
    http_->Post(
        "/v1/embeddings",
        [this](const httplib::Request& req, httplib::Response& res) {
            std::string identity;
            if (!authorize(req, res, identity)) {
                return;
            }
            auth_event("create", "embeddings", identity);
            try {
                const json body = json::parse(req.body);
                std::vector<std::string> inputs;
                if (body.contains("input") &&
                    body["input"].is_string()) {
                    inputs.push_back(
                        body["input"].get<std::string>());
                } else if (body.contains("input") &&
                           body["input"].is_array()) {
                    if (body["input"].size() >
                        opt_.max_embeddings_inputs) {
                        throw std::invalid_argument(
                            "input array exceeds the " +
                            std::to_string(
                                opt_.max_embeddings_inputs) +
                            "-element limit");
                    }
                    for (const auto& s : body["input"]) {
                        inputs.push_back(s.get<std::string>());
                    }
                }
                if (inputs.empty()) {
                    throw std::invalid_argument(
                        "input must be a string or array of "
                        "strings");
                }
                json data = json::array();
                for (std::size_t i = 0; i < inputs.size(); ++i) {
                    data.push_back(
                        {{"object", "embedding"},
                         {"index", i},
                         {"embedding", embed_text(inputs[i])}});
                }
                res.set_content(
                    json{{"object", "list"},
                         {"data", data},
                         {"model", opt_.model_name}}
                        .dump(),
                    "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(make_error(e.what()).dump(),
                                "application/json");
            }
        });
}

}  // namespace locus::server
