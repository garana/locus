#include "locus/server/server.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "json.hpp"
#include "locus/audio/snac.hpp"
#include "locus/audio/wav.hpp"
#include "locus/auth/helper_spawn.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/http/base64.hpp"
#include "locus/backend/fallbacks.hpp"
#include "locus/backend/gpu_metrics.hpp"
#include "locus/backend/variants.hpp"
#include "locus/image/image_input.hpp"
#include "locus/image/janus.hpp"
#include "locus/image/janus_chat.hpp"
#include "locus/image/janus_vision.hpp"
#include "locus/image/flux_pipeline.hpp"
#include "locus/image/pixart_pipeline.hpp"
#include "locus/image/png.hpp"
#include "locus/image/vq16.hpp"
#include "locus/model/codec.hpp"
#include "locus/model/grammar.hpp"
#include "locus/model/sampling.hpp"
#include "locus/server/errors.hpp"
#include "locus/server/metrics_text.hpp"
#include "locus/server/responses.hpp"
#include "locus/server/tools.hpp"
#include "locus/sys/features.hpp"
#include "locus/sys/perf_counters.hpp"

namespace locus::server {

namespace {

using nlohmann::json;

/** dump()'s error handler for every response body: invalid UTF-8 (a
 * parse error quoting the request, text cut mid-character) becomes
 * U+FFFD instead of throwing type_error.316 out of a handler (i#243). */
constexpr auto kLossy = json::error_handler_t::replace;

json make_error(const std::string& message) {
    return json{{"error", {{"message", message},
                           {"type", "invalid_request_error"}}}};
}

/** Fills res for an exception a handler caught: 400 with
 * invalid_request_error for a client error, else 500 with server_error
 * (see http_status_for). */
void set_error(http::ServerResponse& res, const std::exception& e) {
    res.status = http_status_for(e);
    const char* type =
        res.status == 400 ? "invalid_request_error" : "server_error";
    res.set_content(json{{"error", {{"message", e.what()}, {"type", type}}}}
                        .dump(-1, ' ', false, kLossy),
                    "application/json");
}

/**
 * Fills res with a 503: the feature is off, or busy. The type is
 * server_error, so a client that branches on it does not take the
 * server's state for its own bad request; a busy server also sends
 * Retry-After (in seconds).
 */
void set_unavailable(http::ServerResponse& res, const std::string& message,
                     int retry_after_s = 0) {
    res.status = 503;
    if (retry_after_s > 0) {
        res.set_header("Retry-After", std::to_string(retry_after_s));
    }
    res.set_content(json{{"error", {{"message", message},
                                    {"type", "server_error"}}}}
                        .dump(-1, ' ', false, kLossy),
                    "application/json");
}

/**
 * The text of an OpenAI message's content: a string, or an array of
 * parts (the multimodal shape, which some clients send for plain text
 * too) whose "text" parts are joined with newlines, as the vision path
 * joins them. Image requests never get here (parse_vision_chat takes
 * them first), so any other part type is unsupported.
 */
std::string openai_text(const json& content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (!content.is_array()) {
        throw std::invalid_argument(
            "message content must be a string or an array of parts");
    }
    std::string out;
    for (const auto& p : content) {
        const std::string type = p.is_object() ? p.value("type", "") : "";
        if (type != "text" || !p.contains("text") || !p["text"].is_string()) {
            throw std::invalid_argument("unsupported content part: " +
                                        (type.empty() ? "(none)" : type));
        }
        if (!out.empty()) out += "\n";
        out += p["text"].get<std::string>();
    }
    return out;
}

/** Parses and validates the OpenAI messages array. */
std::vector<chat::Message> parse_messages(const json& messages) {
    std::vector<chat::Message> out;
    for (const auto& m : messages) {
        if (!m.contains("content")) {
            throw std::invalid_argument("message content is missing");
        }
        chat::Message msg;
        msg.role = m.value("role", "user");
        if (msg.role != "system" && msg.role != "user" &&
            msg.role != "assistant") {
            throw std::invalid_argument(
                "unsupported message role: " + msg.role);
        }
        msg.content = openai_text(m["content"]);
        out.push_back(std::move(msg));
    }
    return out;
}

/**
 * Anthropic content is either a plain string or an array of content
 * blocks; we take the text of every {"type":"text"} block and
 * ignore the rest (images, tool_use/tool_result -- no tool support
 * yet). @returns the text blocks joined with newlines, as the vision
 * path joins them (Claude Code sends its system prompt as several
 * blocks; with no separator their words would run together).
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
                if (!out.empty()) out += "\n";
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
 * one leading system message, then each user/assistant entry of
 * messages[]. The system text is the top-level `system` (string or
 * blocks) followed by every system-role entry of messages[], joined
 * with newlines: Claude Code sends one after the first user turn,
 * and chat templates expect a single system message first.
 */
std::vector<chat::Message> parse_anthropic_messages(
    const json& body) {
    std::vector<chat::Message> out;
    std::string system;
    if (body.contains("system") && !body["system"].is_null()) {
        system = anthropic_text(body["system"]);
    }
    for (const auto& m : body.at("messages")) {
        chat::Message msg;
        msg.role = m.value("role", "user");
        if (msg.role != "user" && msg.role != "assistant" &&
            msg.role != "system") {
            throw std::invalid_argument(
                "unsupported message role: " + msg.role);
        }
        if (!m.contains("content")) {
            throw std::invalid_argument(
                "message must have content");
        }
        msg.content = anthropic_text(m["content"]);
        if (msg.role == "system") {
            if (!system.empty() && !msg.content.empty()) system += "\n";
            system += msg.content;
            continue;
        }
        out.push_back(std::move(msg));
    }
    if (!system.empty()) {
        out.insert(out.begin(), chat::Message{"system", system});
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

/**
 * The per-route HTTP series (i#285): requests by route and status code
 * (only codes seen), the request and response body sizes, the time from
 * head received to response written, and the requests in flight.
 */
void render_http_metrics(MetricsText& mt, const http::HttpMetrics& hm) {
    constexpr auto kRelaxed = std::memory_order_relaxed;
    const auto names = hm.names();
    mt.family("locus_http_requests_total",
              "HTTP requests answered, by route and status code.",
              "counter");
    for (std::size_t i = 0; i < names.size(); ++i) {
        const auto& r = hm.at(i);
        const std::string route = MetricsText::label("route", names[i]);
        for (std::size_t c = 0; c < r.codes.size(); ++c) {
            const std::uint64_t v = r.codes[c].load(kRelaxed);
            if (v == 0) continue;
            mt.value("locus_http_requests_total",
                     route + "," + MetricsText::label(
                                       "code", std::to_string(
                                                   http::HttpMetrics::kMinCode +
                                                   static_cast<int>(c))),
                     v);
        }
        if (const std::uint64_t v = r.out_of_range.load(kRelaxed); v != 0) {
            mt.value("locus_http_requests_total",
                     route + "," + MetricsText::label("code", "other"), v);
        }
    }
    const auto per_route = [&](const char* name, const char* help,
                               auto pick) {
        mt.family(name, help, "histogram");
        for (std::size_t i = 0; i < names.size(); ++i) {
            mt.histogram_series(name, MetricsText::label("route", names[i]),
                                pick(hm.at(i)).snapshot());
        }
    };
    per_route("locus_http_request_bytes", "Request body size, per route.",
              [](const auto& r) -> const auto& { return r.request_bytes; });
    per_route("locus_http_response_bytes",
              "Response body bytes written (stream payload included), per "
              "route.",
              [](const auto& r) -> const auto& { return r.response_bytes; });
    per_route("locus_http_request_seconds",
              "Request head received to response written, per route.",
              [](const auto& r) -> const auto& { return r.seconds; });
    const std::int64_t f = hm.in_flight.load(kRelaxed);
    mt.gauge("locus_http_requests_in_flight",
             "Requests between head received and response written.",
             f > 0 ? static_cast<std::uint64_t>(f) : 0);
}

/**
 * The forward's wall time per stage and per (layer, stage) (i#285).
 * The per-layer series list each per-layer stage that has any time, for
 * every layer; embed, head and sample belong to no layer.
 */
void render_stage_metrics(MetricsText& mt, const model::StageMetrics& sm) {
    using model::Stage;
    constexpr auto kStages = static_cast<int>(Stage::kCount);
    const auto secs = [](std::uint64_t ns) {
        return static_cast<double>(ns) / 1e9;
    };
    mt.family("locus_stage_seconds_total",
              "Forward wall time per stage (CPU-side; all layers).",
              "counter");
    for (int i = 0; i < kStages; ++i) {
        const auto s = static_cast<Stage>(i);
        mt.value("locus_stage_seconds_total",
                 MetricsText::label("stage", model::stage_name(s)),
                 secs(sm.stage_ns(s)));
    }
    mt.family("locus_layer_seconds_total",
              "Forward wall time per layer and stage (CPU-side).",
              "counter");
    for (const Stage s : {Stage::kNorm, Stage::kAttention, Stage::kFfn,
                          Stage::kMoe, Stage::kLayer, Stage::kPrefetch}) {
        std::uint64_t any = 0;
        for (std::uint32_t l = 0; l < sm.layers(); ++l) any += sm.ns(l, s);
        if (any == 0) continue;  // e.g. moe on a dense model
        const std::string stage = MetricsText::label("stage",
                                                     model::stage_name(s));
        for (std::uint32_t l = 0; l < sm.layers(); ++l) {
            mt.value("locus_layer_seconds_total",
                     MetricsText::label("layer", std::to_string(l)) + "," +
                         stage,
                     secs(sm.ns(l, s)));
        }
    }
}

/**
 * Writes the GPU counters (i#285) of each API this process used: its
 * submits, kernels, host recording and wait time, device time and the
 * memory it holds per kind. An API with no submits and no memory is
 * left out.
 */
void render_gpu_metrics(MetricsText& mt) {
    using backend::GpuApi;
    using backend::MemKind;
    constexpr auto kRelaxed = std::memory_order_relaxed;
    const auto secs = [](std::uint64_t ns) {
        return static_cast<double>(ns) / 1e9;
    };
    std::vector<GpuApi> used;
    for (int a = 0; a < static_cast<int>(GpuApi::kCount); ++a) {
        const backend::GpuMetrics& gm =
            backend::gpu_metrics(static_cast<GpuApi>(a));
        bool any = gm.submits.load(kRelaxed) != 0;
        for (int k = 0; k < static_cast<int>(MemKind::kCount); ++k) {
            any = any || gm.held(static_cast<MemKind>(k)) != 0;
        }
        if (any) used.push_back(static_cast<GpuApi>(a));
    }
    if (used.empty()) return;
    const auto api = [](GpuApi a) {
        return MetricsText::label("api", backend::gpu_api_name(a));
    };
    const auto counter = [&](const char* name, const char* help,
                             auto get) {
        mt.family(name, help, "counter");
        for (const GpuApi a : used) {
            mt.value(name, api(a), get(backend::gpu_metrics(a)));
        }
    };
    counter("locus_gpu_submits_total",
            "Command batches (vulkan) or synchronizing calls (cuda).",
            [&](const backend::GpuMetrics& gm) {
                return static_cast<double>(gm.submits.load(kRelaxed));
            });
    counter("locus_gpu_dispatches_total", "Kernels recorded (vulkan).",
            [&](const backend::GpuMetrics& gm) {
                return static_cast<double>(gm.dispatches.load(kRelaxed));
            });
    counter("locus_gpu_record_seconds_total",
            "Host time recording batches, begin to submit (vulkan).",
            [&](const backend::GpuMetrics& gm) {
                return secs(gm.record_ns.load(kRelaxed));
            });
    counter("locus_gpu_wait_seconds_total",
            "Host time from submit until the device is done.",
            [&](const backend::GpuMetrics& gm) {
                return secs(gm.wait_ns.load(kRelaxed));
            });
    counter("locus_gpu_device_seconds_total",
            "Device time by GPU timestamps, estimated from sampled batches "
            "(vulkan).",
            [&](const backend::GpuMetrics& gm) {
                return secs(gm.device_ns.load(kRelaxed));
            });
    counter("locus_gpu_device_samples_total",
            "Batches timed for the device time estimate (vulkan).",
            [&](const backend::GpuMetrics& gm) {
                return static_cast<double>(gm.device_samples.load(kRelaxed));
            });
    mt.family("locus_gpu_memory_bytes",
              "Device memory this process holds, per kind.", "gauge");
    for (const GpuApi a : used) {
        const backend::GpuMetrics& gm = backend::gpu_metrics(a);
        for (int k = 0; k < static_cast<int>(MemKind::kCount); ++k) {
            const auto kind = static_cast<MemKind>(k);
            mt.value("locus_gpu_memory_bytes",
                     api(a) + "," +
                         MetricsText::label("kind",
                                            backend::mem_kind_name(kind)),
                     static_cast<double>(gm.held(kind)));
        }
    }
}

/**
 * Writes locus_fallbacks_total{from,to,reason} (i#285): how often a
 * faster path handed its work to a slower one. Only fallbacks that
 * happened at least once have a series.
 */
void render_fallbacks(MetricsText& mt) {
    mt.family("locus_fallbacks_total",
              "Work a GPU path handed to a slower path, per reason.",
              "counter");
    backend::for_each_fallback([&](const backend::Fallback& f) {
        mt.value("locus_fallbacks_total",
                 MetricsText::label("from", f.from) + "," +
                     MetricsText::label("to", f.to) + "," +
                     MetricsText::label("reason", f.reason),
                 static_cast<double>(f.n.load(std::memory_order_relaxed)));
    });
}

/** @returns The --kv-type name of `t`, for the locus_info series. */
std::string kv_type_name(kv::KvType t) {
    switch (t) {
        case kv::KvType::kF32: return "f32";
        case kv::KvType::kF16: return "f16";
        case kv::KvType::kQ8: return "q8";
        case kv::KvType::kQ4: return "q4";
    }
    return "f32";
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

/** A chat request with images (i#241), parsed for vision_chat. */
struct VisionChat {
    bool has_system = false;  ///< the request set a system prompt
    std::string system;
    std::vector<image::ChatTurn> turns;
    std::vector<std::vector<std::byte>> images;  ///< encoded, in order
    std::uint32_t max_tokens = 16;
    model::SamplingParams sampling;
    std::uint64_t seed = 0;
};

namespace {

/** The bytes of an OpenAI image_url: a base64 data: URL only. */
std::vector<std::byte> data_url_bytes(const json& image_url,
                                      std::size_t max_bytes) {
    const json& u = image_url.is_object() ? image_url.value("url", json())
                                          : image_url;
    if (!u.is_string()) {
        throw std::invalid_argument("image_url.url must be a string");
    }
    const std::string url = u.get<std::string>();
    if (url.rfind("data:", 0) != 0) {
        throw std::invalid_argument(
            "image_url must be a data: URL (locus does not fetch URLs)");
    }
    const std::size_t comma = url.find(',');
    if (comma == std::string::npos || comma < 12 ||
        url.compare(comma - 7, 7, ";base64") != 0) {
        throw std::invalid_argument("image_url data: URL must be base64");
    }
    return http::base64_decode(std::string_view(url).substr(comma + 1),
                               max_bytes);
}

/** @returns Whether a message's content holds an image part. */
bool has_image_part(const json& content, bool anthropic) {
    if (!content.is_array()) return false;
    for (const auto& p : content) {
        const std::string type = p.value("type", "");
        if (type == (anthropic ? "image" : "image_url")) return true;
    }
    return false;
}

/**
 * One turn's text from its content: text parts as they are, each image
 * as the placeholder, joined by "\n" (the reference's
 * "<image_placeholder>\n<question>" form); image bytes are appended to
 * `images`. A text part may not contain the placeholder itself.
 */
std::string vision_turn_text(const json& content, bool anthropic,
                             bool allow_images, std::size_t max_bytes,
                             std::vector<std::vector<std::byte>>& images) {
    const auto checked = [](std::string t) {
        if (t.find(image::kImagePlaceholder) != std::string::npos) {
            throw std::invalid_argument(
                "message text may not contain <image_placeholder>");
        }
        return t;
    };
    if (content.is_string()) return checked(content.get<std::string>());
    if (!content.is_array()) {
        throw std::invalid_argument(
            "content must be a string or an array of parts");
    }
    std::string out;
    for (const auto& p : content) {
        const std::string type = p.value("type", "");
        std::string piece;
        if (type == "text") {
            piece = checked(p.at("text").get<std::string>());
        } else if (type == (anthropic ? "image" : "image_url")) {
            if (!allow_images) {
                throw std::invalid_argument(
                    "images are only accepted in user messages");
            }
            if (anthropic) {
                const json& src = p.at("source");
                if (src.value("type", "") != "base64") {
                    throw std::invalid_argument(
                        "image source must be base64 (locus does not "
                        "fetch URLs)");
                }
                images.push_back(http::base64_decode(
                    src.at("data").get<std::string>(), max_bytes));
            } else {
                images.push_back(
                    data_url_bytes(p.at("image_url"), max_bytes));
            }
            piece = std::string(image::kImagePlaceholder);
        } else {
            throw std::invalid_argument("unsupported content part: " +
                                        type);
        }
        if (!out.empty()) out += "\n";
        out += piece;
    }
    return out;
}

/**
 * Parses a chat (OpenAI messages[]) or messages (Anthropic, with a
 * top-level system) body that carries images.
 *
 * @returns nullopt when no message has an image part, so the request
 *     takes the text path unchanged.
 */
std::optional<VisionChat> parse_vision_chat(const json& body,
                                            bool anthropic,
                                            std::uint32_t max_tokens_cap,
                                            std::size_t max_bytes) {
    const json& msgs = body.at("messages");
    bool any = false;
    for (const auto& m : msgs) {
        if (m.contains("content") && has_image_part(m["content"], anthropic)) {
            any = true;
        }
    }
    if (!any) return std::nullopt;

    VisionChat vc;
    if (anthropic && body.contains("system") && !body["system"].is_null()) {
        vc.system = anthropic_text(body["system"]);
        vc.has_system = true;
        // Like every turn's text: a typed placeholder would add an
        // image slot with no image.
        if (vc.system.find(image::kImagePlaceholder) != std::string::npos) {
            throw std::invalid_argument(
                "system text may not contain <image_placeholder>");
        }
    }
    for (const auto& m : msgs) {
        const std::string role = m.value("role", "user");
        if (!m.contains("content")) {
            throw std::invalid_argument("message must have content");
        }
        if (role == "system") {
            // Anthropic: as parse_anthropic_messages, after the
            // top-level system text.
            const std::string text = vision_turn_text(
                m["content"], anthropic, false, max_bytes, vc.images);
            if (!vc.system.empty() && !text.empty()) vc.system += "\n";
            vc.system += text;
            vc.has_system = true;
        } else if (role == "user" || role == "assistant") {
            vc.turns.push_back(
                {role, vision_turn_text(m["content"], anthropic,
                                        role == "user", max_bytes,
                                        vc.images)});
        } else {
            throw std::invalid_argument("unsupported message role: " + role);
        }
    }
    if (body.value("stream", false)) {
        throw std::invalid_argument(
            "stream is not supported with image input yet");
    }
    if (body.contains("tools") || wants_json(body)) {
        throw std::invalid_argument(
            "tools and JSON mode are not supported with image input yet");
    }
    if (body.contains("max_tokens")) {
        const auto& v = body["max_tokens"];
        if (!v.is_number_unsigned() || v.get<std::uint64_t>() == 0) {
            throw std::invalid_argument(
                "max_tokens must be a positive integer");
        }
        vc.max_tokens = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(v.get<std::uint64_t>(), max_tokens_cap));
    }
    std::tie(vc.sampling, vc.seed) = parse_sampling(body);
    return vc;
}

}  // namespace

engine::Engine::Config OpenAiServer::resolve_engine_cfg(
    const Options& o) {
    engine::Engine::Config cfg = o.engine;
    cfg.prefix_cache = cfg.prefix_cache || o.prompt_cache;
    return cfg;
}

OpenAiServer::OpenAiServer(const model::TransformerModel& m,
                           const tok::Tokenizer& tok,
                           Options opt)
    : model_(m),
      tok_(tok),
      opt_(std::move(opt)),
      loop_(m, tok.eos_id(), resolve_engine_cfg(opt_)),
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
    if (!opt_.audio_decoder_path.empty()) {
        auto g = gguf::GgufFile::open(opt_.audio_decoder_path);
        snac_ = std::make_unique<audio::SnacDecoder>(
            audio::SnacDecoder::load(g));
        // Resolve the Orpheus audio band base by name; a non-TTS model
        // lacks <custom_token_10>, which leaves the endpoint a 503.
        try {
            audio_band_base_ = model::orpheus_band_base(tok_);
        } catch (const std::exception&) {
            audio_band_base_ = -1;
        }
        // Speech samples only Orpheus's own tokens, <custom_token_0>
        // (10 below the band: the stop and marker ids) through the
        // end of the 7 code bands, so the head computes just those
        // rows (i#173). If the vocab is not laid out that way, keep
        // the whole head.
        if (audio_band_base_ >= 10) {
            const auto first =
                static_cast<std::uint32_t>(audio_band_base_ - 10);
            const std::uint32_t end = std::min<std::uint32_t>(
                n_vocab_, static_cast<std::uint32_t>(audio_band_base_) +
                              7u * 4096u);
            if (tok_.piece(static_cast<tok::TokenId>(first)) ==
                "<custom_token_0>") {
                audio_rows_ = {first, end};
            }
        }
    }
    if (!opt_.image_decoder_path.empty()) {
        auto g = gguf::GgufFile::open(opt_.image_decoder_path);
        vq16_ = std::make_unique<image::Vq16Decoder>(
            image::Vq16Decoder::load(g));
        // A model without janus.image_tokens is not Janus and leaves the
        // endpoint a 503; one that has it but fails to load is a broken
        // file and stops the server here.
        if (!opt_.model_path.empty()) {
            auto mg = std::make_unique<gguf::GgufFile>(
                gguf::GgufFile::open(opt_.model_path));
            if (mg->find("janus.image_tokens") != nullptr) {
                janus_ = std::make_unique<image::JanusImageGen>(
                    image::JanusImageGen::load(*mg, m.hparams().n_embd));
                janus_gguf_ = std::move(mg);
            }
        }
    }
    // A broken image pipeline directory stops the server here too. A
    // flux GGUF in it makes it a FLUX.1 pipeline, else PixArt.
    if (!opt_.image_model_dir.empty()) {
        bool flux = false;
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(
                 opt_.image_model_dir, ec)) {
            if (e.is_regular_file() && e.path().extension() == ".gguf" &&
                gguf::GgufFile::open(e.path().string())
                        .get_string("general.architecture") == "flux") {
                flux = true;
            }
        }
        if (flux) {
            flux_ = image::FluxPipeline::load_dir(opt_.image_model_dir);
        } else {
            pixart_ = image::PixArtPipeline::load_dir(opt_.image_model_dir);
        }
    }
    // Image input (i#241): on when the model's GGUF carries the Janus
    // vision tower. A tower that is present but broken stops the
    // server here, like the image heads above.
    if (!opt_.model_path.empty()) {
        auto vg = std::make_unique<gguf::GgufFile>(
            gguf::GgufFile::open(opt_.model_path));
        if (vg->find_tensor("vit.patch_embd.weight") != nullptr) {
            vision_ = std::make_unique<image::JanusVision>(
                image::JanusVision::load(*vg, m.hparams().n_embd));
            vision_gguf_ = std::move(vg);
        }
    }
    // Hardware names for /metrics (i#285), read once here: the GPU
    // names take the GPU lock, which a scrape must never wait on. Only
    // devices this server uses: the language model's GPU backend, and
    // Vulkan when the PixArt models run there (see handle_pixart).
    cpu_name_ = sys::cpu_name();
    const std::string_view be = m.active_backend().name;
    if (be == "vulkan") {
        devices_.push_back({"vulkan", backend::vulkan_device_name()});
    } else if (be == "cuda") {
        devices_.push_back({"cuda", backend::cuda_device_name()});
    }
    if ((pixart_ || flux_) && be != "vulkan" &&
        backend::vulkan_imports_host_memory()) {
        devices_.push_back({"vulkan", backend::vulkan_device_name()});
    }
    install_routes();
}

bool OpenAiServer::authorize(const http::Request& req,
                             http::ServerResponse& res,
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
                .dump(-1, ' ', false, kLossy),
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

void OpenAiServer::vision_chat(bool anthropic, const VisionChat& chat,
                               http::ServerResponse& res,
                               const http::RequestContext& ctx) {
    // Each API's own error envelope.
    const auto fail = [&](int status, const std::string& msg) {
        res.status = status;
        const char* type = status == 400
                               ? "invalid_request_error"
                               : (anthropic ? "api_error" : "server_error");
        const json body =
            anthropic ? json{{"type", "error"},
                             {"error", {{"type", type}, {"message", msg}}}}
                      : json{{"error", {{"message", msg}, {"type", type}}}};
        res.set_content(body.dump(-1, ' ', false, kLossy),
                        "application/json");
    };
    if (!vision_) {
        fail(400, "this model does not accept image input");
        return;
    }
    const auto alive = [&] { return !ctx.client_alive || ctx.client_alive(); };
    try {
        // The image tower runs one matvec per token per linear. A GPU
        // backend would make each one a separate device call (on vulkan
        // the tower took twice as long as on the CPU), so it runs on the
        // best CPU backend unless the active one is a CPU backend.
        const auto& active = model_.active_backend().ops;
        const auto& op = backend::splits_rows(active)
                             ? active
                             : backend::best_backend().ops;
        const auto& hp = model_.hparams();
        std::lock_guard<std::mutex> lk(vision_mu_);

        // Each image: decode, the reference preprocessing (the long side
        // to image_size, at least min_size 14, padded with gray 127, as
        // in the Janus preprocessor_config.json), then tower + aligner.
        image::DecodeLimits lim;
        lim.max_bytes = opt_.max_body_bytes;
        std::vector<std::vector<float>> embs;
        for (const auto& bytes : chat.images) {
            const auto sq = image::fit_square(image::decode_image(bytes, lim),
                                              vision_->image_size(), 14,
                                              {127, 127, 127});
            embs.push_back(vision_->align(
                vision_->features(vision_->to_pixels(sq.rgb), op), op));
        }
        const std::string text = image::janus_chat_text(
            chat.turns,
            chat.has_system ? std::string_view(chat.system)
                            : image::kJanusSystemPrompt,
            tok_.piece(tok_.eos_id()));
        std::size_t n_images = 0;
        const auto ids = image::janus_chat_ids(tok_, text, *vision_, n_images);
        if (n_images != embs.size()) {
            throw std::runtime_error("vision chat: image count mismatch");
        }

        if (!vision_cache_) {
            vision_cache_ =
                std::make_unique<kv::PagedKvCache>(model_.make_cache());
            vision_ws_ = model_.make_workspace();
        }
        kv::PagedKvCache::Seq seq;
        struct Release {
            kv::PagedKvCache& c;
            kv::PagedKvCache::Seq& s;
            ~Release() { c.release(s); }
        } release{*vision_cache_, seq};
        const auto need =
            static_cast<std::uint32_t>(ids.size() + chat.max_tokens);
        if (!vision_cache_->ensure_capacity(seq, need)) {
            throw std::invalid_argument(
                "the prompt (" + std::to_string(ids.size()) +
                " tokens with the images) plus max_tokens does not fit the "
                "context");
        }
        requests_total_.fetch_add(1, std::memory_order_relaxed);
        prompt_tokens_total_.fetch_add(ids.size(), std::memory_order_relaxed);

        // Prefill. The image's embeddings take its placeholder slots in
        // order. Where the backend has a batched prefill the prompt
        // runs in chunks (i#257: each weight read once per chunk, not
        // once per position); otherwise one position at a time, where
        // only the last position computes the whole head (the others
        // need no logits, so one row keeps the head cheap).
        std::vector<float> logits(n_vocab_);
        std::vector<const float*> embd(ids.size(), nullptr);
        {
            std::size_t img = 0, row = 0;
            for (std::size_t i = 0; i < ids.size(); ++i) {
                if (ids[i] != vision_->placeholder()) continue;
                embd[i] = embs[img].data() + row * hp.n_embd;
                if (++row == vision_->tokens()) {
                    row = 0;
                    ++img;
                }
            }
        }
        if (model_.supports_batch_prefill()) {
            // At most 64 positions between client checks: on vulkan a
            // chunk with embeddings runs one GPU forward per position,
            // so its 512-token prefill_chunk would check almost never.
            const std::size_t chunk =
                std::min<std::size_t>(model_.prefill_chunk(), 64);
            for (std::size_t at = 0; at < ids.size(); at += chunk) {
                if (!alive()) {
                    res.status = 499;
                    return;
                }
                const std::size_t nb = std::min(chunk, ids.size() - at);
                model_.forward_batch(
                    std::span<const tok::TokenId>(ids.data() + at, nb),
                    *vision_cache_, seq, vision_ws_, logits, false,
                    std::span<const float* const>(embd.data() + at, nb));
            }
        } else {
            for (std::size_t i = 0; i < ids.size(); ++i) {
                if (!alive()) {
                    res.status = 499;
                    return;
                }
                const model::LogitRows rows = i + 1 == ids.size()
                                                  ? model::LogitRows{}
                                                  : model::LogitRows{0, 1};
                const auto emb =
                    embd[i] != nullptr
                        ? std::span<const float>(embd[i], hp.n_embd)
                        : std::span<const float>();
                model_.forward_layers(embd[i] != nullptr ? 0 : ids[i], emb,
                                      0, hp.n_layers, *vision_cache_, seq,
                                      vision_ws_, logits, rows);
            }
        }

        // Decode to eos or max_tokens.
        std::mt19937_64 rng(chat.seed != 0 ? chat.seed
                                           : std::random_device{}());
        std::vector<tok::TokenId> history(ids.begin(), ids.end());
        std::vector<tok::TokenId> out;
        bool hit_eos = false;
        for (std::uint32_t n = 0; n < chat.max_tokens; ++n) {
            if (!alive()) {
                res.status = 499;
                return;
            }
            const tok::TokenId t =
                model::sample(logits, chat.sampling, history, rng);
            if (t == tok_.eos_id()) {
                hit_eos = true;
                break;
            }
            out.push_back(t);
            history.push_back(t);
            if (n + 1 == chat.max_tokens) break;
            model_.forward_layers(t, {}, 0, hp.n_layers, *vision_cache_, seq,
                                  vision_ws_, logits);
        }
        const auto n_prompt = static_cast<std::uint32_t>(ids.size());
        const auto n_out = static_cast<std::uint32_t>(out.size() + hit_eos);
        completion_tokens_total_.fetch_add(n_out, std::memory_order_relaxed);
        const std::string reply = tok_.decode(out);
        const std::string rid =
            std::to_string(requests_total_.load(std::memory_order_relaxed));
        json body;
        if (anthropic) {
            body = {{"id", "msg_locus-vision-" + rid},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"model", opt_.model_name},
                    {"content", json::array({{{"type", "text"},
                                              {"text", reply}}})},
                    {"stop_reason", hit_eos ? "end_turn" : "max_tokens"},
                    {"stop_sequence", nullptr},
                    {"usage",
                     {{"input_tokens", n_prompt},
                      {"output_tokens", n_out},
                      {"cache_read_input_tokens", 0},
                      {"cache_creation_input_tokens", 0}}}};
        } else {
            body = {{"id", "locus-vision-" + rid},
                    {"object", "chat.completion"},
                    {"model", opt_.model_name},
                    {"choices",
                     json::array({{{"index", 0},
                                   {"message",
                                    {{"role", "assistant"},
                                     {"content", reply}}},
                                   {"finish_reason",
                                    hit_eos ? "stop" : "length"}}})},
                    {"usage",
                     {{"prompt_tokens", n_prompt},
                      {"completion_tokens", n_out},
                      {"total_tokens", n_prompt + n_out}}}};
        }
        res.set_content(body.dump(-1, ' ', false, kLossy), "application/json");
    } catch (const std::exception& e) {
        fail(http_status_for(e), e.what());
    }
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

OpenAiServer::~OpenAiServer() {
    if (http_) {
        http_->stop();
    }
}

OpenAiServer::Route OpenAiServer::route_of(const std::string& method,
                                           const std::string& path) const {
    const bool get_like = (method == "GET" || method == "HEAD");
    if (get_like) {
        // metrics_path is configurable, so match it first.
        if (path == opt_.metrics_path) {
            return Route::kMetrics;
        }
        if (path == "/health") {
            return Route::kHealth;
        }
        if (path == "/v1/models") {
            return Route::kModels;
        }
        static const std::string kModelsPrefix = "/v1/models/";
        if (path.rfind(kModelsPrefix, 0) == 0 &&
            path.size() > kModelsPrefix.size()) {
            return Route::kModelById;
        }
        return Route::kUnknown;
    }
    if (method == "POST") {
        if (path == "/v1/completions") {
            return Route::kCompletions;
        }
        if (path == "/v1/chat/completions") {
            return Route::kChat;
        }
        if (path == "/v1/messages") {
            return Route::kMessages;
        }
        if (path == "/v1/responses") {
            return Route::kResponses;
        }
        if (path == "/v1/embeddings") {
            return Route::kEmbeddings;
        }
        if (path == "/v1/audio/speech") {
            return Route::kAudioSpeech;
        }
        if (path == "/v1/images/generations") {
            return Route::kImages;
        }
        return Route::kUnknown;
    }
    return Route::kUnknown;
}

int OpenAiServer::build_and_start(const std::string& host, int port) {
    http::ServerConfig cfg;
    cfg.host = host;
    cfg.port = port;
    // Preserve the oversize-body 413-on-head cap httplib gave us via
    // set_payload_max_length: the parser rejects a too-large declared
    // body on the head, before the body is read.
    cfg.limits.max_body_bytes = opt_.max_body_bytes;
    // Serve the cheap monitoring routes on the acceptor's fast path so
    // they answer even when every worker is busy in a generation.
    // serve_connection still runs the head stage, so auth on /metrics
    // (when enabled) is unchanged -- only the serving thread differs.
    cfg.fast_path = [mp = opt_.metrics_path](std::string_view method,
                                             std::string_view target) {
        return method == "GET" && (target == "/health" || target == mp);
    };
    // Every route route_of serves; anything else counts as "other".
    http_metrics_ = std::make_shared<http::HttpMetrics>(
        std::vector<std::string>{
            opt_.metrics_path, "/health", "/v1/models", "/v1/models/",
            "/v1/completions", "/v1/chat/completions", "/v1/messages",
            "/v1/responses", "/v1/embeddings", "/v1/audio/speech",
            "/v1/images/generations"});
    cfg.metrics = http_metrics_;
    http_ = std::make_unique<http::Server>(std::move(cfg), route_handler_,
                                           auth_head_);
    return http_->start();
}

bool OpenAiServer::listen(const std::string& host, int port) {
    if (build_and_start(host, port) < 0) {
        return false;
    }
    http_->wait();  // serve until stop()
    return true;
}

int OpenAiServer::bind_any_port(const std::string& host) {
    return build_and_start(host, 0);
}

bool OpenAiServer::listen_after_bind() {
    if (!http_) {
        return false;
    }
    http_->wait();
    return true;
}

void OpenAiServer::stop() {
    if (http_) {
        http_->stop();
    }
}

void OpenAiServer::install_routes() {
    auto handle_health = [](const http::Request&,
                            http::ServerResponse& res) {
        res.set_content(json{{"status", "ok"}}.dump(-1, ' ', false, kLossy),
                        "application/json");
    };

    // OpenAI model listing: locus serves exactly one model. Auth (when
    // on) is enforced by the head stage before this runs.
    auto handle_models = [this](const http::Request&,
                                http::ServerResponse& res) {
            json entry{{"id", opt_.model_name},
                       {"object", "model"},
                       {"created", 0},
                       {"owned_by", "locus"}};
            res.set_content(
                json{{"object", "list"},
                     {"data", json::array({entry})}}
                    .dump(-1, ' ', false, kLossy),
                "application/json");
    };
    // Retrieve a single model by id (OpenAI GET /v1/models/{id}). The id
    // is the path suffix after "/v1/models/" (prefix match; the regex
    // route retired). Auth enforced by the head stage.
    auto handle_model_by_id = [this](const http::Request& req,
                                     http::ServerResponse& res) {
            const std::string model_id =
                req.path.substr(std::string("/v1/models/").size());
            if (model_id != opt_.model_name) {
                res.status = 404;
                res.set_content(
                    make_error("model not found").dump(-1, ' ', false, kLossy),
                    "application/json");
                return;
            }
            res.set_content(json{{"id", opt_.model_name},
                                 {"object", "model"},
                                 {"created", 0},
                                 {"owned_by", "locus"}}
                                .dump(-1, ' ', false, kLossy),
                            "application/json");
    };

    // Prometheus text-format metrics: request/token counters plus
    // engine gauges (KV pool, prefix reuse, speculative accepts). Auth
    // (when on) is enforced by the head stage before this runs.
    auto handle_metrics = [this](const http::Request&,
                                 http::ServerResponse& res) {
            const auto s = loop_.stats();
            // Every series carries the model and backend (i#285).
            const std::string backend(model_.active_backend().name);
            MetricsText mt({{"model", opt_.model_name},
                            {"backend", backend}});
            auto counter = [&](const char* name, const char* help,
                               std::uint64_t v) {
                mt.counter(name, help, v);
            };
            auto gauge = [&](const char* name, const char* help,
                             std::uint64_t v) {
                mt.gauge(name, help, v);
            };
            mt.info("locus_info", "What this server runs.",
                    {{"arch", opt_.model_arch},
                     {"chat_template",
                      std::string(opt_.chat_template.name())},
                     {"kv_type", kv_type_name(loop_.kv_type())},
                     {"cpu", cpu_name_},
                     {"gpu", devices_.empty() ? std::string()
                                              : devices_.front().second}});
            // Each device this server uses, with its role in the name's
            // kind (i#285); a CPU-only server lists just the CPU.
            mt.family("locus_device_info",
                      "The devices this server computes on.", "gauge");
            mt.value("locus_device_info",
                     MetricsText::label("kind", "cpu") + "," +
                         MetricsText::label("name", cpu_name_),
                     std::uint64_t{1});
            for (const auto& [kind, name] : devices_) {
                mt.value("locus_device_info",
                         MetricsText::label("kind", kind) + "," +
                             MetricsText::label("name", name),
                         std::uint64_t{1});
            }
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
            // Timings and scheduler state, read without the engine
            // lock. Phase times are per-request wall time, so with
            // concurrent requests their sums exceed the elapsed time;
            // engine_busy is the engine's own step time.
            const engine::EngineMetrics& em = loop_.metrics();
            constexpr auto kRelaxed = std::memory_order_relaxed;
            auto secs = [](std::uint64_t ns) {
                return static_cast<double>(ns) / 1e9;
            };
            mt.histogram("locus_queue_seconds",
                         "Submit to first admission, per request.",
                         em.queue.snapshot());
            mt.histogram("locus_prefill_seconds",
                         "First admission to first token, per request.",
                         em.prefill.snapshot());
            mt.histogram("locus_time_to_first_token_seconds",
                         "Submit to first token, per request.",
                         em.ttft.snapshot());
            mt.histogram("locus_decode_token_seconds",
                         "Time per generated token after the first, per "
                         "request.",
                         em.decode_token.snapshot());
            mt.histogram("locus_request_seconds",
                         "Submit to done, per completed request.",
                         em.request.snapshot());
            counter("locus_prefill_tokens_total",
                    "Prompt tokens computed (not reused) by completed "
                    "requests.",
                    em.prefill_tokens.load(kRelaxed));
            // Prefill wall time is locus_prefill_seconds_sum: a
            // locus_prefill_seconds_total counter would repeat it and
            // share the histogram's OpenMetrics family name.
            counter("locus_decode_tokens_total",
                    "Tokens after the first, of completed requests.",
                    em.decode_tokens.load(kRelaxed));
            mt.counter("locus_decode_seconds_total",
                       "Decode wall time of completed requests.",
                       secs(em.decode_ns.load(kRelaxed)));
            mt.counter("locus_engine_busy_seconds_total",
                       "Time the engine spent in scheduling steps.",
                       secs(em.busy_ns.load(kRelaxed)));
            counter("locus_engine_steps_total", "Engine scheduling steps.",
                    em.steps.load(kRelaxed));
            gauge("locus_sequences_running",
                  "Sequences running after the last step.",
                  em.running.load(kRelaxed));
            gauge("locus_requests_waiting",
                  "Requests waiting for admission after the last step.",
                  em.waiting.load(kRelaxed));
            // CPU cache counters for the inference compute threads
            // (Linux perf). Absent -- not zero -- when unavailable
            // (non-Linux, or perf_event_paranoid denies it), so a
            // scraper can tell "no counters" from "no misses". Read is a
            // few fds summed under a short lock, cheap enough for the
            // acceptor path this handler runs on (i#85).
            if (locus::sys::perf::available()) {
                const locus::sys::perf::Totals pc =
                    locus::sys::perf::read_totals();
                // Emit each only if its event is implemented on this PMU
                // (have_*): an unimplemented event is ABSENT, never a
                // misleading 0 (ARM often lacks the LLC event).
                if (pc.have_references) {
                    counter("locus_cpu_cache_references_total",
                            "CPU cache references, inference compute "
                            "threads.",
                            pc.cache_references);
                }
                if (pc.have_misses) {
                    counter("locus_cpu_cache_misses_total",
                            "CPU cache misses, inference compute threads.",
                            pc.cache_misses);
                }
                if (pc.have_llc) {
                    counter("locus_cpu_llc_load_misses_total",
                            "LLC load misses, inference compute threads "
                            "(DRAM-traffic signal).",
                            pc.llc_load_misses);
                }
            }
            if (http_metrics_) render_http_metrics(mt, *http_metrics_);
            if (const model::StageMetrics* sm = model_.stage_metrics()) {
                render_stage_metrics(mt, *sm);
            }
            render_gpu_metrics(mt);
            render_fallbacks(mt);
            res.set_content(mt.text(), "text/plain; version=0.0.4");
    };

    // Both endpoints share one implementation; `chat` only changes
    // how the prompt is built and how the response is shaped. Auth runs
    // in the head stage; ctx.identity carries the authenticated id here.
    auto handle = [this](const http::Request& req,
                         http::ServerResponse& res,
                         const http::RequestContext& ctx, bool chat) {
        auth_event("create", chat ? "chat" : "completion", ctx.identity);
        json body;
        std::string prompt;
        std::uint32_t max_tokens = 16;
        bool stream = false;
        std::vector<ToolSpec> tools;
        std::optional<VisionChat> vision;  // a chat with images (i#241)
        try {
            body = json::parse(req.body);
            if (chat) {
                if (!body.contains("messages") ||
                    !body["messages"].is_array() ||
                    body["messages"].empty()) {
                    throw std::invalid_argument(
                        "messages must be a non-empty array");
                }
                vision = parse_vision_chat(body, false, opt_.max_tokens_cap,
                                           opt_.max_body_bytes);
            }
            if (chat && !vision) {
                auto msgs = parse_messages(body["messages"]);
                tools = parse_tools(body, /*anthropic=*/false);
                if (!tools.empty()) {
                    msgs.insert(msgs.begin(),
                                {"system",
                                 render_tools_system(tools)});
                }
                prompt = opt_.chat_template.apply(msgs);
            } else if (!chat) {
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
            res.set_content(make_error(e.what()).dump(-1, ' ', false, kLossy),
                            "application/json");
            return;
        }
        if (vision) {
            vision_chat(false, *vision, res, ctx);
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
            res.set_content(make_error(why).dump(-1, ' ', false, kLossy),
                            "application/json");
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
            // Stop decoding if the client disconnects before we finish,
            // instead of running the abandoned request to max_tokens.
            auto v = loop_.wait_done_or_cancel(id, ctx.client_alive);
            // v is a full snapshot copy; drop the engine record now
            // (evict-on-consume) -- nothing below touches the id.
            loop_.release(id);
            if (v.status == engine::Status::kCancelled) {
                // Client disconnected before the response. Emit a
                // diagnosable 499 rather than the default 200/empty: a
                // client that merely half-closed its write side
                // (indistinguishable from abandonment over TCP -- see
                // http::RequestContext::client_alive) would otherwise
                // read a bare 200 as a valid empty completion. The write
                // is discarded when the client really is gone.
                res.status = 499;
                res.set_content(
                    make_error("request cancelled: client closed the "
                               "connection")
                        .dump(-1, ' ', false, kLossy),
                    "application/json");
                return;
            }
            if (v.status == engine::Status::kFailed) {
                res.status = 500;
                res.set_content(
                    make_error(v.error).dump(-1, ' ', false, kLossy),
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
                            {"arguments",
                             tc->arguments.dump(-1, ' ', false, kLossy)}}}};
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
                       {"total_tokens", n_prompt + n_completion},
                       // OpenAI shape: cached_tokens is a SUBSET of
                       // prompt_tokens (prompt-cache KV reuse).
                       {"prompt_tokens_details",
                        {{"cached_tokens",
                          v.reused_prefix_tokens}}}}}};
            res.set_content(out.dump(-1, ' ', false, kLossy),
                            "application/json");
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
             buffer_tools](http::Sink& sink) {
              // The shim calls a provider ONCE; loop until terminal,
              // one chunk per progress step (httplib re-invoked instead).
              for (;;) {
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
                        payload += "data: " +
                                   chunk.dump(-1, ' ', false, kLossy) +
                                   "\n\n";
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
                                   tc->arguments.dump(-1, ' ', false,
                                                      kLossy)}}}};
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
                        payload += "data: " +
                                   chunk.dump(-1, ' ', false, kLossy) +
                                   "\n\n";
                    }
                    payload += "data: [DONE]\n\n";
                    // Stream fully built from the snapshot v; drop the
                    // engine record (the provider won't run again).
                    loop_.release(id);
                }
                if (!payload.empty() &&
                    !sink.write(std::string_view(payload.data(),
                                                 payload.size()))) {
                    // Client vanished mid-stream: stop the generation
                    // and drop the record (both no-ops if the terminal
                    // path above already released it).
                    loop_.cancel(id);
                    loop_.release(id);
                    return;
                }
                if (terminal) {
                    sink.done();
                    return;
                }
              }
            });
    };

    // Anthropic Messages API (POST /v1/messages): same engine, its
    // own request/response shape and SSE event sequence, so Claude
    // Code and the Anthropic SDK can target locus directly.
    auto handle_messages = [this](const http::Request& req,
                                  http::ServerResponse& res,
                                  const http::RequestContext& ctx) {
        auth_event("create", "messages", ctx.identity);
        std::string prompt;
        std::uint32_t max_tokens = 16;
        std::uint32_t input_tokens = 0;
        bool stream = false;
        std::vector<tok::TokenId> ids;
        std::vector<ToolSpec> tools;
        model::SamplingParams sampling;
        std::uint64_t seed = 0;
        std::unique_ptr<model::TokenConstraint> constraint;
        std::optional<VisionChat> vision;  // a chat with images (i#241)
        try {
            const json body = json::parse(req.body);
            if (!body.contains("messages") ||
                !body["messages"].is_array() ||
                body["messages"].empty()) {
                throw std::invalid_argument(
                    "messages must be a non-empty array");
            }
            vision = parse_vision_chat(body, true, opt_.max_tokens_cap,
                                       opt_.max_body_bytes);
            if (!vision) {
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
            }
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(
                json{{"type", "error"},
                     {"error",
                      {{"type", "invalid_request_error"},
                       {"message", e.what()}}}}
                    .dump(-1, ' ', false, kLossy),
                "application/json");
            return;
        }
        if (vision) {
            vision_chat(true, *vision, res, ctx);
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
            // Stop decoding if the client disconnects before we finish.
            auto v = loop_.wait_done_or_cancel(id, ctx.client_alive);
            loop_.release(id);  // consumed: drop the engine record
            if (v.status == engine::Status::kCancelled) {
                // Client disconnected before the response (see the
                // completions path and client_alive): 499 + the
                // Anthropic error envelope, never a default 200/empty.
                res.status = 499;
                res.set_content(
                    json{{"type", "error"},
                         {"error",
                          {{"type", "api_error"},
                           {"message", "request cancelled: client "
                                       "closed the connection"}}}}
                        .dump(-1, ' ', false, kLossy),
                    "application/json");
                return;
            }
            if (v.status == engine::Status::kFailed) {
                res.status = 500;
                res.set_content(
                    json{{"type", "error"},
                         {"error",
                          {{"type", "api_error"},
                           {"message", v.error}}}}
                        .dump(-1, ' ', false, kLossy),
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
                 // Anthropic shape: input_tokens is the UNCACHED
                 // remainder; cache_read + cache_creation are separate
                 // and the three sum to the full prompt length.
                 {{"input_tokens", input_tokens -
                                       v.reused_prefix_tokens -
                                       v.cached_prefix_tokens},
                  {"output_tokens",
                   static_cast<std::uint32_t>(v.generated.size())},
                  {"cache_read_input_tokens", v.reused_prefix_tokens},
                  {"cache_creation_input_tokens",
                   v.cached_prefix_tokens}}}};
            completion_tokens_total_.fetch_add(
                v.generated.size(), std::memory_order_relaxed);
            res.set_content(out.dump(-1, ' ', false, kLossy),
                            "application/json");
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
                http::Sink& sink) {
              // Call-once provider: loop until terminal (see completions).
              for (;;) {
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
                               start.dump(-1, ' ', false, kLossy) + "\n\n";
                    if (!buffer_tools) {
                        payload +=
                            "event: content_block_start\ndata: " +
                            json{{"type", "content_block_start"},
                                 {"index", 0},
                                 {"content_block",
                                  {{"type", "text"},
                                   {"text", ""}}}}
                                .dump(-1, ' ', false, kLossy) +
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
                                .dump(-1, ' ', false, kLossy) +
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
                                    .dump(-1, ' ', false, kLossy) +
                                "\n\n";
                            payload +=
                                "event: content_block_delta\n"
                                "data: " +
                                json{{"type", "content_block_delta"},
                                     {"index", 0},
                                     {"delta",
                                      {{"type", "input_json_delta"},
                                       {"partial_json",
                                        tc->arguments.dump(-1, ' ', false,
                                                           kLossy)}}}}
                                    .dump(-1, ' ', false, kLossy) +
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
                                    .dump(-1, ' ', false, kLossy) +
                                "\n\n";
                            payload +=
                                "event: content_block_delta\n"
                                "data: " +
                                json{{"type", "content_block_delta"},
                                     {"index", 0},
                                     {"delta",
                                      {{"type", "text_delta"},
                                       {"text", text}}}}
                                    .dump(-1, ' ', false, kLossy) +
                                "\n\n";
                        }
                    }
                    payload +=
                        "event: content_block_stop\ndata: " +
                        json{{"type", "content_block_stop"},
                             {"index", 0}}
                            .dump(-1, ' ', false, kLossy) +
                        "\n\n";
                    payload +=
                        "event: message_delta\ndata: " +
                        json{{"type", "message_delta"},
                             {"delta",
                              {{"stop_reason", stop_reason},
                               {"stop_sequence", nullptr}}},
                             // Prompt-cache fields land here at
                             // stream end (cache_creation is only
                             // known once the request finishes).
                             {"usage",
                              {{"output_tokens",
                                static_cast<std::uint32_t>(
                                    st->emitted)},
                               {"cache_read_input_tokens",
                                v.reused_prefix_tokens},
                               {"cache_creation_input_tokens",
                                v.cached_prefix_tokens}}}}
                            .dump(-1, ' ', false, kLossy) +
                        "\n\n";
                    payload += "event: message_stop\ndata: " +
                               json{{"type", "message_stop"}}
                                   .dump(-1, ' ', false, kLossy) +
                               "\n\n";
                    completion_tokens_total_.fetch_add(
                        st->emitted, std::memory_order_relaxed);
                    // Stream fully built from v; drop the record.
                    loop_.release(id);
                }
                if (!payload.empty() &&
                    !sink.write(std::string_view(payload.data(),
                                                 payload.size()))) {
                    // Client vanished mid-stream: stop the generation
                    // and drop the record (both no-ops if the terminal
                    // path above already released it).
                    loop_.cancel(id);
                    loop_.release(id);
                    return;
                }
                if (terminal) {
                    sink.done();
                    return;
                }
              }
            });
    };

    // OpenAI embeddings: pooled hidden state, L2-normalized. `input`
    // is a string or an array of strings. Auth runs in the head stage.
    auto handle_embeddings = [this](const http::Request& req,
                                    http::ServerResponse& res,
                                    const http::RequestContext& ctx) {
            auth_event("create", "embeddings", ctx.identity);
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
                        .dump(-1, ' ', false, kLossy),
                    "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(
                    make_error(e.what()).dump(-1, ' ', false, kLossy),
                    "application/json");
            }
    };

    // POST /v1/audio/speech (OpenAI shape): text -> Orpheus tokens ->
    // SNAC waveform. 503 unless --audio-decoder loaded a decoder AND the
    // model is an Orpheus-TTS checkpoint (audio_band_base_ >= 0). The
    // decode is serialized on a dedicated cache, like embeddings.
    auto handle_audio = [this](const http::Request& req,
                               http::ServerResponse& res,
                               const http::RequestContext& ctx) {
            auth_event("create", "audio.speech", ctx.identity);
            if (!snac_ || audio_band_base_ < 0) {
                set_unavailable(res,
                                "audio speech is not configured (needs "
                                "--audio-decoder and an Orpheus-TTS model)");
                return;
            }
            try {
                const json body = json::parse(req.body);
                if (!body.contains("input") ||
                    !body["input"].is_string()) {
                    throw std::invalid_argument("input must be a string");
                }
                const auto input = body["input"].get<std::string>();
                std::string voice = "tara";
                if (body.contains("voice") && body["voice"].is_string()) {
                    voice = body["voice"].get<std::string>();
                }
                std::string fmt = "wav";
                if (body.contains("response_format") &&
                    body["response_format"].is_string()) {
                    fmt = body["response_format"].get<std::string>();
                }
                audio::WaveFormat wf;
                std::string ctype;
                if (fmt == "wav") {
                    wf = audio::WaveFormat::kWav;
                    ctype = "audio/wav";
                } else if (fmt == "pcm") {
                    wf = audio::WaveFormat::kPcm16;
                    ctype = "audio/L16";
                } else {
                    throw std::invalid_argument(
                        "response_format must be wav or pcm");
                }

                // Orpheus reference prompt wrapping: a start marker, the
                // "voice: text" tokenization (with BOS), then the four
                // end markers.
                constexpr tok::TokenId kStart = 128259;
                const tok::TokenId kEnd[4] = {128009, 128260, 128261,
                                              128257};
                auto btoks = tok_.encode(voice + ": " + input, true);
                std::vector<tok::TokenId> ids;
                ids.reserve(btoks.size() + 5);
                ids.push_back(kStart);
                ids.insert(ids.end(), btoks.begin(), btoks.end());
                for (auto e : kEnd) ids.push_back(e);

                // Sampling. Defaults are the Orpheus reference settings
                // (temperature 0.6, top_p 0.8, repetition penalty 1.3);
                // greedy decoding never ends the utterance and decodes
                // to near-silence on this model. temperature 0 is
                // deterministic but is argmax AFTER the repetition
                // penalty (as in vLLM); true greedy also needs
                // repetition_penalty 1.
                auto num = [&](const char* k, double def) {
                    if (!body.contains(k)) return def;
                    if (!body[k].is_number()) {
                        throw std::invalid_argument(
                            std::string(k) + " must be a number");
                    }
                    return body[k].get<double>();
                };
                model::SamplingParams sp;
                sp.temperature = static_cast<float>(num("temperature", 0.6));
                sp.top_p = static_cast<float>(num("top_p", 0.8));
                sp.repeat_penalty =
                    static_cast<float>(num("repetition_penalty", 1.3));
                // Penalize over the whole history (prompt + generated),
                // as HF/vLLM repetition_penalty does, not a 64 window.
                sp.repeat_last_n = UINT32_MAX;
                if (sp.temperature < 0.0f || sp.top_p <= 0.0f ||
                    sp.top_p > 1.0f || sp.repeat_penalty <= 0.0f) {
                    throw std::invalid_argument(
                        "need temperature >= 0, 0 < top_p <= 1, "
                        "repetition_penalty > 0");
                }
                std::uint32_t cap = opt_.audio_max_tokens;
                if (body.contains("max_tokens")) {
                    if (!body["max_tokens"].is_number_unsigned() ||
                        body["max_tokens"].get<std::uint64_t>() == 0) {
                        throw std::invalid_argument(
                            "max_tokens must be a positive integer");
                    }
                    cap = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                        cap, body["max_tokens"].get<std::uint64_t>()));
                }
                std::uint64_t seed = std::random_device{}();
                if (body.contains("seed")) {
                    if (!body["seed"].is_number_integer()) {
                        throw std::invalid_argument(
                            "seed must be an integer");
                    }
                    seed = static_cast<std::uint64_t>(
                        body["seed"].get<std::int64_t>());
                }
                bool stream = false;
                if (body.contains("stream")) {
                    if (!body["stream"].is_boolean()) {
                        throw std::invalid_argument(
                            "stream must be a boolean");
                    }
                    stream = body["stream"].get<bool>();
                }

                // Prefill + sampling on the dedicated cache, under
                // audio_mu_. on_frame runs after each token that
                // completes a frame and returns false to stop early;
                // alive (may be empty) is polled before each token.
                // Returns the audio tokens generated and the prefill /
                // generate end times.
                using Clock = std::chrono::steady_clock;
                struct Run {
                    std::uint32_t generated = 0;
                    Clock::time_point t0, t1, t2;
                };
                auto run = [this, ids, sp, cap](
                               std::mt19937_64& rng,
                               model::OrpheusCodecSampler& sampler,
                               const std::function<bool()>& on_frame,
                               const std::function<bool()>& alive) {
                    Run r;
                    std::vector<tok::TokenId> history = ids;
                    std::lock_guard<std::mutex> lk(audio_mu_);
                    if (!audio_cache_) {
                        audio_cache_ = std::make_unique<kv::PagedKvCache>(
                            model_.make_cache());
                        audio_ws_ = model_.make_workspace();
                    }
                    std::vector<float> logits(n_vocab_);
                    kv::PagedKvCache::Seq seq;
                    bool ok = true;
                    // Prompt in chunks through the batched forward where
                    // the backend has one (one token at a time otherwise),
                    // the way the engine prefills.
                    r.t0 = Clock::now();
                    const std::size_t chunk = model_.prefill_chunk();
                    for (std::size_t at = 0; ok && at < ids.size();) {
                        const std::size_t nb =
                            std::min(chunk, ids.size() - at);
                        if (!audio_cache_->ensure_capacity(
                                seq, static_cast<std::uint32_t>(nb))) {
                            ok = false;
                            break;
                        }
                        if (nb >= 2 && model_.supports_batch_prefill()) {
                            model_.forward_batch(
                                std::span<const tok::TokenId>(
                                    ids.data() + at, nb),
                                *audio_cache_, seq, audio_ws_, logits);
                        } else {
                            for (std::size_t k = 0; k < nb; ++k) {
                                model_.forward(ids[at + k], *audio_cache_,
                                               seq, audio_ws_, logits,
                                               audio_rows_);
                            }
                        }
                        at += nb;
                    }
                    // The batched prefill computes the whole head; mask
                    // it to the same rows the decode steps compute, so
                    // the first draw sees the same candidate set.
                    if (audio_rows_.end != 0) {
                        const float ninf =
                            -std::numeric_limits<float>::infinity();
                        std::fill(logits.begin(),
                                  logits.begin() + audio_rows_.begin, ninf);
                        std::fill(logits.begin() + audio_rows_.end,
                                  logits.end(), ninf);
                    }
                    r.t1 = Clock::now();
                    for (std::uint32_t i = 0; ok && i < cap; ++i) {
                        if (alive && !alive()) break;
                        const auto next =
                            model::sample(logits, sp, history, rng);
                        const std::size_t before = sampler.frames();
                        if (sampler.push(next) ==
                            model::OrpheusCodecSampler::Status::kStop) {
                            break;
                        }
                        history.push_back(next);
                        ++r.generated;
                        if (sampler.frames() > before && !on_frame()) break;
                        if (!audio_cache_->ensure_capacity(seq, 1)) break;
                        model_.forward(next, *audio_cache_, seq, audio_ws_,
                                       logits, audio_rows_);
                    }
                    audio_cache_->release(seq);
                    r.t2 = Clock::now();
                    return r;
                };

                // The seed actually used (drawn by the server when the
                // request sent none), so any utterance can be replayed.
                res.set_header("X-Audio-Seed", std::to_string(seed));

                if (stream) {
                    // Chunked body: each frame range goes out as soon as
                    // the context_frames() frames after it exist (that is
                    // all decode_range reads), so playback starts after
                    // the first few frames. The bytes equal the buffered
                    // response's, except that a WAV header carries
                    // 0xFFFFFFFF sizes. The token count and stop token
                    // are not known when the head is sent, so they are
                    // not reported.
                    res.set_chunked_content_provider(
                        ctype, [this, run, seed, wf](http::Sink& sink) {
                            const std::uint32_t rate = snac_->sample_rate();
                            const std::uint32_t lookahead =
                                snac_->context_frames();
                            std::mt19937_64 rng(seed);
                            model::OrpheusCodecSampler sampler(
                                audio_band_base_);
                            std::uint32_t emitted = 0;
                            bool alive = true;
                            auto send = [&](const std::vector<std::byte>& b) {
                                alive = alive &&
                                        sink.write(std::string_view(
                                            reinterpret_cast<const char*>(
                                                b.data()),
                                            b.size()));
                                return alive;
                            };
                            // Decodes and sends frames [emitted, upto).
                            auto emit = [&](const model::SnacCodes& codes,
                                            std::uint32_t upto) {
                                if (upto <= emitted) return alive;
                                const auto pcm = snac_->decode_range(
                                    codes, emitted, upto);
                                emitted = upto;
                                return send(audio::encode_pcm(
                                    pcm, rate, audio::WaveFormat::kPcm16));
                            };
                            try {
                                if (wf == audio::WaveFormat::kWav &&
                                    !send(audio::wav_stream_header(rate))) {
                                    return;
                                }
                                run(rng, sampler,
                                    [&] {
                                        const auto codes =
                                            sampler.codes_lenient();
                                        const auto n =
                                            static_cast<std::uint32_t>(
                                                codes.coarse.size());
                                        const std::uint32_t ready =
                                            n > lookahead ? n - lookahead
                                                          : 0;
                                        // One frame first (latency),
                                        // then 8-frame runs (the window
                                        // overhead per run is fixed).
                                        const std::uint32_t min_run =
                                            emitted == 0 ? 1 : 8;
                                        if (ready >= emitted + min_run) {
                                            return emit(codes, ready);
                                        }
                                        return alive;
                                    },
                                    {});
                                if (alive) {
                                    const auto codes =
                                        sampler.codes_lenient();
                                    emit(codes, static_cast<std::uint32_t>(
                                                    codes.coarse.size()));
                                }
                            } catch (const std::exception&) {
                                // The head is out; end the stream short.
                            }
                        });
                    return;
                }

                model::OrpheusCodecSampler sampler(audio_band_base_);
                std::mt19937_64 rng(seed);
                const Run r = run(rng, sampler, [] { return true; },
                                  ctx.client_alive);
                const auto t3a = Clock::now();
                const auto wav = snac_->decode(sampler.codes_lenient());
                const auto t3 = Clock::now();
                const std::uint32_t generated = r.generated;
                const auto ms = [](auto a, auto b) {
                    return std::to_string(
                        std::chrono::duration<double, std::milli>(b - a)
                            .count());
                };
                // Where the time went: prompt prefill, token generation
                // and the SNAC decode (i#173).
                const std::string timing = "prefill;dur=" + ms(r.t0, r.t1) +
                                           ", generate;dur=" +
                                           ms(r.t1, r.t2) +
                                           ", decode;dur=" + ms(t3a, t3);
                auto bytes =
                    audio::encode_pcm(wav, snac_->sample_rate(), wf);
                // How generation ended: the stop token id when the model
                // ended speech itself, -1 when it ran to the cap.
                res.set_header("X-Audio-Tokens", std::to_string(generated));
                res.set_header("X-Audio-Stop-Token",
                               std::to_string(sampler.stop_token()));
                res.set_header("Server-Timing", timing);
                res.set_content(
                    std::string(
                        reinterpret_cast<const char*>(bytes.data()),
                        bytes.size()),
                    ctype);
            } catch (const std::exception& e) {
                set_error(res, e);
            }
    };

    // POST /v1/images/generations with the PixArt pipeline
    // (--image-model): prompt -> T5 -> DiT and DPM-Solver++ with CFG ->
    // VAE -> PNG, as b64_json. Extensions to the OpenAI shape:
    // negative_prompt, steps (default 20), guidance_scale (default the
    // GGUF's, 4.5) and seed. One image at a time, under image_mu_.
    auto handle_pixart = [this](const http::Request& req,
                                http::ServerResponse& res,
                                const http::RequestContext& ctx) {
            try {
                const json body = json::parse(req.body);
                auto str = [&](const char* k) -> std::string {
                    if (!body.contains(k)) return {};
                    if (!body[k].is_string()) {
                        throw std::invalid_argument(std::string(k) +
                                                    " must be a string");
                    }
                    return body[k].get<std::string>();
                };
                image::PixArtPipeline::Request p;
                p.prompt = str("prompt");
                if (p.prompt.empty()) {
                    throw std::invalid_argument(
                        "prompt must be a non-empty string");
                }
                p.negative_prompt = str("negative_prompt");
                if (body.contains("n") &&
                    (!body["n"].is_number_integer() ||
                     body["n"].get<std::int64_t>() != 1)) {
                    throw std::invalid_argument("n must be 1");
                }
                const std::uint32_t side = pixart_->image_size();
                const std::string size =
                    std::to_string(side) + "x" + std::to_string(side);
                if (body.contains("size") && str("size") != size) {
                    throw std::invalid_argument("size must be " + size);
                }
                if (body.contains("response_format") &&
                    str("response_format") != "b64_json") {
                    throw std::invalid_argument(
                        "response_format must be b64_json");
                }
                if (body.contains("steps")) {
                    if (!body["steps"].is_number_integer() ||
                        body["steps"].get<std::int64_t>() < 1 ||
                        body["steps"].get<std::int64_t>() > 100) {
                        throw std::invalid_argument(
                            "steps must be an integer from 1 to 100");
                    }
                    p.steps = static_cast<std::uint32_t>(
                        body["steps"].get<std::int64_t>());
                }
                p.guidance = pixart_->default_guidance();
                if (body.contains("guidance_scale")) {
                    if (!body["guidance_scale"].is_number()) {
                        throw std::invalid_argument(
                            "guidance_scale must be a number");
                    }
                    // Checked as a double: a JSON value past float's range
                    // would make the cast undefined.
                    const double g = body["guidance_scale"].get<double>();
                    if (!std::isfinite(g) || g < 0.0 || g > 100.0) {
                        throw std::invalid_argument(
                            "guidance_scale must be from 0 to 100");
                    }
                    p.guidance = static_cast<float>(g);
                }
                p.seed = std::random_device{}();
                if (body.contains("seed")) {
                    if (!body["seed"].is_number_integer()) {
                        throw std::invalid_argument(
                            "seed must be an integer");
                    }
                    p.seed = static_cast<std::uint64_t>(
                        body["seed"].get<std::int64_t>());
                }
                // The image models' linears run on the GPU when the
                // Vulkan device imports host memory (unified memory:
                // their mmapped weights cost it nothing), else on the
                // CPU backend, whatever --backend picked for the LLM.
                // A discrete GPU would copy 12 GB of weights into the
                // pool, evicting the LLM's. With these ops the DiT's
                // attention and the VAE's convolutions run on the GPU
                // too; T5's attention stays on the CPU.
                const auto& op = backend::vulkan_imports_host_memory()
                                     ? backend::vulkan_image_ops()
                                     : backend::best_backend().ops;
                image::PixArtPipeline::Timing took;
                std::vector<std::uint8_t> rgb;
                {
                    // An image takes minutes: a second request gets a
                    // 503 instead of holding a server thread in a queue.
                    std::unique_lock<std::mutex> lk(image_mu_,
                                                    std::try_to_lock);
                    if (!lk.owns_lock()) {
                        set_unavailable(res,
                                        "an image is already being "
                                        "generated; retry later",
                                        60);
                        return;
                    }
                    rgb = pixart_->generate(p, op, ctx.client_alive, &took);
                }
                if (rgb.empty()) {
                    res.status = 499;  // the client went away
                    return;
                }
                const auto png = image::encode_png(rgb, side, side);
                res.set_header("X-Image-Seed", std::to_string(p.seed));
                res.set_header(
                    "Server-Timing",
                    "encode;dur=" + std::to_string(took.encode) +
                        ", denoise;dur=" + std::to_string(took.denoise) +
                        ", decode;dur=" + std::to_string(took.decode));
                const auto created =
                    std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
                res.set_content(
                    json{{"created", created},
                         {"data", json::array({json{
                                      {"b64_json",
                                       http::base64_encode(png)}}})}}
                        .dump(-1, ' ', false, kLossy),
                    "application/json");
            } catch (const std::exception& e) {
                set_error(res, e);
            }
    };

    // POST /v1/images/generations with the FLUX.1 pipeline
    // (--image-model with a flux GGUF): prompt -> CLIP-L pooled and T5
    // states -> the flow-matching loop -> VAE -> PNG, as b64_json.
    // Extensions to the OpenAI shape: steps (default 28, or 4 for
    // schnell), guidance_scale (default 3.5; schnell ignores it),
    // max_sequence_length (T5 tokens, default 512) and seed. size is
    // "WxH", multiples of 16 up to 1024 (default 1024x1024). One image
    // at a time, under image_mu_.
    auto handle_flux = [this](const http::Request& req,
                              http::ServerResponse& res,
                              const http::RequestContext& ctx) {
            try {
                const json body = json::parse(req.body);
                auto str = [&](const char* k) -> std::string {
                    if (!body.contains(k)) return {};
                    if (!body[k].is_string()) {
                        throw std::invalid_argument(std::string(k) +
                                                    " must be a string");
                    }
                    return body[k].get<std::string>();
                };
                auto integer = [&](const char* k, std::int64_t lo,
                                   std::int64_t hi) {
                    if (!body[k].is_number_integer() ||
                        body[k].get<std::int64_t>() < lo ||
                        body[k].get<std::int64_t>() > hi) {
                        throw std::invalid_argument(
                            std::string(k) + " must be an integer from " +
                            std::to_string(lo) + " to " + std::to_string(hi));
                    }
                    return body[k].get<std::int64_t>();
                };
                image::FluxPipeline::Request p;
                p.prompt = str("prompt");
                if (p.prompt.empty()) {
                    throw std::invalid_argument(
                        "prompt must be a non-empty string");
                }
                if (body.contains("n") && integer("n", 1, 1) != 1) {
                    throw std::invalid_argument("n must be 1");
                }
                if (body.contains("size")) {
                    // "WxH": two decimal sides, no signs or spaces.
                    const std::string s = str("size");
                    const auto x = s.find('x');
                    const auto digits = [](std::string_view d) {
                        return !d.empty() && d.size() <= 4 &&
                               std::all_of(d.begin(), d.end(), [](char c) {
                                   return c >= '0' && c <= '9';
                               });
                    };
                    if (x == std::string::npos ||
                        !digits(std::string_view(s).substr(0, x)) ||
                        !digits(std::string_view(s).substr(x + 1))) {
                        throw std::invalid_argument(
                            "size must be WIDTHxHEIGHT");
                    }
                    p.width = static_cast<std::uint32_t>(
                        std::stoul(s.substr(0, x)));
                    p.height = static_cast<std::uint32_t>(
                        std::stoul(s.substr(x + 1)));
                }
                if (body.contains("response_format") &&
                    str("response_format") != "b64_json") {
                    throw std::invalid_argument(
                        "response_format must be b64_json");
                }
                if (body.contains("steps")) {
                    p.steps =
                        static_cast<std::uint32_t>(integer("steps", 1, 100));
                }
                if (body.contains("max_sequence_length")) {
                    p.max_sequence_length = static_cast<std::uint32_t>(
                        integer("max_sequence_length", 1, 512));
                }
                if (body.contains("guidance_scale")) {
                    if (!body["guidance_scale"].is_number()) {
                        throw std::invalid_argument(
                            "guidance_scale must be a number");
                    }
                    const double g = body["guidance_scale"].get<double>();
                    if (!std::isfinite(g) || g < 0.0 || g > 100.0) {
                        throw std::invalid_argument(
                            "guidance_scale must be from 0 to 100");
                    }
                    p.guidance = static_cast<float>(g);
                }
                p.seed = std::random_device{}();
                if (body.contains("seed")) {
                    if (!body["seed"].is_number_integer()) {
                        throw std::invalid_argument(
                            "seed must be an integer");
                    }
                    p.seed = static_cast<std::uint64_t>(
                        body["seed"].get<std::int64_t>());
                }
                // As for PixArt: the GPU when the Vulkan device imports
                // host memory, else the CPU backend.
                const auto& op = backend::vulkan_imports_host_memory()
                                     ? backend::vulkan_image_ops()
                                     : backend::best_backend().ops;
                image::FluxPipeline::Timing took;
                std::vector<std::uint8_t> rgb;
                {
                    std::unique_lock<std::mutex> lk(image_mu_,
                                                    std::try_to_lock);
                    if (!lk.owns_lock()) {
                        set_unavailable(res,
                                        "an image is already being "
                                        "generated; retry later",
                                        60);
                        return;
                    }
                    rgb = flux_->generate(p, op, ctx.client_alive, &took);
                }
                if (rgb.empty()) {
                    res.status = 499;  // the client went away
                    return;
                }
                const auto png = image::encode_png(rgb, p.width, p.height);
                res.set_header("X-Image-Seed", std::to_string(p.seed));
                res.set_header(
                    "Server-Timing",
                    "encode;dur=" + std::to_string(took.encode) +
                        ", denoise;dur=" + std::to_string(took.denoise) +
                        ", decode;dur=" + std::to_string(took.decode));
                const auto created =
                    std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
                res.set_content(
                    json{{"created", created},
                         {"data", json::array({json{
                                      {"b64_json",
                                       http::base64_encode(png)}}})}}
                        .dump(-1, ' ', false, kLossy),
                    "application/json");
            } catch (const std::exception& e) {
                set_error(res, e);
            }
    };

    // POST /v1/images/generations (OpenAI shape): prompt -> Janus-Pro
    // image codes (CFG) -> VQ-16 pixels -> PNG, returned as b64_json.
    // 503 unless --image-decoder loaded a decoder AND the model is a
    // Janus-Pro checkpoint, or --image-model loaded PixArt (then a
    // request whose model starts with "pixart", or every request when
    // Janus is not loaded, goes to handle_pixart). One image at a time,
    // under image_mu_.
    auto handle_images = [this, handle_pixart, handle_flux](
                             const http::Request& req,
                             http::ServerResponse& res,
                             const http::RequestContext& ctx) {
            auth_event("create", "images.generations", ctx.identity);
            if (flux_) {
                bool named = false;
                try {
                    const json b = json::parse(req.body);
                    named = b.contains("model") && b["model"].is_string() &&
                            b["model"].get<std::string>().rfind("flux", 0) ==
                                0;
                } catch (const std::exception&) {
                    // handle_flux reports the parse error
                }
                if (named || !vq16_ || !janus_) {
                    handle_flux(req, res, ctx);
                    return;
                }
            }
            if (pixart_) {
                bool named = false;
                try {
                    const json b = json::parse(req.body);
                    named = b.contains("model") && b["model"].is_string() &&
                            b["model"].get<std::string>().rfind("pixart",
                                                                0) == 0;
                } catch (const std::exception&) {
                    // handle_pixart reports the parse error
                }
                if (named || !vq16_ || !janus_) {
                    handle_pixart(req, res, ctx);
                    return;
                }
            }
            if (!vq16_ || !janus_) {
                set_unavailable(res,
                                "image generation is not configured (needs "
                                "--image-decoder and a Janus-Pro model, or "
                                "--image-model)");
                return;
            }
            try {
                const json body = json::parse(req.body);
                if (!body.contains("prompt") ||
                    !body["prompt"].is_string() ||
                    body["prompt"].get<std::string>().empty()) {
                    throw std::invalid_argument(
                        "prompt must be a non-empty string");
                }
                const auto prompt = body["prompt"].get<std::string>();
                if (body.contains("n") &&
                    (!body["n"].is_number_integer() ||
                     body["n"].get<std::int64_t>() != 1)) {
                    throw std::invalid_argument("n must be 1");
                }
                // The model makes one fixed size: grid codes per side,
                // upscale() pixels per code (24 * 16 = 384).
                const std::uint32_t side = janus_->grid() * vq16_->upscale();
                const std::string size =
                    std::to_string(side) + "x" + std::to_string(side);
                if (body.contains("size") &&
                    (!body["size"].is_string() ||
                     body["size"].get<std::string>() != size)) {
                    throw std::invalid_argument("size must be " + size);
                }
                if (body.contains("response_format") &&
                    (!body["response_format"].is_string() ||
                     body["response_format"].get<std::string>() !=
                         "b64_json")) {
                    throw std::invalid_argument(
                        "response_format must be b64_json");
                }
                // Guidance and sampling; the defaults are the
                // reference's (CFG 5, temperature 1). temperature 0
                // picks the argmax code at each step.
                image::JanusImageGen::Params p;
                p.cfg = janus_->cfg_scale();
                auto num = [&](const char* k, float& dst) {
                    if (!body.contains(k)) return;
                    if (!body[k].is_number()) {
                        throw std::invalid_argument(
                            std::string(k) + " must be a number");
                    }
                    dst = body[k].get<float>();
                };
                num("cfg", p.cfg);
                num("temperature", p.temperature);
                if (!std::isfinite(p.cfg) || p.cfg < 0.0f ||
                    !std::isfinite(p.temperature) || p.temperature < 0.0f) {
                    throw std::invalid_argument(
                        "need finite cfg >= 0 and temperature >= 0");
                }
                p.seed = std::random_device{}();
                if (body.contains("seed")) {
                    if (!body["seed"].is_number_integer()) {
                        throw std::invalid_argument(
                            "seed must be an integer");
                    }
                    p.seed = static_cast<std::uint64_t>(
                        body["seed"].get<std::int64_t>());
                }

                using Clock = std::chrono::steady_clock;
                const auto ids = janus_->prompt_ids(tok_, prompt);
                std::vector<std::int32_t> codes;
                std::vector<float> chw;
                Clock::time_point t0, t1, t2;
                {
                    // As the PixArt path: a request that finds an image
                    // in progress gets a 503, not a wait on a server
                    // thread.
                    std::unique_lock<std::mutex> lk(image_mu_,
                                                    std::try_to_lock);
                    if (!lk.owns_lock()) {
                        set_unavailable(res,
                                        "an image is already being "
                                        "generated; retry later",
                                        30);
                        return;
                    }
                    t0 = Clock::now();
                    codes = janus_->generate(model_, ids, p,
                                             ctx.client_alive);
                    t1 = Clock::now();
                    if (codes.size() == janus_->image_tokens()) {
                        chw = vq16_->decode(codes, janus_->grid(),
                                            janus_->grid());
                    }
                    t2 = Clock::now();
                }
                if (chw.empty()) {
                    // The client went away mid-generation; nobody reads
                    // this.
                    res.status = 499;
                    return;
                }
                const auto png =
                    image::encode_png(image::to_rgb8(chw, side, side), side,
                                      side);
                const auto ms = [](auto a, auto b) {
                    return std::to_string(
                        std::chrono::duration<double, std::milli>(b - a)
                            .count());
                };
                // The seed actually used, so any image can be replayed;
                // and where the time went (codes, then pixels).
                res.set_header("X-Image-Seed", std::to_string(p.seed));
                res.set_header("Server-Timing", "generate;dur=" +
                                                    ms(t0, t1) +
                                                    ", decode;dur=" +
                                                    ms(t1, t2));
                const auto created =
                    std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
                res.set_content(
                    json{{"created", created},
                         {"data", json::array({json{
                                      {"b64_json",
                                       http::base64_encode(png)}}})}}
                        .dump(-1, ' ', false, kLossy),
                    "application/json");
            } catch (const std::exception& e) {
                set_error(res, e);
            }
    };

    // OpenAI Responses API (POST /v1/responses), stateless (i#279):
    // OpenAI Codex speaks only this. The request becomes chat messages
    // (responses.hpp); the reply is one output item, a message or a
    // function call. With tools offered the text is buffered until the
    // end, as on the chat path, so a tool call never streams as text.
    auto handle_responses = [this](const http::Request& req,
                                   http::ServerResponse& res,
                                   const http::RequestContext& ctx) {
        auth_event("create", "responses", ctx.identity);
        json body;
        ResponsesRequest rq;
        try {
            body = json::parse(req.body);
            rq = parse_responses_request(body);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(make_error(e.what()).dump(-1, ' ', false, kLossy),
                            "application/json");
            return;
        }
        // No max_output_tokens means "as long as needed" in this API
        // (Codex sends none), so the cap is the default here.
        const auto max_tokens = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(
                rq.max_output_tokens.value_or(opt_.max_tokens_cap),
                opt_.max_tokens_cap));
        auto [sampling, seed] = parse_sampling(body);
        auto prompt_ids =
            tok_.encode(opt_.chat_template.apply(rq.messages), true);
        const auto n_prompt = static_cast<std::uint32_t>(prompt_ids.size());
        if (const std::string why = loop_.admit_error(n_prompt);
            !why.empty()) {
            res.status = 400;
            res.set_content(make_error(why).dump(-1, ' ', false, kLossy),
                            "application/json");
            return;
        }
        requests_total_.fetch_add(1, std::memory_order_relaxed);
        prompt_tokens_total_.fetch_add(n_prompt, std::memory_order_relaxed);
        const std::uint64_t id = loop_.submit(std::move(prompt_ids),
                                              max_tokens, sampling, seed,
                                              nullptr, {});
        auto r = std::make_shared<ResponsesResult>();
        r->id = "resp_" + std::to_string(id);
        r->model = opt_.model_name;
        r->created_at = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now()
                                .time_since_epoch())
                            .count();
        r->input_tokens = n_prompt;
        auto rq_sp = std::make_shared<ResponsesRequest>(std::move(rq));
        // Fills r from a finished generation.
        auto settle = [this, r, rq_sp](const EngineLoop::View& v) {
            const bool hit_eos = !v.generated.empty() &&
                                 v.generated.back() == tok_.eos_id();
            r->text = tok_.decode(v.generated);
            r->call = detect_tool_call(r->text, rq_sp->tools);
            if (r->call) r->text.clear();
            r->incomplete = !hit_eos && !r->call;
            r->cached_tokens = v.reused_prefix_tokens;
            r->output_tokens = static_cast<std::uint32_t>(v.generated.size());
            completion_tokens_total_.fetch_add(r->output_tokens,
                                               std::memory_order_relaxed);
        };

        if (!rq_sp->stream) {
            auto v = loop_.wait_done_or_cancel(id, ctx.client_alive);
            loop_.release(id);
            if (v.status == engine::Status::kCancelled) {
                res.status = 499;
                res.set_content(
                    make_error("request cancelled: client closed the "
                               "connection")
                        .dump(-1, ' ', false, kLossy),
                    "application/json");
                return;
            }
            if (v.status == engine::Status::kFailed) {
                res.status = 500;
                res.set_content(make_error(v.error).dump(-1, ' ', false, kLossy),
                                "application/json");
                return;
            }
            settle(v);
            res.set_content(
                responses_object(*r,
                                 json::array({responses_output_item(*r,
                                                                    *rq_sp)}),
                                 r->incomplete ? "incomplete" : "completed")
                    .dump(-1, ' ', false, kLossy),
                "application/json");
            return;
        }

        res.set_chunked_content_provider(
            "text/event-stream",
            [this, id, r, rq_sp, settle](http::Sink& sink) {
                ResponsesSse sse;
                std::size_t seen = 0;
                bool opened = false;
                const bool live = rq_sp->tools.empty();
                std::string payload = sse.start(*r);
                for (;;) {
                    auto v = loop_.wait_progress(id, seen);
                    if (live) {
                        for (std::size_t i = seen; i < v.generated.size();
                             ++i) {
                            const tok::TokenId t = v.generated[i];
                            if (t == tok_.eos_id()) continue;
                            if (!opened) {
                                payload += sse.message_open(*r);
                                opened = true;
                            }
                            payload += sse.text_delta(*r, tok_.decode({&t, 1}));
                        }
                    }
                    seen = v.generated.size();
                    const bool terminal = v.status == engine::Status::kDone ||
                                          v.status == engine::Status::kFailed;
                    if (terminal) {
                        if (v.status == engine::Status::kFailed) {
                            payload += sse.event(
                                {{"type", "response.failed"},
                                 {"response",
                                  [&] {
                                      json o = responses_object(
                                          *r, json::array(), "failed");
                                      o["error"] = {{"code", "server_error"},
                                                    {"message", v.error}};
                                      return o;
                                  }()}});
                        } else {
                            settle(v);
                            payload += sse.finish(*r, *rq_sp, opened);
                        }
                        loop_.release(id);
                    }
                    if (!payload.empty() &&
                        !sink.write(std::string_view(payload.data(),
                                                     payload.size()))) {
                        loop_.cancel(id);
                        loop_.release(id);
                        return;
                    }
                    payload.clear();
                    if (terminal) {
                        sink.done();
                        return;
                    }
                }
            });
    };

    // One dispatch handler over all routes (route_of is the single home
    // for routing, shared with the auth head stage below).
    route_handler_ = [this, handle_health, handle_models,
                      handle_model_by_id, handle_metrics, handle,
                      handle_messages, handle_responses, handle_embeddings,
                      handle_audio, handle_images](
                         const http::Request& req, http::ServerResponse& res,
                         const http::RequestContext& ctx) {
        switch (route_of(req.method, req.path)) {
            case Route::kHealth:      handle_health(req, res); break;
            case Route::kModels:      handle_models(req, res); break;
            case Route::kModelById:   handle_model_by_id(req, res); break;
            case Route::kMetrics:     handle_metrics(req, res); break;
            case Route::kCompletions: handle(req, res, ctx, false); break;
            case Route::kChat:        handle(req, res, ctx, true); break;
            case Route::kMessages:    handle_messages(req, res, ctx); break;
            case Route::kResponses:   handle_responses(req, res, ctx); break;
            case Route::kEmbeddings:  handle_embeddings(req, res, ctx); break;
            case Route::kAudioSpeech: handle_audio(req, res, ctx); break;
            case Route::kImages:      handle_images(req, res, ctx); break;
            case Route::kUnknown:
                res.status = 404;  // httplib's default: 404, empty body
                break;
        }
    };

    // Auth on the HEAD, before the body: /health is open and an unknown
    // path 404s without auth (both preserve the pre-port behaviour); every
    // other route authenticates. On deny authorize() fills the 401 (the
    // byte-identical shape-B body) and we reject; on allow we thread the
    // identity to the handler via ctx and proceed.
    auth_head_ = [this](const http::Request& head, http::ServerResponse& res,
                        http::RequestContext& ctx) -> bool {
        switch (route_of(head.method, head.path)) {
            case Route::kHealth:
            case Route::kUnknown:
                return false;  // proceed without auth
            default: {
                std::string identity;
                if (!authorize(head, res, identity)) {
                    return true;  // reject with the 401 authorize() set
                }
                ctx.identity = std::move(identity);
                return false;  // proceed
            }
        }
    };
}

}  // namespace locus::server
