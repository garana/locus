#include "locus/server/server.hpp"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
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
#include "locus/http/json_body.hpp"
#include "locus/backend/fallbacks.hpp"
#include "locus/backend/gpu_metrics.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/image/classifier.hpp"
#include "locus/image/image_input.hpp"
#include "locus/image/janus.hpp"
#include "locus/image/janus_chat.hpp"
#include "locus/image/janus_vision.hpp"
#include "locus/image/flux_pipeline.hpp"
#include "locus/image/ltxv_pipeline.hpp"
#include "locus/image/pixart_pipeline.hpp"
#include "locus/image/png.hpp"
#include "locus/image/qwen_edit_pipeline.hpp"
#include "locus/image/vq16.hpp"
#include "locus/model/codec.hpp"
#include "locus/model/expert_metrics.hpp"
#include "locus/model/gbnf.hpp"
#include "locus/model/json_schema_gbnf.hpp"
#include "locus/model/sampling.hpp"
#include "locus/server/errors.hpp"
#include "locus/server/metrics_text.hpp"
#include "locus/server/responses.hpp"
#include "locus/server/image_lora.hpp"
#include "locus/server/tools.hpp"
#include "locus/sys/features.hpp"
#include "locus/sys/perf_counters.hpp"
#include "locus/video/jobs.hpp"

namespace locus::server {

namespace {

using nlohmann::json;

/** dump()'s error handler for every response body: invalid UTF-8 (a
 * parse error quoting the request, text cut mid-character) becomes
 * U+FFFD instead of throwing type_error.316 out of a handler (i#243). */
constexpr auto kLossy = json::error_handler_t::replace;

/**
 * An SSE stream's keepalive (i#619): a stream that has written nothing
 * for `every` gets an SSE comment, which clients ignore, so a long
 * prefill or a buffered tool reply does not look like a dead
 * connection.
 */
struct SseKeepalive {
    std::chrono::milliseconds every;
    std::chrono::steady_clock::time_point last =
        std::chrono::steady_clock::now();

    /** Fills an empty payload with ": keepalive" when one is due, and
     *  notes the time of any payload about to be written. */
    void pad(std::string& payload) {
        const auto now = std::chrono::steady_clock::now();
        if (payload.empty() && every.count() > 0 && now - last >= every) {
            payload = ": keepalive\n\n";
        }
        if (!payload.empty()) {
            last = now;
        }
    }
};

/**
 * Writes one step of a generation's SSE stream: payload, or a keepalive
 * when one is due. When the client has gone, cancels the generation and
 * drops its record (both no-ops if the terminal path already released
 * it).
 *
 * @param terminal Whether this is the stream's last step; the sink is
 *     then closed.
 * @returns false when the stream is over: the client went away, or
 *     terminal.
 */
bool write_sse_step(http::Sink& sink, SseKeepalive& ka, std::string& payload,
                    EngineLoop& loop, std::uint64_t id, bool terminal) {
    ka.pad(payload);
    if (!payload.empty() &&
        !sink.write(std::string_view(payload.data(), payload.size()))) {
        loop.cancel(id);
        loop.release(id);
        return false;
    }
    if (terminal) {
        sink.done();
        return false;
    }
    return true;
}

/** The GPU ops for the embedding and rerank models' large passes
 *  (EmbeddingModel::kGpuMacs): the Vulkan image ops where the GPU
 *  imports host memory, as for PixArt (the weights stay in the GGUF
 *  mapping, #512); else the CUDA ops when a CUDA device is usable, the
 *  matrices through its batched matvec (i#519); else the Vulkan image
 *  ops on a device measured faster than the CPU (a discrete GPU), which
 *  copy the weights into the device pool: under 700 MB for these models
 *  (i#680); else null, the CPU everywhere. Not on V3D or llvmpipe,
 *  which run these passes slower than the CPU (i#612). */
const backend::Ops* encoder_gpu_ops() {
    if (backend::vulkan_imports_host_memory()) {
        return &backend::vulkan_image_ops();
    }
    const backend::Backend* cuda = backend::find_backend("cuda");
    if (cuda != nullptr && cuda->available) {
        return &cuda->ops;
    }
    if (backend::vulkan_backend_usable() &&
        backend::vulkan_device_beats_cpu()) {
        return &backend::vulkan_image_ops();
    }
    return nullptr;
}

/** @returns The OpenAI flat error envelope. @param type
 *  invalid_request_error for a request the client got wrong; api_error
 *  for a condition that is not the request's fault (a cancellation). */
json make_error(const std::string& message,
                const char* type = "invalid_request_error") {
    return json{{"error", {{"message", message}, {"type", type}}}};
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
 * Takes the image lock without waiting. An image takes minutes, so when
 * another image or video holds it, a request gets a 503 instead of
 * holding a server thread in a queue.
 *
 * @param retry_after_s The Retry-After of that 503, in seconds.
 * @returns The lock; when it does not own mu, res holds the 503.
 */
std::unique_lock<std::mutex> try_image_lock(std::mutex& mu,
                                            http::ServerResponse& res,
                                            int retry_after_s) {
    std::unique_lock<std::mutex> lk(mu, std::try_to_lock);
    if (!lk.owns_lock()) {
        set_unavailable(res,
                        "an image or video is already being generated; "
                        "retry later",
                        retry_after_s);
    }
    return lk;
}

/** Fills res with an OpenAI images reply: one PNG, base64-encoded. */
void set_png_reply(http::ServerResponse& res,
                   std::span<const std::byte> png) {
    const auto created = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::system_clock::now()
                                 .time_since_epoch())
                             .count();
    res.set_content(
        json{{"created", created},
             {"data",
              json::array({json{{"b64_json", http::base64_encode(png)}}})}}
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

/**
 * Parses and validates the OpenAI messages array. Tool history renders
 * as /v1/responses renders it: an assistant entry's tool_calls become
 * one assistant turn each (render_tool_call; its content, if any,
 * comes first), and a "tool" entry becomes a user turn
 * (render_tool_result) named after its tool_call_id's call.
 *
 * @param has_tool_history Set when the history holds a tool call or
 *     result.
 */
std::vector<chat::Message> parse_messages(const json& messages,
                                          bool* has_tool_history = nullptr) {
    std::vector<chat::Message> out;
    std::map<std::string, std::string> call_names;  // tool_call_id -> name
    const auto text_of = [](const json& m) {
        return m.contains("content") && !m["content"].is_null()
                   ? openai_text(m["content"])
                   : std::string();
    };
    for (const auto& m : messages) {
        if (!m.is_object()) {
            throw std::invalid_argument("messages must be objects");
        }
        const std::string role = m.value("role", "user");
        const bool calls = role == "assistant" && m.contains("tool_calls") &&
                           m["tool_calls"].is_array() &&
                           !m["tool_calls"].empty();
        if (role == "tool") {
            const auto it = call_names.find(m.value("tool_call_id", ""));
            const std::string name =
                it != call_names.end() ? it->second : m.value("name", "");
            out.push_back({"user", render_tool_result(name, text_of(m))});
            if (has_tool_history != nullptr) *has_tool_history = true;
            continue;
        }
        if (role != "system" && role != "user" && role != "assistant") {
            throw std::invalid_argument("unsupported message role: " + role);
        }
        if (!calls && !m.contains("content")) {
            throw std::invalid_argument("message content is missing");
        }
        const std::string text = text_of(m);
        if (!calls || !text.empty()) out.push_back({role, text});
        if (!calls) continue;
        for (const auto& c : m["tool_calls"]) {
            const json f = c.is_object() ? c.value("function", json::object())
                                         : json::object();
            const std::string name = f.value("name", "");
            if (c.is_object() && c.contains("id") && c["id"].is_string()) {
                call_names[c["id"].get<std::string>()] = name;
            }
            out.push_back(
                {"assistant",
                 render_tool_call(name, f.value("arguments", json()))});
        }
        if (has_tool_history != nullptr) *has_tool_history = true;
    }
    return out;
}

/**
 * Anthropic content is either a plain string or an array of content
 * blocks; we take the text of every {"type":"text"} block and
 * ignore the rest (parse_anthropic_messages renders tool_use and
 * tool_result blocks). @returns the text blocks joined with newlines,
 * as the vision
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
 *
 * Tool history renders as on the OpenAI paths (i#590), in block
 * order: a tool_use block is an assistant turn (render_tool_call of
 * its name and input), a tool_result block a user turn
 * (render_tool_result, named after its tool_use_id's call; an image in
 * it becomes "[image omitted]"); the text blocks around them are turns
 * of the message's role.
 *
 * @param has_tool_history Set when the history holds a tool block.
 */
std::vector<chat::Message> parse_anthropic_messages(
    const json& body, bool* has_tool_history = nullptr) {
    std::vector<chat::Message> out;
    std::string system;
    if (body.contains("system") && !body["system"].is_null()) {
        system = anthropic_text(body["system"]);
    }
    std::map<std::string, std::string> call_names;  // tool_use id -> name
    // A tool_result's content: a string, or blocks (text, images).
    const auto result_text = [](const json& c) {
        if (c.is_string()) return c.get<std::string>();
        std::string t;
        if (!c.is_array()) return t;
        for (const auto& b : c) {
            const std::string type = b.is_object() ? b.value("type", "") : "";
            std::string piece;
            if (type == "text" && b.contains("text") && b["text"].is_string()) {
                piece = b["text"].get<std::string>();
            } else if (type == "image") {
                piece = "[image omitted]";
            } else {
                continue;
            }
            if (!t.empty()) t += "\n";
            t += piece;
        }
        return t;
    };
    for (const auto& m : body.at("messages")) {
        const std::string role = m.value("role", "user");
        if (role != "user" && role != "assistant" && role != "system") {
            throw std::invalid_argument("unsupported message role: " + role);
        }
        if (!m.contains("content")) {
            throw std::invalid_argument(
                "message must have content");
        }
        const json& content = m["content"];
        if (role == "system") {
            const std::string text = anthropic_text(content);
            if (!system.empty() && !text.empty()) system += "\n";
            system += text;
            continue;
        }
        const auto is_tool = [](const json& b) {
            const std::string t = b.is_object() ? b.value("type", "") : "";
            return t == "tool_use" || t == "tool_result";
        };
        const bool tools =
            content.is_array() &&
            std::any_of(content.begin(), content.end(), is_tool);
        if (!tools) {
            out.push_back({role, anthropic_text(content)});
            continue;
        }
        if (has_tool_history != nullptr) *has_tool_history = true;
        std::string text;  // text blocks since the last tool block
        const auto flush = [&] {
            if (!text.empty()) out.push_back({role, std::move(text)});
            text.clear();
        };
        for (const auto& b : content) {
            const std::string type = b.is_object() ? b.value("type", "") : "";
            if (type == "text" && b.contains("text") && b["text"].is_string()) {
                if (!text.empty()) text += "\n";
                text += b["text"].get<std::string>();
            } else if (type == "tool_use") {
                flush();
                const std::string name = b.value("name", "");
                if (b.contains("id") && b["id"].is_string()) {
                    call_names[b["id"].get<std::string>()] = name;
                }
                out.push_back(
                    {"assistant",
                     render_tool_call(name, b.value("input", json()))});
            } else if (type == "tool_result") {
                flush();
                const auto it = call_names.find(b.value("tool_use_id", ""));
                std::string output =
                    b.contains("content") ? result_text(b["content"]) : "";
                if (b.value("is_error", false)) output = "Error: " + output;
                out.push_back(
                    {"user",
                     render_tool_result(
                         it != call_names.end() ? it->second : "", output)});
            }
        }
        flush();
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
    // Device time by forward stage (vulkan); "other" is the time outside
    // a marked stage. Only stages with time are listed.
    mt.family("locus_gpu_device_stage_seconds_total",
              "Device time by forward stage, estimated from sampled "
              "batches, fewer than the device time's (vulkan).",
              "counter");
    for (const GpuApi a : used) {
        const backend::GpuMetrics& gm = backend::gpu_metrics(a);
        for (int i = 0; i < static_cast<int>(model::Stage::kCount); ++i) {
            const std::uint64_t ns = gm.stage_ns[i].load(kRelaxed);
            if (ns == 0) continue;
            mt.value("locus_gpu_device_stage_seconds_total",
                     api(a) + "," +
                         MetricsText::label(
                             "stage",
                             model::stage_name(static_cast<model::Stage>(i))),
                     secs(ns));
        }
        const std::uint64_t rest = gm.unmarked_ns.load(kRelaxed);
        if (rest != 0) {
            mt.value("locus_gpu_device_stage_seconds_total",
                     api(a) + "," + MetricsText::label("stage", "other"),
                     secs(rest));
        }
    }
    counter("locus_gpu_device_samples_total",
            "Batches timed for the device time estimate (vulkan).",
            [&](const backend::GpuMetrics& gm) {
                return static_cast<double>(gm.device_samples.load(kRelaxed));
            });
    counter("locus_gpu_device_stage_samples_total",
            "Batches timed by stage for the device stage estimate (vulkan).",
            [&](const backend::GpuMetrics& gm) {
                return static_cast<double>(gm.stage_samples.load(kRelaxed));
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

/**
 * Writes locus_moe_expert_calls_total{layer,expert,device,api} (i#252):
 * how often each routed expert ran, and where. Only experts that ran at
 * least once have a series.
 */
void render_expert_calls(MetricsText& mt) {
    const auto calls = model::ExpertCalls::snapshot();
    if (calls.empty()) return;
    mt.family("locus_moe_expert_calls_total",
              "Routed MoE expert calls, per layer, expert and device.",
              "counter");
    for (const auto& c : calls) {
        mt.value("locus_moe_expert_calls_total",
                 MetricsText::label("layer", std::to_string(c.layer)) + "," +
                     MetricsText::label("expert", std::to_string(c.expert)) +
                     "," + MetricsText::label("device", c.device) + "," +
                     MetricsText::label("api", c.api),
                 c.calls);
    }
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

/**
 * The grammar of one output-format object. `nested` is the Chat
 * Completions form (the schema under json_schema.schema); the
 * Responses and Anthropic forms carry it as `schema`. `ptr` is the
 * object's JSON pointer in the body, where the schema is read again
 * with its key order kept.
 */
OutputGrammar format_grammar(const json& f, std::string_view raw,
                             const std::string& ptr, bool nested) {
    if (!f.is_object()) {
        throw std::invalid_argument("the output format must be an object");
    }
    const std::string type = f.value("type", "");
    if (type == "text") {
        return {};
    }
    if (type == "json_object") {
        // OpenAI's JSON mode promises an object, not any value.
        return {model::Gbnf::parse(model::json_value_gbnf(), "object"), true};
    }
    if (type == "json_schema") {
        const json* s = nested && f.contains("json_schema") &&
                                f["json_schema"].is_object() &&
                                f["json_schema"].contains("schema")
                            ? &f["json_schema"]["schema"]
                        : !nested && f.contains("schema") ? &f["schema"]
                                                          : nullptr;
        if (s == nullptr || !(s->is_object() || s->is_boolean())) {
            throw std::invalid_argument(
                nested ? "json_schema.schema must be a JSON Schema object"
                       : "schema must be a JSON Schema object");
        }
        const auto ordered = http::parse_json<nlohmann::ordered_json>(raw);
        const std::string at = ptr + (nested ? "/json_schema/schema" : "/schema");
        return {model::Gbnf::parse(model::json_schema_to_gbnf(
                    ordered.at(nlohmann::ordered_json::json_pointer(at))
                        .dump())),
                true};
    }
    // locus extensions: a GBNF grammar (rule "root") or a regex.
    if (type == "grammar") {
        if (!f.contains("grammar") || !f["grammar"].is_string()) {
            throw std::invalid_argument("grammar must be a GBNF string");
        }
        return {model::Gbnf::parse(f["grammar"].get<std::string>()), false};
    }
    if (type == "regex") {
        if (!f.contains("regex") || !f["regex"].is_string()) {
            throw std::invalid_argument("regex must be a string");
        }
        return {model::Gbnf::parse(
                    "root ::= " +
                    model::regex_to_gbnf(f["regex"].get<std::string>())),
                false};
    }
    throw std::invalid_argument(
        "unsupported output format type '" + type +
        "' (text, json_object, json_schema, grammar or regex)");
}

/**
 * The grammar a request asks its output to follow (i#402), or nullptr
 * for free text. Read from `response_format` (Chat Completions and
 * Completions), `text.format` (Responses), and `output_config.format`
 * or the older `output_format` (Anthropic Messages).
 * @param body The parsed body.
 * @param raw The body text, parsed again only for a schema, so its
 *     properties keep their order.
 * @throws std::invalid_argument on a malformed or unsupported format,
 *     a bad schema, grammar or regex.
 */
OutputGrammar output_grammar(const json& body, std::string_view raw) {
    if (body.contains("response_format") &&
        body["response_format"].is_object()) {
        return format_grammar(body["response_format"], raw,
                              "/response_format", true);
    }
    if (body.contains("output_config") &&
        body["output_config"].is_object() &&
        body["output_config"].contains("format")) {
        return format_grammar(body["output_config"]["format"], raw,
                              "/output_config/format", false);
    }
    if (body.contains("output_format")) {
        return format_grammar(body["output_format"], raw, "/output_format",
                              false);
    }
    if (body.contains("text") && body["text"].is_object() &&
        body["text"].contains("format")) {
        return format_grammar(body["text"]["format"], raw, "/text/format",
                              false);
    }
    return {};
}

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

/** @returns The bytes of an image given as a data: URL or as bare
 *  base64 (/v1/classify, image inputs to /v1/embeddings). */
std::vector<std::byte> image_input_bytes(const json& v,
                                         std::size_t max_bytes) {
    if (!v.is_string()) {
        throw std::invalid_argument(
            "an image must be a base64 string or a data: URL");
    }
    const std::string s = v.get<std::string>();
    if (s.rfind("data:", 0) == 0) return data_url_bytes(v, max_bytes);
    return http::base64_decode(s, max_bytes);
}

/**
 * @param s "WxH": two decimal sides of 1 to 4 digits, no signs or spaces.
 * @returns {width, height}.
 * @throws std::invalid_argument otherwise.
 */
std::pair<std::uint32_t, std::uint32_t> parse_size_wxh(const std::string& s) {
    const auto x = s.find('x');
    const auto digits = [](std::string_view d) {
        return !d.empty() && d.size() <= 4 &&
               std::all_of(d.begin(), d.end(),
                           [](char c) { return c >= '0' && c <= '9'; });
    };
    if (x == std::string::npos || !digits(std::string_view(s).substr(0, x)) ||
        !digits(std::string_view(s).substr(x + 1))) {
        throw std::invalid_argument("size must be WIDTHxHEIGHT");
    }
    return {static_cast<std::uint32_t>(std::stoul(s.substr(0, x))),
            static_cast<std::uint32_t>(std::stoul(s.substr(x + 1)))};
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
 * The output limit of a completions or chat body: max_completion_tokens
 * (OpenAI's current name, sent by the Antigravity SDK) or else
 * max_tokens; a null value counts as absent.
 *
 * @param cap The server's --max-tokens; a larger request is clamped.
 * @param dflt The limit when the body sets none.
 * @throws std::invalid_argument when the value is not a positive integer.
 */
std::uint32_t read_max_tokens(const json& body, std::uint32_t cap,
                              std::uint32_t dflt) {
    for (const char* key : {"max_completion_tokens", "max_tokens"}) {
        if (!body.contains(key) || body[key].is_null()) continue;
        const auto& v = body[key];
        if (!v.is_number_unsigned() || v.get<std::uint64_t>() == 0) {
            throw std::invalid_argument(std::string(key) +
                                        " must be a positive integer");
        }
        return static_cast<std::uint32_t>(
            std::min<std::uint64_t>(v.get<std::uint64_t>(), cap));
    }
    return dflt;
}

/**
 * The sampling seed of a generation body: its seed, or a random one when
 * the body sets none.
 *
 * @throws std::invalid_argument when seed is not an integer.
 */
std::uint64_t read_seed(const json& body) {
    if (!body.contains("seed")) return std::random_device{}();
    if (!body["seed"].is_number_integer()) {
        throw std::invalid_argument("seed must be an integer");
    }
    return static_cast<std::uint64_t>(body["seed"].get<std::int64_t>());
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
    if (body.contains("tools") || output_grammar(body, body.dump())) {
        throw std::invalid_argument(
            "tools and output formats are not supported with image input "
            "yet");
    }
    vc.max_tokens = read_max_tokens(body, max_tokens_cap, vc.max_tokens);
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
    if (!opt_.embedding_model_path.empty()) {
        bert_ = embed::EmbeddingModel::load(opt_.embedding_model_path);
        if (bert_->name() == opt_.model_name) {
            throw gguf::Error("--embedding-model " + bert_->name() +
                              " has the chat model's name");
        }
        if (bert_->reranker()) {
            throw gguf::Error("--embedding-model " + bert_->name() +
                              " is a reranker; pass it as --rerank-model");
        }
    }
    if (!opt_.rerank_model_path.empty()) {
        rerank_ = embed::EmbeddingModel::load(opt_.rerank_model_path);
        if (!rerank_->reranker()) {
            throw gguf::Error("--rerank-model " + rerank_->name() +
                              " is not a reranker (pooling_type rank)");
        }
        if (rerank_->name() == opt_.model_name ||
            (bert_ && rerank_->name() == bert_->name())) {
            throw gguf::Error("--rerank-model " + rerank_->name() +
                              " has another model's name");
        }
    }
    if (!opt_.image_classifier_path.empty()) {
        classifier_ = image::ImageClassifier::open(opt_.image_classifier_path);
        const std::string& n = classifier_->name();
        if (n == opt_.model_name || (bert_ && n == bert_->name()) ||
            (rerank_ && n == rerank_->name())) {
            throw gguf::Error("--image-classifier " + n +
                              " has another model's name");
        }
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
    // flux or clip_text GGUF (parts only FLUX.1 has) makes it a FLUX.1
    // pipeline, so a FLUX set missing a part names the FLUX parts; a
    // qwen_image GGUF makes it Qwen-Image-Edit; else PixArt, whose
    // error names both sets.
    if (!opt_.image_model_dir.empty()) {
        bool flux = false, qwen = false;
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(
                 opt_.image_model_dir, ec)) {
            if (!e.is_regular_file() || e.path().extension() != ".gguf") {
                continue;
            }
            // arch views into g, so g must outlive the comparison.
            const auto g = gguf::GgufFile::open(e.path().string());
            const auto arch = g.get_string("general.architecture");
            if (arch == "flux" || arch == "clip_text") flux = true;
            if (arch == "qwen_image") qwen = true;
        }
        if (qwen) {
            qwen_edit_ =
                image::QwenEditPipeline::load_dir(opt_.image_model_dir);
        } else if (flux) {
            flux_ = image::FluxPipeline::load_dir(opt_.image_model_dir);
        } else {
            try {
                pixart_ =
                    image::PixArtPipeline::load_dir(opt_.image_model_dir);
            } catch (const gguf::Error& e) {
                throw gguf::Error(std::string(e.what()) +
                                  " (or, for FLUX.1, a flux, a clip_text, a "
                                  "t5encoder and an autoencoder_kl GGUF)");
            }
        }
    }
    // FLUX.1 LoRAs (--lora-dir): every one is checked against the DiT
    // now, so a LoRA for another model stops the server, not a request.
    if (!opt_.lora_dir.empty()) {
        if (!flux_) {
            throw gguf::Error("--lora-dir needs --image-model with FLUX.1");
        }
        loras_ = image::LoraLibrary::load_dir(opt_.lora_dir);
        for (const std::string& name : loras_->names()) {
            const image::LoraPick pick{name, 1.0f};
            try {
                flux_->check_lora(loras_->select({&pick, 1}));
            } catch (const std::invalid_argument& e) {
                throw gguf::Error("lora " + name + ": " + e.what());
            }
        }
    }
    // Video (--video-model): a broken directory stops the server; a
    // missing ffmpeg only turns the video routes into 501s.
    if (!opt_.video_model_dir.empty()) {
        ltxv_ = image::LtxvPipeline::load_dir(opt_.video_model_dir);
        ffmpeg_ = video::probe_ffmpeg(opt_.ffmpeg_path);
        if (!ffmpeg_) {
            std::fprintf(stderr,
                         "warning: no usable ffmpeg at \"%s\" (--ffmpeg); "
                         "/v1/videos will answer 501\n",
                         opt_.ffmpeg_path.c_str());
        }
        // At most 4 clips queued or running; a finished one is kept an
        // hour.
        video_jobs_ = std::make_unique<video::VideoJobs>(
            (std::filesystem::temp_directory_path() /
             ("locus-videos-" + std::to_string(::getpid())))
                .string(),
            4, 3600);
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
    if ((pixart_ || flux_ || qwen_edit_) && be != "vulkan" &&
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
        // With host-memory import (unified memory, as for PixArt and
        // FLUX) the image tower runs on the GPU with its activations
        // resident (JanusVision::features_vulkan), whatever backend the
        // language model uses. Otherwise per op it would make each
        // linear a separate device call, so it runs on the active
        // backend when that is a CPU one, else on the best CPU backend.
        const auto& active = model_.active_backend().ops;
        const auto& op = backend::vulkan_imports_host_memory()
                             ? backend::vulkan_image_ops()
                         : backend::splits_rows(active)
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

std::shared_ptr<const model::TokenTrie> OpenAiServer::token_trie() {
    std::call_once(trie_once_, [this] {
        auto pieces = std::make_shared<std::vector<std::string>>();
        pieces->reserve(n_vocab_);
        for (std::uint32_t t = 0; t < n_vocab_; ++t) {
            const tok::TokenId id = static_cast<tok::TokenId>(t);
            pieces->push_back(tok_.decode({&id, 1}));
        }
        trie_ = std::make_shared<const model::TokenTrie>(std::move(pieces));
    });
    return trie_;
}

std::unique_ptr<model::TokenConstraint> OpenAiServer::output_constraint(
    OutputGrammar g) {
    if (g.g == nullptr) {
        return nullptr;
    }
    return std::make_unique<model::GrammarConstraint>(
        std::move(g.g), token_trie(), tok_.eos_id(), g.json);
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
        // /v1/videos/{id} and /v1/videos/{id}/content; an id has no '/'.
        static const std::string kVideosPrefix = "/v1/videos/";
        static const std::string kContent = "/content";
        if (path.rfind(kVideosPrefix, 0) == 0) {
            std::string_view rest(path);
            rest.remove_prefix(kVideosPrefix.size());
            const bool content =
                rest.size() > kContent.size() &&
                rest.substr(rest.size() - kContent.size()) == kContent;
            if (content) rest.remove_suffix(kContent.size());
            if (!rest.empty() && rest.find('/') == std::string_view::npos) {
                return content ? Route::kVideoContent : Route::kVideoById;
            }
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
        if (path == "/v1/rerank" || path == "/rerank") {
            return Route::kRerank;
        }
        if (path == "/v1/classify") {
            return Route::kClassify;
        }
        if (path == "/v1/audio/speech") {
            return Route::kAudioSpeech;
        }
        if (path == "/v1/images/generations") {
            return Route::kImages;
        }
        if (path == "/v1/images/edits") {
            return Route::kImageEdits;
        }
        if (path == "/v1/videos") {
            return Route::kVideos;
        }
        return Route::kUnknown;
    }
    return Route::kUnknown;
}

int OpenAiServer::build_and_start(const std::string& host, int port) {
    http::ServerConfig cfg;
    cfg.host = host;
    cfg.port = port;
    // The oversize-body cap: the parser answers 413 to a too-large
    // declared body on the head, before the body is read.
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
            "/v1/responses", "/v1/embeddings", "/v1/rerank", "/rerank",
            "/v1/classify", "/v1/audio/speech",
            "/v1/images/generations", "/v1/images/edits"});
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

    // OpenAI model listing: the chat model, then the embedding model
    // when one is loaded. Auth (when on) is enforced by the head stage
    // before this runs. `locus_kinds` (an extension OpenAI clients
    // ignore) lists the request kinds this instance serves, for
    // locus-router (#388), which routes by these ids.
    auto handle_models = [this](const http::Request&,
                                http::ServerResponse& res) {
            json data = json::array({json{{"id", opt_.model_name},
                                          {"object", "model"},
                                          {"created", 0},
                                          {"owned_by", "locus"}}});
            for (const auto* m : {bert_.get(), rerank_.get()}) {
                if (m == nullptr) continue;
                data.push_back({{"id", m->name()},
                                {"object", "model"},
                                {"created", 0},
                                {"owned_by", "locus"}});
            }
            if (classifier_) {
                data.push_back({{"id", classifier_->name()},
                                {"object", "model"},
                                {"created", 0},
                                {"owned_by", "locus"}});
            }
            json kinds = json::array({"text"});
            if (vision_) kinds.push_back("image-to-text");
            // Generations and edits apart (i#633): FLUX.1 Kontext serves
            // both, other FLUX.1 checkpoints generate (i#648),
            // Qwen-Image-Edit edits only, Janus and PixArt generate only.
            if (janus_ || pixart_ || flux_) kinds.push_back("image");
            if ((flux_ && flux_->may_edit()) || qwen_edit_) {
                kinds.push_back("image-edit");
            }
            if (ltxv_) kinds.push_back("video");
            if (snac_) kinds.push_back("speech");
            if (rerank_) kinds.push_back("rerank");
            if (classifier_) kinds.push_back("classify");
            res.set_content(
                json{{"object", "list"},
                     {"data", data},
                     {"locus_kinds", kinds}}
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
            if (model_id != opt_.model_name &&
                !(bert_ && model_id == bert_->name()) &&
                !(rerank_ && model_id == rerank_->name()) &&
                !(classifier_ && model_id == classifier_->name())) {
                res.status = 404;
                res.set_content(
                    make_error("model not found").dump(-1, ' ', false, kLossy),
                    "application/json");
                return;
            }
            res.set_content(json{{"id", model_id},
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
            counter("locus_grammar_stack_overflows_total",
                    "Structured-output replies whose grammar state had "
                    "more parse stacks than the cap and dropped the rest.",
                    model::gbnf_stack_overflows());
            counter("locus_prefix_reused_tokens_total",
                    "Prompt tokens served from the prefix cache.",
                    s.prefix_reused_tokens);
            counter("locus_disk_prefix_hits_total",
                    "Requests whose prefix was read from the disk tier.",
                    s.disk_hits);
            counter("locus_disk_prefix_loaded_tokens_total",
                    "Positions read back from the disk tier.",
                    s.disk_loaded_tokens);
            counter("locus_disk_prefix_swapped_tokens_total",
                    "Positions of preempted sequences written to the disk "
                    "tier.",
                    s.disk_swapped_tokens);
            counter("locus_disk_prefix_refused_tokens_total",
                    "Positions of preempted sequences the disk tier "
                    "refused, recomputed on readmission.",
                    s.disk_refused_tokens);
            counter("locus_spec_accepted_tokens_total",
                    "Speculative draft tokens accepted.",
                    s.spec_accepted_tokens);
            counter("locus_spec_steps_total",
                    "Speculative verify forwards run.",
                    s.spec_steps);
            counter("locus_spec_drafted_tokens_total",
                    "Speculative draft tokens verified, accepted or not.",
                    s.spec_drafted_tokens);
            if (!s.spec_draft_counts.empty()) {
                mt.family("locus_spec_verifies_total",
                          "Speculative verify forwards by draft length.",
                          "counter");
                for (std::size_t d = 0; d < s.spec_draft_counts.size();
                     ++d) {
                    if (s.spec_draft_counts[d] > 0) {
                        mt.value("locus_spec_verifies_total",
                                 MetricsText::label("draft_tokens",
                                                    std::to_string(d)),
                                 s.spec_draft_counts[d]);
                    }
                }
            }
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
            gauge("locus_batch_tokens",
                  "Tokens the model computed in the last engine step "
                  "(prefill, decode and speculative verify); 0 when "
                  "idle.",
                  em.batch_tokens.load(kRelaxed));
            // Each block in use counted once; with locus_kv_blocks_free
            // the three sum to locus_kv_blocks_total.
            mt.family("locus_kv_blocks_used",
                      "KV blocks in use after the last step, by what holds "
                      "them: a prefilling sequence, else a decoding one, "
                      "else only the prefix cache.",
                      "gauge");
            for (const auto& [state, v] :
                 {std::pair<const char*, std::uint32_t>{
                      "prefilling", em.kv_blocks_prefilling.load(kRelaxed)},
                  {"decoding", em.kv_blocks_decoding.load(kRelaxed)},
                  {"cached", em.kv_blocks_cached.load(kRelaxed)}}) {
                mt.value("locus_kv_blocks_used",
                         MetricsText::label("state", state),
                         std::uint64_t{v});
            }
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
            render_expert_calls(mt);
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
        OutputGrammar grammar;  // an output format
        try {
            body = http::parse_json(req.body);
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
                bool tool_history = false;
                auto msgs = parse_messages(body["messages"], &tool_history);
                tools = parse_tools(body, /*anthropic=*/false);
                if (!tools.empty()) {
                    msgs.insert(msgs.begin(),
                                {"system",
                                 render_tools_system(tools) +
                                     (tool_history ? kToolResultNote : "")});
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
            max_tokens =
                read_max_tokens(body, opt_.max_tokens_cap, max_tokens);
            stream = body.value("stream", false);
            if (!vision) {
                grammar = output_grammar(body, req.body);
            }
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
        std::unique_ptr<model::TokenConstraint> constraint =
            output_constraint(std::move(grammar));
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
                               "connection",
                               "api_error")
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
              // The server calls a provider ONCE; loop until terminal,
              // one chunk per progress step.
              SseKeepalive ka{
                  std::chrono::milliseconds(opt_.sse_keepalive_ms)};
              for (;;) {
                auto v = loop_.wait_progress(id, *state, ka.every);
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
                    } else {
                        // The OpenAI stream ends with an empty delta
                        // that carries the finish reason, as the
                        // non-streamed reply does.
                        const bool hit_eos =
                            !v.generated.empty() &&
                            v.generated.back() == tok_.eos_id();
                        json last{{"index", 0},
                                  {"finish_reason",
                                   hit_eos ? "stop" : "length"}};
                        if (chat) {
                            last["delta"] = json::object();
                        } else {
                            last["text"] = "";
                        }
                        json chunk{{"id", rid},
                                   {"object", sse_object},
                                   {"choices", json::array({last})}};
                        payload += "data: " +
                                   chunk.dump(-1, ' ', false, kLossy) +
                                   "\n\n";
                    }
                    payload += "data: [DONE]\n\n";
                    // Stream fully built from the snapshot v; drop the
                    // engine record (the provider won't run again).
                    loop_.release(id);
                }
                if (!write_sse_step(sink, ka, payload, loop_, id,
                                    terminal)) {
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
            const json body = http::parse_json(req.body);
            if (!body.contains("messages") ||
                !body["messages"].is_array() ||
                body["messages"].empty()) {
                throw std::invalid_argument(
                    "messages must be a non-empty array");
            }
            vision = parse_vision_chat(body, true, opt_.max_tokens_cap,
                                       opt_.max_body_bytes);
            if (!vision) {
                constraint =
                    output_constraint(output_grammar(body, req.body));
                bool tool_history = false;
                auto msgs = parse_anthropic_messages(body, &tool_history);
                tools = parse_tools(body, /*anthropic=*/true);
                if (!tools.empty()) {
                    msgs.insert(msgs.begin(),
                                {"system",
                                 render_tools_system(tools) +
                                     (tool_history ? kToolResultNote : "")});
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
              SseKeepalive ka{
                  std::chrono::milliseconds(opt_.sse_keepalive_ms)};
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
                auto v = loop_.wait_progress(id, st->emitted, ka.every);
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
                if (!write_sse_step(sink, ka, payload, loop_, id,
                                    terminal)) {
                    return;
                }
              }
            });
    };

    // OpenAI embeddings, L2-normalized. `input` is a string or an array
    // of strings. With --embedding-model, the embedding model serves
    // every request whose model is not the chat model's name (clients
    // often send an OpenAI model name), in one batch; else the chat
    // model's last-token hidden state, one input at a time. Auth runs
    // in the head stage.
    auto handle_embeddings = [this](const http::Request& req,
                                    http::ServerResponse& res,
                                    const http::RequestContext& ctx) {
            auth_event("create", "embeddings", ctx.identity);
            try {
                const json body = http::parse_json(req.body);
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
                // The image classifier's name: each input is an image
                // (base64 or a data: URL), its pooled CNN feature out.
                if (classifier_ && body.contains("model") &&
                    body["model"].is_string() &&
                    body["model"].get<std::string>() ==
                        classifier_->name()) {
                    image::DecodeLimits lim;
                    lim.max_bytes = opt_.max_body_bytes;
                    json data = json::array();
                    for (std::size_t i = 0; i < inputs.size(); ++i) {
                        const auto img = image::decode_image(
                            image_input_bytes(json(inputs[i]),
                                              opt_.max_body_bytes),
                            lim);
                        std::vector<float> e;
                        {
                            std::lock_guard<std::mutex> lk(classifier_mu_);
                            // Its convs run on Vulkan only (not CUDA).
                            e = classifier_
                                    ->run(img, backend::
                                                   vulkan_imports_host_memory())
                                    .embedding;
                        }
                        data.push_back({{"object", "embedding"},
                                        {"index", i},
                                        {"embedding", std::move(e)}});
                    }
                    res.set_content(
                        json{{"object", "list"},
                             {"data", data},
                             {"model", classifier_->name()},
                             {"usage",
                              {{"prompt_tokens", 0}, {"total_tokens", 0}}}}
                            .dump(-1, ' ', false, kLossy),
                        "application/json");
                    return;
                }
                const bool chat = !bert_ ||
                                  (body.contains("model") &&
                                   body["model"].is_string() &&
                                   body["model"].get<std::string>() ==
                                       opt_.model_name);
                json data = json::array();
                if (chat) {
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
                    return;
                }
                std::size_t tokens = 0;
                std::vector<std::vector<float>> vectors;
                {
                    std::lock_guard<std::mutex> lk(bert_mu_);
                    vectors = bert_->embed(inputs,
                                           backend::best_backend().ops,
                                           &tokens, encoder_gpu_ops());
                }
                for (std::size_t i = 0; i < vectors.size(); ++i) {
                    data.push_back({{"object", "embedding"},
                                    {"index", i},
                                    {"embedding", std::move(vectors[i])}});
                }
                res.set_content(
                    json{{"object", "list"},
                         {"data", data},
                         {"model", bert_->name()},
                         {"usage",
                          {{"prompt_tokens", tokens},
                           {"total_tokens", tokens}}}}
                        .dump(-1, ' ', false, kLossy),
                    "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(
                    make_error(e.what()).dump(-1, ' ', false, kLossy),
                    "application/json");
            }
    };

    // POST /v1/classify, Hugging Face's image-classification shape:
    // {image, top_k?} -> [{label, score}], the highest softmax
    // probabilities first, top_k of them (5 by default). The image is
    // base64 or a data: URL; locus never fetches one. 503 without
    // --image-classifier.
    auto handle_classify = [this](const http::Request& req,
                                  http::ServerResponse& res,
                                  const http::RequestContext& ctx) {
            auth_event("create", "classify", ctx.identity);
            const auto fail = [&](int status, const std::string& msg) {
                res.status = status;
                res.set_content(make_error(msg).dump(-1, ' ', false, kLossy),
                                "application/json");
            };
            if (!classifier_) {
                fail(503, "no image classifier loaded (start locus-server "
                          "with --image-classifier)");
                return;
            }
            try {
                const json body = http::parse_json(req.body);
                if (!body.contains("image")) {
                    throw std::invalid_argument("image is required");
                }
                if (body.contains("model") &&
                    (!body["model"].is_string() ||
                     body["model"].get<std::string>() !=
                         classifier_->name())) {
                    throw std::invalid_argument(
                        "model must be " + classifier_->name());
                }
                std::int64_t k = 5;
                if (body.contains("top_k")) {
                    if (!body["top_k"].is_number_integer()) {
                        throw std::invalid_argument(
                            "top_k must be an integer");
                    }
                    k = body["top_k"].get<std::int64_t>();
                }
                if (k < 1 || k > classifier_->num_classes()) {
                    throw std::invalid_argument(
                        "top_k must be 1 to " +
                        std::to_string(classifier_->num_classes()));
                }
                image::DecodeLimits lim;
                lim.max_bytes = opt_.max_body_bytes;
                const auto img = image::decode_image(
                    image_input_bytes(body["image"], opt_.max_body_bytes),
                    lim);
                std::vector<float> logits;
                {
                    std::lock_guard<std::mutex> lk(classifier_mu_);
                    // The convs on the GPU where it shares host memory
                    // (Apple), as the embedding encoders (#512); they
                    // run on Vulkan only, so not on CUDA.
                    logits = classifier_
                                 ->run(img,
                                       backend::vulkan_imports_host_memory())
                                 .logits;
                }
                json out = json::array();
                for (const auto& [c, p] : image::cnn::top_k(
                         logits, static_cast<std::uint32_t>(k))) {
                    out.push_back(
                        {{"label", classifier_->label(c)}, {"score", p}});
                }
                res.set_content(out.dump(-1, ' ', false, kLossy),
                                "application/json");
            } catch (const std::exception& e) {
                fail(400, e.what());
            }
    };

    // POST /v1/rerank (and /rerank), the shape Jina, Cohere and
    // llama-server share: {query, documents, top_n?, return_documents?}
    // -> {results: [{index, relevance_score[, document]}]}, best first,
    // top_n of them (all by default). The score is the reranker's raw
    // logit, as llama-server gives it. 503 without --rerank-model.
    auto handle_rerank = [this](const http::Request& req,
                                http::ServerResponse& res,
                                const http::RequestContext& ctx) {
            auth_event("create", "rerank", ctx.identity);
            const auto fail = [&](int status, const std::string& msg) {
                res.status = status;
                res.set_content(make_error(msg).dump(-1, ' ', false, kLossy),
                                "application/json");
            };
            if (!rerank_) {
                fail(503, "no reranker loaded (start with --rerank-model)");
                return;
            }
            try {
                const json body = http::parse_json(req.body);
                if (!body.contains("query") || !body["query"].is_string()) {
                    throw std::invalid_argument("query must be a string");
                }
                if (!body.contains("documents") ||
                    !body["documents"].is_array() ||
                    body["documents"].empty()) {
                    throw std::invalid_argument(
                        "documents must be a non-empty array");
                }
                if (body["documents"].size() > opt_.max_rerank_documents) {
                    throw std::invalid_argument(
                        "documents exceeds the " +
                        std::to_string(opt_.max_rerank_documents) +
                        "-document limit");
                }
                // A document is a string or, as Jina sends it, {text}.
                std::vector<std::string> docs;
                for (const auto& d : body["documents"]) {
                    if (d.is_string()) {
                        docs.push_back(d.get<std::string>());
                    } else if (d.is_object() && d.contains("text") &&
                               d["text"].is_string()) {
                        docs.push_back(d["text"].get<std::string>());
                    } else {
                        throw std::invalid_argument(
                            "a document must be a string or {\"text\": "
                            "string}");
                    }
                }
                std::size_t top_n = docs.size();
                if (body.contains("top_n") && !body["top_n"].is_null()) {
                    if (!body["top_n"].is_number_integer() ||
                        body["top_n"].get<std::int64_t>() < 1) {
                        throw std::invalid_argument(
                            "top_n must be a positive integer");
                    }
                    top_n = std::min(
                        top_n, static_cast<std::size_t>(
                                   body["top_n"].get<std::int64_t>()));
                }
                const bool with_docs =
                    body.value("return_documents", false) == true;
                std::size_t tokens = 0;
                std::vector<float> scores;
                {
                    std::lock_guard<std::mutex> lk(rerank_mu_);
                    scores = rerank_->rank(body["query"].get<std::string>(),
                                           docs, backend::best_backend().ops,
                                           &tokens, encoder_gpu_ops());
                }
                std::vector<std::size_t> order(docs.size());
                for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
                std::stable_sort(order.begin(), order.end(),
                                 [&](std::size_t a, std::size_t b) {
                                     return scores[a] > scores[b];
                                 });
                json results = json::array();
                for (std::size_t k = 0; k < top_n; ++k) {
                    const std::size_t i = order[k];
                    json r{{"index", i}, {"relevance_score", scores[i]}};
                    if (with_docs) r["document"] = {{"text", docs[i]}};
                    results.push_back(std::move(r));
                }
                res.set_content(
                    json{{"object", "list"},
                         {"model", rerank_->name()},
                         {"results", results},
                         {"usage",
                          {{"prompt_tokens", tokens},
                           {"total_tokens", tokens}}}}
                        .dump(-1, ' ', false, kLossy),
                    "application/json");
            } catch (const std::exception& e) {
                fail(400, e.what());
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
                const json body = http::parse_json(req.body);
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
                const std::uint64_t seed = read_seed(body);
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
                const json body = http::parse_json(req.body);
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
                p.seed = read_seed(body);
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
                    const auto lk = try_image_lock(image_mu_, res, 60);
                    if (!lk.owns_lock()) return;
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
                set_png_reply(res, png);
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
    // With edit (POST /v1/images/edits, FLUX.1 Kontext): image, a
    // base64 PNG or JPEG or a data: URL of one, is the reference; the
    // output keeps the aspect of size (default: the image's) at the area
    // of size, or 1024 x 1024 pixels without one (and at most that);
    // auto_resize (default true) resizes the reference to Kontext's
    // nearest training resolution.
    auto handle_flux = [this](const http::Request& req,
                              http::ServerResponse& res,
                              const http::RequestContext& ctx, bool edit) {
            try {
                const json body = http::parse_json(req.body);
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
                if (edit) {
                    if (!body.contains("image") || !body["image"].is_string()) {
                        throw std::invalid_argument(
                            "image must be a base64 PNG or JPEG, or a data: "
                            "URL of one");
                    }
                    const std::string im = body["image"].get<std::string>();
                    const auto bytes =
                        im.rfind("data:", 0) == 0
                            ? data_url_bytes(body["image"], opt_.max_body_bytes)
                            : http::base64_decode(im, opt_.max_body_bytes);
                    image::DecodeLimits lim;
                    lim.max_bytes = opt_.max_body_bytes;
                    p.reference = image::decode_image(bytes, lim);
                    p.width = p.reference->w;
                    p.height = p.reference->h;
                    if (body.contains("auto_resize")) {
                        if (!body["auto_resize"].is_boolean()) {
                            throw std::invalid_argument(
                                "auto_resize must be a boolean");
                        }
                        p.auto_resize = body["auto_resize"].get<bool>();
                    }
                }
                if (body.contains("size")) {
                    std::tie(p.width, p.height) =
                        parse_size_wxh(str("size"));
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
                // LoRAs (--lora-dir): the set lives until generate returns.
                image::LoraSet lora_set;
                const auto picks = parse_lora_fields(body);
                if (!picks.empty()) {
                    if (!loras_) {
                        throw std::invalid_argument(
                            "lora: this server has no LoRAs (start it with "
                            "--lora-dir)");
                    }
                    lora_set = loras_->select(picks);
                    p.lora = &lora_set;
                }
                p.seed = read_seed(body);
                // With host-memory import (as for PixArt) the DiT runs
                // each step on the GPU, activations resident
                // (FluxDit::forward_vulkan); CLIP, T5 and the VAE run
                // per op. Without it, the CPU backend. A GPU step holds
                // the Vulkan lock for its whole length, so text
                // generation on the same GPU waits for it.
                const auto& op = backend::vulkan_imports_host_memory()
                                     ? backend::vulkan_image_ops()
                                     : backend::best_backend().ops;
                if (edit) {
                    // FluxKontextPipeline's size rule: the aspect at
                    // max_area pixels, 1024 x 1024 by default; a given
                    // size sets max_area (at most that), so a client can
                    // ask for a smaller, faster image.
                    constexpr std::uint32_t kArea = 1024 * 1024;
                    const std::uint32_t area =
                        body.contains("size")
                            ? std::min(p.width * p.height, kArea)
                            : kArea;
                    std::tie(p.width, p.height) = image::FluxPipeline::
                        kontext_size(p.width, p.height, area);
                }
                image::FluxPipeline::Timing took;
                std::vector<std::uint8_t> rgb;
                {
                    const auto lk = try_image_lock(image_mu_, res, 60);
                    if (!lk.owns_lock()) return;
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
                set_png_reply(res, png);
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
                    const json b = http::parse_json(req.body);
                    named = b.contains("model") && b["model"].is_string() &&
                            b["model"].get<std::string>().rfind("flux", 0) ==
                                0;
                } catch (const std::exception&) {
                    // handle_flux reports the parse error
                }
                if (named || !vq16_ || !janus_) {
                    handle_flux(req, res, ctx, false);
                    return;
                }
            }
            if (pixart_) {
                bool named = false;
                try {
                    const json b = http::parse_json(req.body);
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
                const json body = http::parse_json(req.body);
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
                p.seed = read_seed(body);

                using Clock = std::chrono::steady_clock;
                const auto ids = janus_->prompt_ids(tok_, prompt);
                std::vector<std::int32_t> codes;
                std::vector<float> chw;
                Clock::time_point t0, t1, t2;
                {
                    const auto lk = try_image_lock(image_mu_, res, 30);
                    if (!lk.owns_lock()) return;
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
                set_png_reply(res, png);
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
        OutputGrammar grammar;  // text.format
        try {
            body = http::parse_json(req.body);
            rq = parse_responses_request(body);
            grammar = output_grammar(body, req.body);
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
        const std::uint64_t id = loop_.submit(
            std::move(prompt_ids), max_tokens, sampling, seed,
            output_constraint(std::move(grammar)), {});
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
                               "connection",
                               "api_error")
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
                SseKeepalive ka{
                    std::chrono::milliseconds(opt_.sse_keepalive_ms)};
                for (;;) {
                    auto v = loop_.wait_progress(id, seen, ka.every);
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
                    if (!write_sse_step(sink, ka, payload, loop_, id,
                                        terminal)) {
                        return;
                    }
                    payload.clear();
                }
            });
    };

    // POST /v1/images/edits with Qwen-Image-Edit (i#616): image (a
    // base64 PNG or JPEG, or a data: URL of one) and prompt -> the edited
    // image, as b64_json. Extensions to the OpenAI shape: negative_prompt
    // (true CFG when true_cfg_scale is over 1; default none),
    // true_cfg_scale (default 4), steps (2 to 100, default 50) and seed.
    // size "WxH" sets the output (default: the reference's aspect at
    // about 1024 x 1024 pixels), each side floored to a multiple of 16,
    // up to 2048. auto_resize false (default true) encodes the reference
    // at the output's area when below about 1024 x 1024 (i#670). One
    // image at a time, under image_mu_.
    auto handle_qwen_edit = [this](const http::Request& req,
                                   http::ServerResponse& res,
                                   const http::RequestContext& ctx) {
        try {
            const json body = http::parse_json(req.body);
            auto str = [&](const char* k) -> std::string {
                if (!body[k].is_string()) {
                    throw std::invalid_argument(std::string(k) +
                                                " must be a string");
                }
                return body[k].get<std::string>();
            };
            image::QwenEditPipeline::Request p;
            if (body.contains("prompt")) p.prompt = str("prompt");
            if (p.prompt.empty()) {
                throw std::invalid_argument(
                    "prompt must be a non-empty string");
            }
            if (body.contains("negative_prompt")) {
                p.negative_prompt = str("negative_prompt");
            }
            if (body.contains("n") &&
                (!body["n"].is_number_integer() ||
                 body["n"].get<std::int64_t>() != 1)) {
                throw std::invalid_argument("n must be 1");
            }
            if (body.contains("response_format") &&
                str("response_format") != "b64_json") {
                throw std::invalid_argument(
                    "response_format must be b64_json");
            }
            if (!body.contains("image")) {
                throw std::invalid_argument(
                    "image must be a base64 PNG or JPEG, or a data: URL of "
                    "one");
            }
            image::DecodeLimits lim;
            lim.max_bytes = opt_.max_body_bytes;
            p.reference = image::decode_image(
                image_input_bytes(body["image"], opt_.max_body_bytes), lim);
            if (body.contains("size")) {
                std::tie(p.width, p.height) = parse_size_wxh(str("size"));
                if (p.width < 16 || p.height < 16 || p.width > 2048 ||
                    p.height > 2048) {
                    throw std::invalid_argument(
                        "size sides must be from 16 to 2048");
                }
            }
            if (body.contains("steps")) {
                const auto& s = body["steps"];
                if (!s.is_number_integer() || s.get<std::int64_t>() < 2 ||
                    s.get<std::int64_t>() > 100) {
                    throw std::invalid_argument(
                        "steps must be an integer from 2 to 100");
                }
                p.steps = static_cast<std::uint32_t>(s.get<std::int64_t>());
            }
            if (body.contains("auto_resize")) {
                if (!body["auto_resize"].is_boolean()) {
                    throw std::invalid_argument(
                        "auto_resize must be a boolean");
                }
                p.auto_resize = body["auto_resize"].get<bool>();
            }
            if (body.contains("true_cfg_scale")) {
                const auto& g = body["true_cfg_scale"];
                if (!g.is_number() || !std::isfinite(g.get<double>()) ||
                    g.get<double>() < 0.0 || g.get<double>() > 100.0) {
                    throw std::invalid_argument(
                        "true_cfg_scale must be from 0 to 100");
                }
                p.true_cfg_scale = static_cast<float>(g.get<double>());
            }
            p.seed = read_seed(body);
            // The MMDiT and the VAE run on the GPU where it maps host
            // memory, else on the CPU; the language model always on the
            // CPU.
            const auto& op = backend::vulkan_imports_host_memory()
                                 ? backend::vulkan_image_ops()
                                 : backend::best_backend().ops;
            image::QwenEditPipeline::Timing took;
            std::vector<std::uint8_t> rgb;
            std::uint32_t w = 0, h = 0;
            {
                const auto lk = try_image_lock(image_mu_, res, 60);
                if (!lk.owns_lock()) return;
                rgb = qwen_edit_->edit(p, op, ctx.client_alive, w, h, &took);
            }
            if (rgb.empty()) {
                res.status = 499;  // the client went away
                return;
            }
            const auto png = image::encode_png(rgb, w, h);
            res.set_header("X-Image-Seed", std::to_string(p.seed));
            res.set_header("Server-Timing",
                           "encode;dur=" + std::to_string(took.encode) +
                               ", denoise;dur=" +
                               std::to_string(took.denoise) +
                               ", decode;dur=" + std::to_string(took.decode));
            set_png_reply(res, png);
        } catch (const std::exception& e) {
            set_error(res, e);
        }
    };

    // POST /v1/images/edits: Qwen-Image-Edit (handle_qwen_edit) or FLUX.1
    // Kontext (handle_flux): the image and the prompt -> the edited
    // image, as b64_json. A JSON body with the image as base64 or a
    // data: URL (OpenAI's multipart form is not parsed). 503 unless
    // --image-model loaded one of the two.
    auto handle_edits = [this, handle_flux, handle_qwen_edit](
                            const http::Request& req,
                            http::ServerResponse& res,
                            const http::RequestContext& ctx) {
        auth_event("create", "images.edits", ctx.identity);
        if (qwen_edit_) {
            handle_qwen_edit(req, res, ctx);
            return;
        }
        // A FLUX.1 dev or schnell checkpoint (flux.kontext false) does
        // not edit (i#648).
        if (!flux_ || !flux_->may_edit()) {
            set_unavailable(res,
                            "image edits are not configured (needs "
                            "--image-model with a Qwen-Image-Edit or "
                            "FLUX.1 Kontext pipeline)");
            return;
        }
        handle_flux(req, res, ctx, true);
    };

    // /v1/videos (i#384), as OpenAI's Videos API shapes it: POST starts
    // an LTX-Video job, GET /v1/videos/{id} reports it, and GET
    // /v1/videos/{id}/content returns the MP4, which ffmpeg encodes. 503
    // without --video-model; 501 without a usable ffmpeg.
    const auto video_json = [](const video::VideoJob& j) {
        return json{
            {"id", j.id},
            {"object", "video"},
            {"model", j.model},
            {"status", j.status},
            {"progress", j.progress},
            {"created_at", j.created_at},
            {"completed_at", j.completed_at ? json(j.completed_at) : json()},
            {"expires_at", j.expires_at ? json(j.expires_at) : json()},
            {"size", j.size},
            {"seconds", j.seconds},
            {"prompt", j.prompt},
            {"remixed_from_video_id", nullptr},
            {"error", j.error.empty()
                          ? json()
                          : json{{"code", "video_generation_failed"},
                                 {"message", j.error}}}};
    };
    const auto not_implemented = [](http::ServerResponse& res,
                                    const std::string& message) {
        res.status = 501;
        res.set_content(json{{"error", {{"message", message},
                                        {"type", "server_error"}}}}
                            .dump(-1, ' ', false, kLossy),
                        "application/json");
    };
    const auto videos_on = [this, not_implemented](
                               http::ServerResponse& res) {
        if (!video_jobs_) {
            set_unavailable(res, "video generation is not configured "
                                 "(needs --video-model)");
            return false;
        }
        if (!ffmpeg_) {
            not_implemented(res, "video needs ffmpeg to encode the MP4: "
                                 "install it, or point --ffmpeg at it");
            return false;
        }
        return true;
    };
    const auto video_not_found = [](http::ServerResponse& res,
                                    const std::string& id) {
        res.status = 404;
        res.set_content(make_error("no video " + id).dump(-1, ' ', false,
                                                          kLossy),
                        "application/json");
    };
    // The {id} of /v1/videos/{id}[/content] (route_of checked the shape).
    const auto video_id = [](const std::string& path) {
        std::string id = path.substr(std::string("/v1/videos/").size());
        const std::size_t slash = id.find('/');
        if (slash != std::string::npos) id.resize(slash);
        return id;
    };

    // POST /v1/videos: prompt, size ("WxH", multiples of 32 up to 1280;
    // default 512x320), seconds (a string, as OpenAI sends it, or a
    // number; default 2), input_reference.image_url (a data: URL: the
    // first frame), and locus's fps (default 30), sampling ("stochastic",
    // the default, or "deterministic") and seed. The clip has 8 k + 1
    // frames, k = round(seconds * fps / 8), at most 257.
    auto handle_videos = [this, video_json, videos_on, not_implemented](
                             const http::Request& req,
                             http::ServerResponse& res,
                             const http::RequestContext& ctx) {
        auth_event("create", "videos", ctx.identity);
        if (!videos_on(res)) return;
        try {
            const json body = http::parse_json(req.body);
            if (!body.is_object()) {
                throw std::invalid_argument("the body must be a JSON object");
            }
            image::LtxvPipeline::Request p;
            if (!body.contains("prompt") || !body["prompt"].is_string() ||
                body["prompt"].get<std::string>().empty()) {
                throw std::invalid_argument(
                    "prompt must be a non-empty string");
            }
            p.prompt = body["prompt"].get<std::string>();
            std::string size = "512x320";
            if (body.contains("size")) {
                if (!body["size"].is_string()) {
                    throw std::invalid_argument("size must be a string");
                }
                size = body["size"].get<std::string>();
            }
            std::tie(p.width, p.height) = parse_size_wxh(size);
            double seconds = 2.0;
            if (body.contains("seconds")) {
                const json& s = body["seconds"];
                if (s.is_number()) {
                    seconds = s.get<double>();
                } else if (s.is_string()) {
                    const std::string t = s.get<std::string>();
                    std::size_t used = 0;
                    try {
                        seconds = std::stod(t, &used);
                    } catch (const std::exception&) {
                        used = 0;
                    }
                    if (used == 0 || used != t.size()) {
                        throw std::invalid_argument(
                            "seconds must be a number");
                    }
                } else {
                    throw std::invalid_argument("seconds must be a number");
                }
            }
            if (!std::isfinite(seconds) || seconds <= 0.0) {
                throw std::invalid_argument("seconds must be positive");
            }
            if (body.contains("fps")) {
                if (!body["fps"].is_number() ||
                    !(body["fps"].get<double>() > 0.0 &&
                      body["fps"].get<double>() <= 60.0)) {
                    throw std::invalid_argument(
                        "fps must be a number in (0, 60]");
                }
                p.frame_rate = body["fps"].get<float>();
            }
            const double k = std::round(seconds * p.frame_rate / 8.0);
            constexpr std::uint32_t kMaxK =
                (image::LtxvPipeline::kMaxFrames - 1) / 8;  // 32
            if (k > static_cast<double>(kMaxK)) {
                throw std::invalid_argument(
                    "seconds * fps must be at most 256 frames");
            }
            p.frames = 8 * static_cast<std::uint32_t>(k) + 1;
            if (body.contains("sampling")) {
                const json& s = body["sampling"];
                if (s == "stochastic") {
                    p.stochastic = true;
                } else if (s == "deterministic") {
                    p.stochastic = false;
                } else {
                    throw std::invalid_argument(
                        "sampling must be \"stochastic\" or "
                        "\"deterministic\"");
                }
            }
            p.seed = read_seed(body);
            if (body.contains("input_reference")) {
                const json& ref = body["input_reference"];
                if (!ref.is_object() || ref.contains("file_id") ||
                    !ref.contains("image_url")) {
                    throw std::invalid_argument(
                        "input_reference must be {\"image_url\": a data: "
                        "URL} (file ids are not supported)");
                }
                if (!ltxv_->can_condition()) {
                    not_implemented(res,
                                    "image to video needs the LTX VAE with "
                                    "its encoder (reconvert it with "
                                    "convert-ltxv.py)");
                    return;
                }
                image::DecodeLimits lim;
                lim.max_bytes = opt_.max_body_bytes;
                p.image = image::decode_image(
                    data_url_bytes(ref["image_url"], opt_.max_body_bytes),
                    lim);
            }
            image::LtxvPipeline::check(p);

            video::VideoJob meta;
            meta.model = "ltx-video";
            meta.prompt = p.prompt;
            meta.size = size;
            char secs[32];
            std::snprintf(secs, sizeof secs, "%g",
                          (p.frames - 1) / static_cast<double>(p.frame_rate));
            meta.seconds = secs;
            // The job: denoise and decode under image_mu_ (so a clip and
            // an image never run together), then ffmpeg. alive() is
            // polled before each of the 8 DiT steps and the decode, which
            // gives the progress.
            const auto job = video_jobs_->submit(
                meta,
                [this, p](const std::string& out,
                          const std::function<void(int)>& progress,
                          const std::function<bool()>& alive) {
                    std::vector<std::vector<std::uint8_t>> frames;
                    {
                        std::lock_guard<std::mutex> lk(image_mu_);
                        int polls = 0;
                        const auto& op =
                            backend::vulkan_imports_host_memory()
                                ? backend::vulkan_image_ops()
                                : backend::best_backend().ops;
                        // The decode takes about three times the 8 DiT
                        // steps (M2 Pro, 512x320x65: 170 s vs 60 s), so
                        // the steps span 0 to 30 and the decode 30 to 90.
                        frames = ltxv_->generate(p, op, [&] {
                            progress(30 * polls++ / 8);
                            return alive();
                        });
                    }
                    if (frames.empty()) {
                        throw std::runtime_error("cancelled");
                    }
                    progress(90);
                    video::encode_mp4(*ffmpeg_, frames, p.width, p.height,
                                      p.frame_rate, out, alive);
                });
            if (!job) {
                set_unavailable(res, "too many video jobs; retry later", 60);
                return;
            }
            res.set_header("X-Video-Seed", std::to_string(p.seed));
            res.set_content(video_json(*job).dump(-1, ' ', false, kLossy),
                            "application/json");
        } catch (const std::exception& e) {
            set_error(res, e);
        }
    };

    auto handle_video_by_id = [this, video_json, videos_on, video_not_found,
                               video_id](const http::Request& req,
                                         http::ServerResponse& res) {
        if (!videos_on(res)) return;
        const std::string id = video_id(req.path);
        const auto job = video_jobs_->get(id);
        if (!job) {
            video_not_found(res, id);
            return;
        }
        res.set_content(video_json(*job).dump(-1, ' ', false, kLossy),
                        "application/json");
    };

    // The MP4 of a completed job; 409 while it is queued or running, or
    // when it failed.
    auto handle_video_content = [this, videos_on, video_not_found,
                                 video_id](const http::Request& req,
                                           http::ServerResponse& res) {
        if (!videos_on(res)) return;
        const std::string id = video_id(req.path);
        const auto job = video_jobs_->get(id);
        if (!job) {
            video_not_found(res, id);
            return;
        }
        if (job->status != "completed") {
            res.status = 409;
            res.set_content(make_error("video " + id + " is " + job->status +
                                       ", not completed")
                                .dump(-1, ' ', false, kLossy),
                            "application/json");
            return;
        }
        // The file can expire between get() and the open: then a 404.
        std::ifstream f(job->file, std::ios::binary);
        if (!f.is_open()) {
            video_not_found(res, id);
            return;
        }
        std::string mp4((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
        if (!f.good() && !f.eof()) {
            set_error(res, std::runtime_error("cannot read the video file"));
            return;
        }
        res.set_content(std::move(mp4), "video/mp4");
    };

    // One dispatch handler over all routes (route_of is the single home
    // for routing, shared with the auth head stage below).
    route_handler_ = [this, handle_health, handle_models,
                      handle_model_by_id, handle_metrics, handle,
                      handle_messages, handle_responses, handle_embeddings,
                      handle_rerank, handle_classify, handle_audio,
                      handle_images, handle_edits, handle_videos,
                      handle_video_by_id, handle_video_content](
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
            case Route::kRerank:      handle_rerank(req, res, ctx); break;
            case Route::kClassify:    handle_classify(req, res, ctx); break;
            case Route::kAudioSpeech: handle_audio(req, res, ctx); break;
            case Route::kImages:      handle_images(req, res, ctx); break;
            case Route::kImageEdits:  handle_edits(req, res, ctx); break;
            case Route::kVideos:      handle_videos(req, res, ctx); break;
            case Route::kVideoById:   handle_video_by_id(req, res); break;
            case Route::kVideoContent:
                handle_video_content(req, res);
                break;
            case Route::kUnknown:
                res.status = 404;  // an unknown route: 404, empty body
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
