#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/grammar.hpp"
#include "locus/model/transformer.hpp"
#include "locus/pipeline/net.hpp"
#include "locus/server/server.hpp"
#include "locus/tok/tokenizer.hpp"
#include "httplib.h"
#include "json.hpp"
#include "png_parse.hpp"

using nlohmann::json;

namespace {

std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

/** Server bound to an ephemeral port, serving on a thread. */
struct TestServer {
    TestServer(const locus::model::TransformerModel& m,
               const locus::tok::SpmTokenizer& tok,
               locus::server::OpenAiServer::Options opt = {})
        : server(m, tok, std::move(opt)) {
        port = server.bind_any_port("127.0.0.1");
        REQUIRE(port > 0);
        thread = std::thread([this] {
            server.listen_after_bind();
        });
    }
    ~TestServer() {
        server.stop();
        thread.join();
    }

    locus::server::OpenAiServer server;
    int port = -1;
    std::thread thread;
};

}  // namespace

TEST_CASE("server stop() unblocks the listener after requests",
          "[server][e2e]") {
    // Regression: httplib's non-Windows default idle interval is 0, so
    // the accept loop blocks directly in accept(); on macOS/BSD stop()
    // closing the listen socket does not wake a thread already parked
    // in accept(), so ~TestServer (server.stop()+thread.join()) hangs.
    // A small server idle interval makes the loop poll with
    // select_read so stop() is prompt. This test would hang (not fail)
    // on regression, which the CI wall-clock timeout catches.
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    TestServer ts(model, tok);
    httplib::Client client("127.0.0.1", ts.port);
    const json req{{"prompt", "hello world"}, {"max_tokens", 2}};
    auto a = client.Post("/v1/completions", req.dump(),
                         "application/json");
    REQUIRE(a);
    REQUIRE(a->status == 200);
    // ~TestServer runs here; it must not hang.
}

TEST_CASE("openai endpoints serve completions", "[server][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    TestServer ts(model, tok);
    httplib::Client client("127.0.0.1", ts.port);

    SECTION("health") {
        auto res = client.Get("/health");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        REQUIRE(json::parse(res->body)["status"] == "ok");
    }

    SECTION("model listing") {
        auto res = client.Get("/v1/models");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto body = json::parse(res->body);
        REQUIRE(body["object"] == "list");
        REQUIRE(body["data"].size() == 1);
        const std::string id = body["data"][0]["id"];
        REQUIRE(body["data"][0]["object"] == "model");

        auto one = client.Get("/v1/models/" + id);
        REQUIRE(one);
        REQUIRE(one->status == 200);
        REQUIRE(json::parse(one->body)["id"] == id);

        auto miss = client.Get("/v1/models/nope");
        REQUIRE(miss);
        REQUIRE(miss->status == 404);
    }

    SECTION("metrics count requests and tokens") {
        json req{{"prompt", "Once upon a time"}, {"max_tokens", 4}};
        auto gen = client.Post("/v1/completions", req.dump(),
                               "application/json");
        REQUIRE(gen);
        REQUIRE(gen->status == 200);

        auto res = client.Get("/metrics");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        const std::string& b = res->body;
        REQUIRE(b.find("locus_requests_total") != std::string::npos);
        REQUIRE(b.find("locus_completion_tokens_total") !=
                std::string::npos);
        REQUIRE(b.find("locus_kv_blocks_total") !=
                std::string::npos);
        // At least the request we just issued is counted.
        REQUIRE(b.find("locus_requests_total 0\n") ==
                std::string::npos);
    }

    SECTION("completions, non-streaming") {
        json req{{"prompt", "Once upon a time"},
                 {"max_tokens", 16}};
        auto res = client.Post("/v1/completions", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto body = json::parse(res->body);
        REQUIRE(body["object"] == "text_completion");
        REQUIRE(body["choices"].size() == 1);
        const std::string text = body["choices"][0]["text"];
        REQUIRE(text.find("little girl") != std::string::npos);
        REQUIRE(body["choices"][0]["finish_reason"] == "length");
        REQUIRE(body["usage"]["completion_tokens"] == 16);
        REQUIRE(body["usage"]["total_tokens"] ==
                body["usage"]["prompt_tokens"].get<int>() + 16);
        // No logprobs unless requested.
        REQUIRE(!body["choices"][0].contains("logprobs"));
    }

    SECTION("completions logprobs") {
        json req{{"prompt", "Once upon a time"},
                 {"max_tokens", 5},
                 {"logprobs", 3}};
        auto res = client.Post("/v1/completions", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto lp = json::parse(res->body)["choices"][0]["logprobs"];
        REQUIRE(lp["tokens"].size() == 5);
        REQUIRE(lp["token_logprobs"].size() == 5);
        REQUIRE(lp["top_logprobs"].size() == 5);
        // Each step lists 3 alternatives; all logprobs <= 0.
        REQUIRE(lp["top_logprobs"][0].size() == 3);
        for (const auto& v : lp["token_logprobs"]) {
            REQUIRE(v.get<double>() <= 0.0);
        }
    }

    SECTION("chat logprobs") {
        json req{{"messages",
                  json::array({{{"role", "user"},
                                {"content", "Once upon a time"}}})},
                 {"max_tokens", 4},
                 {"logprobs", true},
                 {"top_logprobs", 2}};
        auto res = client.Post("/v1/chat/completions", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto content =
            json::parse(res->body)["choices"][0]["logprobs"]
                                  ["content"];
        REQUIRE(content.size() == 4);
        REQUIRE(content[0].contains("token"));
        REQUIRE(content[0]["top_logprobs"].size() == 2);
    }

    SECTION("chat completions, non-streaming") {
        json req{{"messages",
                  json::array(
                      {{{"role", "user"},
                        {"content", "Once upon a time"}}})},
                 {"max_tokens", 12}};
        auto res = client.Post("/v1/chat/completions", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto body = json::parse(res->body);
        REQUIRE(body["object"] == "chat.completion");
        REQUIRE(body["choices"][0]["message"]["role"] ==
                "assistant");
        REQUIRE(!body["choices"][0]["message"]["content"]
                     .get<std::string>()
                     .empty());
    }

    SECTION("chat completions with tools is accepted") {
        // The toy model won't emit a valid tool call, so this just
        // confirms the tools plumbing (system injection + response
        // branch) doesn't break a normal request.
        json req{
            {"messages",
             json::array({{{"role", "user"},
                           {"content", "Once upon a time"}}})},
            {"max_tokens", 8},
            {"tools",
             json::array(
                 {{{"type", "function"},
                   {"function",
                    {{"name", "get_weather"},
                     {"description", "weather"},
                     {"parameters", {{"type", "object"}}}}}}})}};
        auto res = client.Post("/v1/chat/completions", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto body = json::parse(res->body);
        REQUIRE(body["choices"][0].contains("message"));
    }

    SECTION("chat tools + stream buffers and emits a final chunk") {
        // With tools, the stream must BUFFER (a tool call can only be
        // recognized from the whole completion) and emit one terminal
        // chunk carrying finish_reason -- so an agent never sees raw
        // tool JSON as incremental content. The toy model won't emit
        // a real call, so finish_reason is stop/length and the text
        // rides in the final delta.content.
        json req{
            {"messages",
             json::array({{{"role", "user"},
                           {"content", "Once upon a time"}}})},
            {"max_tokens", 8},
            {"stream", true},
            {"tools",
             json::array(
                 {{{"type", "function"},
                   {"function",
                    {{"name", "get_weather"},
                     {"description", "weather"},
                     {"parameters", {{"type", "object"}}}}}}})}};
        auto res = client.Post("/v1/chat/completions", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        const std::string& sse = res->body;
        REQUIRE(sse.find("data: [DONE]\n\n") != std::string::npos);

        int content_chunks = 0, finish_chunks = 0;
        std::size_t at = 0;
        const std::string dp = "data: ";
        while ((at = sse.find(dp, at)) != std::string::npos) {
            const std::size_t nl = sse.find("\n\n", at);
            const std::string line =
                sse.substr(at + dp.size(),
                           nl - at - dp.size());
            at = nl + 2;
            if (line == "[DONE]") {
                continue;
            }
            auto chunk = json::parse(line);
            const auto& choice = chunk["choices"][0];
            if (choice.contains("finish_reason") &&
                !choice["finish_reason"].is_null()) {
                ++finish_chunks;
            }
            if (choice.contains("delta") &&
                choice["delta"].contains("content")) {
                ++content_chunks;
            }
        }
        // Buffered: content is not streamed per token -- it arrives in
        // the single terminal chunk, which also carries finish_reason.
        REQUIRE(content_chunks == 1);
        REQUIRE(finish_chunks == 1);
    }

    SECTION("response_format json_object constrains to valid JSON") {
        json req{
            {"messages",
             json::array({{{"role", "user"},
                           {"content", "Give me an object"}}})},
            {"max_tokens", 24},
            {"response_format", {{"type", "json_object"}}}};
        auto res = client.Post("/v1/chat/completions", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        const std::string content = json::parse(
            res->body)["choices"][0]["message"]["content"];
        REQUIRE(!content.empty());
        // Every byte must keep the output a valid JSON prefix.
        locus::model::JsonGrammar g;
        for (unsigned char c : content) {
            INFO("content: " << content);
            REQUIRE(g.feed(c));
        }
    }

    SECTION("anthropic messages, non-streaming") {
        json req{{"system", "You are a storyteller."},
                 {"messages",
                  json::array(
                      {{{"role", "user"},
                        {"content", "Once upon a time"}}})},
                 {"max_tokens", 12}};
        auto res = client.Post("/v1/messages", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto body = json::parse(res->body);
        REQUIRE(body["type"] == "message");
        REQUIRE(body["role"] == "assistant");
        REQUIRE(body["content"][0]["type"] == "text");
        REQUIRE(!body["content"][0]["text"]
                     .get<std::string>()
                     .empty());
        REQUIRE((body["stop_reason"] == "end_turn" ||
                 body["stop_reason"] == "max_tokens"));
        REQUIRE(body["usage"]["input_tokens"].get<int>() > 0);
    }

    SECTION("anthropic messages, content blocks") {
        // Content as an array of text blocks must also work.
        json req{{"messages",
                  json::array({{{"role", "user"},
                                {"content",
                                 json::array({{{"type", "text"},
                                               {"text",
                                                "Once upon"}}})}}})},
                 {"max_tokens", 8}};
        auto res = client.Post("/v1/messages", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        REQUIRE(json::parse(res->body)["type"] == "message");
    }

    SECTION("anthropic messages, streaming SSE") {
        json req{{"messages",
                  json::array({{{"role", "user"},
                                {"content", "Once upon a time"}}})},
                 {"max_tokens", 8},
                 {"stream", true}};
        auto res = client.Post("/v1/messages", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        REQUIRE(res->get_header_value("Content-Type") ==
                "text/event-stream");
        const std::string& sse = res->body;
        REQUIRE(sse.find("event: message_start") == 0);
        REQUIRE(sse.find("event: content_block_start") !=
                std::string::npos);
        REQUIRE(sse.find("event: content_block_delta") !=
                std::string::npos);
        REQUIRE(sse.find("\"type\":\"text_delta\"") !=
                std::string::npos);
        REQUIRE(sse.find("event: message_delta") !=
                std::string::npos);
        REQUIRE(sse.find("event: message_stop") !=
                std::string::npos);
        // Reassemble the streamed text_delta pieces (parse each
        // SSE data line as JSON -- key order is not guaranteed).
        std::string text;
        std::size_t at = 0;
        const std::string dp = "data: ";
        while ((at = sse.find(dp, at)) != std::string::npos) {
            at += dp.size();
            const std::size_t end = sse.find("\n\n", at);
            const auto j = json::parse(sse.substr(at, end - at));
            if (j.value("type", "") == "content_block_delta") {
                text += j["delta"]["text"].get<std::string>();
            }
            at = end;
        }
        REQUIRE(!text.empty());
    }

    SECTION("completions, streaming SSE") {
        json req{{"prompt", "Once upon a time"},
                 {"max_tokens", 8},
                 {"stream", true}};
        std::string sse;
        auto res = client.Post(
            "/v1/completions", req.dump(), "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        REQUIRE(res->get_header_value("Content-Type") ==
                "text/event-stream");
        sse = res->body;
        REQUIRE(sse.find("data: ") == 0);
        REQUIRE(sse.find("data: [DONE]\n\n") != std::string::npos);
        // Concatenating the streamed text chunks reproduces the
        // non-streaming completion.
        std::string text;
        std::size_t at = 0;
        while ((at = sse.find("data: ", at)) !=
               std::string::npos) {
            at += 6;
            const std::size_t end = sse.find("\n\n", at);
            const std::string part = sse.substr(at, end - at);
            if (part == "[DONE]") {
                break;
            }
            text += json::parse(part)["choices"][0]["text"]
                        .get<std::string>();
        }
        REQUIRE(text.find(", there was") != std::string::npos);
    }

    SECTION("bad requests get 400") {
        auto res = client.Post("/v1/completions", "{not json",
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 400);

        json req{{"prompt", 42}};
        res = client.Post("/v1/completions", req.dump(),
                          "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 400);
        // OpenAI flat envelope (shape A).
        auto err_a = json::parse(res->body);
        REQUIRE(err_a["error"]["type"] == "invalid_request_error");

        // /v1/messages keeps the Anthropic NESTED envelope (shape C): an
        // outer {"type":"error"} the other routes do not have. Pinned
        // because the port is exactly when it would get flattened, and
        // nothing else asserts it.
        auto m = client.Post("/v1/messages", "{not json",
                             "application/json");
        REQUIRE(m);
        REQUIRE(m->status == 400);
        auto err_c = json::parse(m->body);
        REQUIRE(err_c["type"] == "error");
        REQUIRE(err_c["error"]["type"] == "invalid_request_error");
    }

    SECTION("concurrent clients are served") {
        auto worker = [&](std::string* out) {
            httplib::Client c("127.0.0.1", ts.port);
            c.set_read_timeout(60, 0);
            json req{{"prompt", "Once upon a time"},
                     {"max_tokens", 12}};
            auto res = c.Post("/v1/completions", req.dump(),
                              "application/json");
            if (res && res->status == 200) {
                *out = json::parse(res->body)["choices"][0]
                                 ["text"]
                                     .get<std::string>();
            }
        };
        std::string t1, t2, t3;
        std::thread a(worker, &t1), b(worker, &t2),
            c(worker, &t3);
        a.join();
        b.join();
        c.join();
        REQUIRE(t1 == t2);
        REQUIRE(t2 == t3);
        REQUIRE(!t1.empty());
    }
}

TEST_CASE("embeddings endpoint returns normalized vectors",
          "[server][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    // Embeddings need a CPU/CUDA backend (the Vulkan full-forward
    // does not surface the hidden state); force the scalar backend.
    model.use_backend(*locus::backend::find_backend("scalar"));
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    TestServer ts(model, tok);
    httplib::Client client("127.0.0.1", ts.port);

    SECTION("single input, unit-normalized, right dimension") {
        json req{{"input", "Once upon a time"}};
        auto res = client.Post("/v1/embeddings", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto body = json::parse(res->body);
        REQUIRE(body["object"] == "list");
        auto emb = body["data"][0]["embedding"];
        REQUIRE(emb.size() == model.hparams().n_embd);
        double n = 0.0;
        for (double v : emb) {
            n += v * v;
        }
        REQUIRE(n == Catch::Approx(1.0).margin(1e-4));
    }

    SECTION("batch input and determinism") {
        json req{{"input", json::array({"hello", "hello"})}};
        auto res = client.Post("/v1/embeddings", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        auto data = json::parse(res->body)["data"];
        REQUIRE(data.size() == 2);
        REQUIRE(data[0]["embedding"] == data[1]["embedding"]);
    }
}

TEST_CASE("auth gates protected routes via a helper", "[server][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

    locus::server::OpenAiServer::Options opt;
    opt.auth_helper_argv = {LOCUS_AUTH_HELPER_BIN, "sk-secret"};
    locus::server::OpenAiServer server(model, tok, opt);
    const int port = server.bind_any_port("127.0.0.1");
    REQUIRE(port > 0);
    std::thread th([&] { server.listen_after_bind(); });
    httplib::Client client("127.0.0.1", port);

    const json req{{"prompt", "Once upon a time"}, {"max_tokens", 4}};
    const std::string ct = "application/json";

    // No credential -> 401. Pin the auth-error envelope (shape B) so a
    // later refactor cannot silently flatten it to the generic shape --
    // SDKs branch on error.type to decide whether to re-auth.
    auto r1 = client.Post("/v1/completions", req.dump(), ct);
    REQUIRE(r1);
    REQUIRE(r1->status == 401);
    REQUIRE(json::parse(r1->body)["error"]["type"] ==
            "authentication_error");

    // Wrong credential -> 401.
    auto r2 = client.Post("/v1/completions",
                          {{"Authorization", "Bearer nope"}},
                          req.dump(), ct);
    REQUIRE(r2);
    REQUIRE(r2->status == 401);

    // Valid credential -> 200 (Bearer).
    auto r3 = client.Post("/v1/completions",
                          {{"Authorization", "Bearer sk-secret"}},
                          req.dump(), ct);
    REQUIRE(r3);
    REQUIRE(r3->status == 200);

    // Valid credential via x-api-key -> 200.
    auto r4 = client.Post("/v1/completions",
                          {{"x-api-key", "sk-secret"}}, req.dump(),
                          ct);
    REQUIRE(r4);
    REQUIRE(r4->status == 200);

    // /health is not protected.
    auto h = client.Get("/health");
    REQUIRE(h);
    REQUIRE(h->status == 200);

    // Info routes are gated too (#60-S3): they disclose the model id
    // and traffic/capacity counters, so with auth on they need a
    // credential -- 401 without, 200 with.
    for (const char* path : {"/v1/models", "/v1/models/locus",
                             "/metrics"}) {
        auto no = client.Get(path);
        REQUIRE(no);
        REQUIRE(no->status == 401);
        auto ok = client.Get(
            path, {{"Authorization", "Bearer sk-secret"}});
        REQUIRE(ok);
        REQUIRE(ok->status == 200);
    }

    server.stop();
    th.join();
}

TEST_CASE("metrics path is configurable", "[server][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

    locus::server::OpenAiServer::Options opt;
    opt.metrics_path = "/internal/metrics";
    locus::server::OpenAiServer server(model, tok, opt);
    const int port = server.bind_any_port("127.0.0.1");
    REQUIRE(port > 0);
    std::thread th([&] { server.listen_after_bind(); });
    httplib::Client client("127.0.0.1", port);

    auto m = client.Get("/internal/metrics");
    REQUIRE(m);
    REQUIRE(m->status == 200);
    REQUIRE(m->body.find("locus_requests_total") != std::string::npos);
    // The default path is no longer served.
    auto d = client.Get("/metrics");
    REQUIRE(d);
    REQUIRE(d->status == 404);

    server.stop();
    th.join();
}

TEST_CASE("request-edge size limits reject before work",
          "[server][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

    locus::server::OpenAiServer::Options opt;
    opt.max_body_bytes = 2048;        // tiny body cap
    opt.max_embeddings_inputs = 4;    // tiny embeddings cap
    TestServer ts(model, tok, opt);
    httplib::Client client("127.0.0.1", ts.port);

    SECTION("oversized body is refused before the handler") {
        // A body past the cap must not be buffered/parsed; httplib
        // answers 413 without running the route (or authorize()).
        const std::string big(8192, 'x');
        const json body = {{"prompt", big}, {"max_tokens", 1}};
        auto res = client.Post("/v1/completions", body.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 413);
    }

    SECTION("embeddings input array is bounded") {
        json ok = {{"input", json::array({"a", "b"})}};
        auto r1 = client.Post("/v1/embeddings", ok.dump(),
                              "application/json");
        REQUIRE(r1);
        REQUIRE(r1->status == 200);

        json over = {
            {"input", json::array({"a", "b", "c", "d", "e"})}};
        auto r2 = client.Post("/v1/embeddings", over.dump(),
                              "application/json");
        REQUIRE(r2);
        REQUIRE(r2->status == 400);
    }
}

TEST_CASE("prompt caching reports cache-hit usage fields",
          "[server][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    // Default Options: prompt caching is on.
    TestServer ts(model, tok);
    httplib::Client client("127.0.0.1", ts.port);

    // Long enough to span several KV blocks so a cacheable
    // block-aligned prefix exists (and survives whatever the block
    // size is).
    std::string prompt;
    for (int i = 0; i < 40; ++i) {
        prompt += "the quick brown fox jumps over the lazy dog. ";
    }

    SECTION("OpenAI: cached_tokens is a subset of prompt_tokens") {
        const json req{{"prompt", prompt}, {"max_tokens", 2}};
        auto a = client.Post("/v1/completions", req.dump(),
                             "application/json");
        REQUIRE(a);
        REQUIRE(a->status == 200);
        auto ua = json::parse(a->body)["usage"];
        // First request: nothing cached yet.
        REQUIRE(ua["prompt_tokens_details"]["cached_tokens"] == 0);

        auto b = client.Post("/v1/completions", req.dump(),
                             "application/json");
        REQUIRE(b);
        REQUIRE(b->status == 200);
        auto ub = json::parse(b->body)["usage"];
        const int cached =
            ub["prompt_tokens_details"]["cached_tokens"].get<int>();
        REQUIRE(cached > 0);  // second request hit the cache
        // cached_tokens is a subset of the (unchanged) prompt_tokens.
        REQUIRE(cached < ub["prompt_tokens"].get<int>());
        REQUIRE(ub["prompt_tokens"] == ua["prompt_tokens"]);
    }

    SECTION("Anthropic: input_tokens splits into cache fields") {
        const json req{
            {"messages",
             json::array({{{"role", "user"}, {"content", prompt}}})},
            {"max_tokens", 2}};
        auto a = client.Post("/v1/messages", req.dump(),
                             "application/json");
        REQUIRE(a);
        REQUIRE(a->status == 200);
        auto ua = json::parse(a->body)["usage"];
        REQUIRE(ua["cache_read_input_tokens"] == 0);
        const int full = ua["input_tokens"].get<int>() +
                         ua["cache_creation_input_tokens"].get<int>();

        auto b = client.Post("/v1/messages", req.dump(),
                             "application/json");
        REQUIRE(b);
        REQUIRE(b->status == 200);
        auto ub = json::parse(b->body)["usage"];
        const int read = ub["cache_read_input_tokens"].get<int>();
        REQUIRE(read > 0);  // second request read from the cache
        // input + cache_read + cache_creation == the full prompt, and
        // input_tokens is now the uncached remainder (smaller).
        const int sum = ub["input_tokens"].get<int>() + read +
                        ub["cache_creation_input_tokens"].get<int>();
        REQUIRE(sum == full);
        REQUIRE(ub["input_tokens"].get<int>() < full);
    }
}

namespace {

/** Minimal HTTP/1.1 POST with a JSON body and Connection: close, so the
 *  server frames the request by Content-Length and closes after
 *  responding (the client then reads to EOF). */
std::string make_post(const std::string& path, const std::string& body) {
    return "POST " + path + " HTTP/1.1\r\n" +
           "Host: 127.0.0.1\r\n" +
           "Content-Type: application/json\r\n" +
           "Content-Length: " + std::to_string(body.size()) + "\r\n" +
           "Connection: close\r\n\r\n" + body;
}

bool send_all(int fd, const std::string& s) {
    std::size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = ::send(fd, s.data() + off, s.size() - off, 0);
        if (n <= 0) {
            return false;
        }
        off += static_cast<std::size_t>(n);
    }
    return true;
}

/** Reads the whole response to EOF (Connection: close). */
std::string recv_to_eof(int fd) {
    std::string out;
    char buf[4096];
    for (;;) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            break;
        }
        out.append(buf, static_cast<std::size_t>(n));
    }
    return out;
}

/** Parses the status code from an HTTP response's first line. */
int status_of(const std::string& resp) {
    // "HTTP/1.1 <code> <reason>\r\n"
    const auto sp = resp.find(' ');
    if (sp == std::string::npos) {
        return -1;
    }
    return std::atoi(resp.c_str() + sp + 1);
}

/** The JSON body after the blank line, or "" if none. */
std::string body_of(const std::string& resp) {
    const auto hdr_end = resp.find("\r\n\r\n");
    return hdr_end == std::string::npos
               ? std::string()
               : resp.substr(hdr_end + 4);
}

}  // namespace

// i#83 (follow-up to i#81): the non-streaming cancelled path must emit
// 499 Client Closed Request plus the route's error envelope, not a bare
// empty 200 (which a half-closed client would read as a valid empty
// completion). The loop-layer kCancelled wiring is covered elsewhere;
// this pins the ROUTE-level 499 emission and the per-route body shape,
// end to end through the real socket, peer_half_closed, and client_alive.
//
// Deterministic trigger (no sleep, no engine race): send the COMPLETE
// request, then shutdown(SHUT_WR) with the read side open. The body is
// buffered before dispatch, so the handler's first client_alive poll
// sees peer_half_closed's EOF and cancels before the first decode step.
// The negative control -- same request, no half-close -- returns 200
// with real content, proving the half-close is what causes the 499.
TEST_CASE("server emits 499 + error envelope on a cancelled request",
          "[server][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    TestServer ts(model, tok);

    // 64 new tokens so the request is still generating (not already
    // terminal) when the handler's first liveness poll runs.
    const json completions_body{{"prompt", "Once upon a time"},
                                {"max_tokens", 64}};
    const json chat_body{
        {"messages", json::array({{{"role", "user"},
                                   {"content", "Once upon a time"}}})},
        {"max_tokens", 64}};

    // Sends `body` to `path`, half-closes the write side, and returns the
    // full response read to EOF.
    auto cancel_via_halfclose = [&](const std::string& path,
                                    const json& body) {
        const int fd =
            locus::pipeline::connect_to("127.0.0.1", ts.port, 2000);
        REQUIRE(fd >= 0);
        // Bound the read so a hung handler fails fast instead of wedging
        // the whole suite (which runs with no external timeout). 5000 ms
        // is a wide margin over the slowest LEGITIMATE wait here -- the
        // negative control's full 64-token generation, tens of ms even on
        // the slowest host -- while still well under any CI budget; it is
        // a hang bound, not an idle-gap heuristic like drain()'s 300 ms.
        // On a timeout recv returns -1/EAGAIN, recv_to_eof returns what it
        // has (nothing), and status_of("") is -1, so a hang surfaces as a
        // legible `-1 == 499` ("no response arrived"), not a wrong status.
        locus::pipeline::set_recv_timeout(fd, 5000);
        REQUIRE(send_all(fd, make_post(path, body.dump())));
        // The whole request is now sent; closing just the write side
        // delivers a FIN after it, which peer_half_closed reads as EOF.
        REQUIRE(::shutdown(fd, SHUT_WR) == 0);
        const std::string resp = recv_to_eof(fd);
        ::close(fd);
        return resp;
    };

    SECTION("/v1/completions -> 499, OpenAI flat error envelope") {
        const std::string resp =
            cancel_via_halfclose("/v1/completions", completions_body);
        REQUIRE(status_of(resp) == 499);
        const auto b = json::parse(body_of(resp));
        REQUIRE(b.contains("error"));
        // i#75 (error-body unification) tracks that this type is
        // misdescriptive for a cancellation -- the request was well
        // formed, the client left -- yet make_error hardcodes
        // "invalid_request_error". Pinned as-is here; do NOT change the
        // type in i#83 (client-visible; belongs to i#75's gate).
        REQUIRE(b["error"]["type"] == "invalid_request_error");
        REQUIRE(b["error"]["message"].get<std::string>().find(
                    "cancelled") != std::string::npos);
    }

    SECTION("/v1/chat/completions -> 499, OpenAI flat error envelope") {
        const std::string resp =
            cancel_via_halfclose("/v1/chat/completions", chat_body);
        REQUIRE(status_of(resp) == 499);
        const auto b = json::parse(body_of(resp));
        REQUIRE(b.contains("error"));
        REQUIRE(b["error"]["type"] == "invalid_request_error");  // see i#75
    }

    SECTION("/v1/messages -> 499, Anthropic nested error envelope") {
        const std::string resp =
            cancel_via_halfclose("/v1/messages", chat_body);
        REQUIRE(status_of(resp) == 499);
        const auto b = json::parse(body_of(resp));
        REQUIRE(b["type"] == "error");
        REQUIRE(b["error"]["type"] == "api_error");
        REQUIRE(b["error"]["message"].get<std::string>().find(
                    "cancelled") != std::string::npos);
    }

    SECTION("negative control: no half-close -> 200 with real content") {
        const int fd =
            locus::pipeline::connect_to("127.0.0.1", ts.port, 2000);
        REQUIRE(fd >= 0);
        locus::pipeline::set_recv_timeout(fd, 5000);  // hang bound; see above
        REQUIRE(send_all(fd, make_post("/v1/completions",
                                       completions_body.dump())));
        // No shutdown: the client stays connected, so client_alive holds
        // and the request runs to completion.
        const std::string resp = recv_to_eof(fd);
        ::close(fd);
        REQUIRE(status_of(resp) == 200);
        const auto b = json::parse(body_of(resp));
        REQUIRE(!b["choices"][0]["text"].get<std::string>().empty());
    }
}

namespace {

/** Minimal HTTP/1.1 GET with Connection: close (for /health, /metrics). */
std::string make_get(const std::string& path) {
    return "GET " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\n" +
           "Connection: close\r\n\r\n";
}

/** Reads a Prometheus gauge/counter value line "name <number>" from a
 *  /metrics body. Matches only a value line (name at column 0 followed
 *  by a space and a digit), so it never picks up the "# HELP name" or
 *  "# TYPE name" lines. @returns the value, or -1 if absent. */
long metric_value(const std::string& body, const std::string& name) {
    const std::string key = "\n" + name + " ";
    // Allow a match at the very start of the body too.
    std::size_t pos =
        body.compare(0, name.size() + 1, name + " ") == 0 ? 0
                                                           : body.find(key);
    if (pos == std::string::npos) {
        return -1;
    }
    const std::size_t num = (pos == 0 ? 0 : pos + 1) + name.size() + 1;
    if (num >= body.size() || body[num] < '0' || body[num] > '9') {
        return -1;
    }
    return std::atol(body.c_str() + num);
}

/** GETs /metrics and returns the named metric's value, or -1. */
long scrape_metric(int port, const std::string& name) {
    const int fd = locus::pipeline::connect_to("127.0.0.1", port, 2000);
    if (fd < 0) {
        return -1;
    }
    locus::pipeline::set_recv_timeout(fd, 3000);
    const bool sent = send_all(fd, make_get("/metrics"));
    const std::string resp = sent ? recv_to_eof(fd) : std::string();
    ::close(fd);
    return metric_value(body_of(resp), name);
}

}  // namespace

// Follow-up to i#81 (cancel on disconnect) / i#85 + i#88 (fast-path
// acceptor): the STREAMING cancel path is reached by the Sink write
// FAILING, not by the liveness probe (server.cpp: the cancel lives only
// in the !stream branch; the stream path cancels when sink.write returns
// false). Under CONCURRENT mid-stream client aborts this path must not
// race, hang a worker, double-free an id, or leak KV. The specific
// hazards, all only reachable under real multi-core parallelism (an
// arm64 CI box with 2 cores cannot drive them):
//   - double release: a generation that finishes as its client dies runs
//     BOTH the terminal release and the write-failure release for the
//     same id; this must be idempotent under contention, not just in
//     isolation.
//   - turnstile: each streaming cancel joins the read_waiters_ turnstile
//     against the running worker, so N concurrent aborts are N
//     concurrent turnstile entries -- the lost-wakeup-adjacent surface.
// Aborts are HARD closes (not shutdown(SHUT_WR)): only a hard close makes
// the server's next send() fail, which is what the streaming path keys
// on. ASan would catch a use-after-free from the double release, but it
// does not build on x86 here (the SSE4 matvec intrinsics fail under
// -fsanitize on GCC 13: "last argument must be an 8-bit immediate"); the
// arm64 NEON path DOES build under ASan, so that memory-safety leg is
// covered there, while this x86 case carries the real-parallelism one.
// Here the double release is safe by construction anyway: request ids
// are monotonic and never reused, and release()/cancel() no-op on an
// absent/terminal id under the engine mutex.
TEST_CASE("server survives concurrent mid-stream SSE aborts",
          "[server][e2e][stress]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    // prompt_cache off (PR#92 review finding): a COMPLETED request --
    // the step-2 fresh completion below -- otherwise caches its
    // whole-block prefix and the prefix cache RETAINS it, so the
    // free_after == free_before assertion would hold only while the
    // prompt stays under one KV block (block_tokens 16). With it off the
    // KV-baseline check means "the aborted generations released their
    // blocks" for ANY prompt length, not a block-size coincidence. The
    // aborted requests never cache regardless (insertion is gated on
    // kDone).
    locus::server::OpenAiServer::Options opt;
    opt.prompt_cache = false;
    TestServer ts(model, tok, opt);

    // Baseline KV and request count (also proves /metrics works before
    // the storm).
    const long free_before = scrape_metric(ts.port, "locus_kv_blocks_free");
    REQUIRE(free_before > 0);
    const long reqs_before = scrape_metric(ts.port, "locus_requests_total");
    REQUIRE(reqs_before >= 0);

    // Many concurrent clients that start an SSE stream, read a little
    // (so the stream is actually flowing and a worker is engaged), then
    // HARD close mid-stream. A short max_tokens makes some aborts land
    // right as the generation finishes -> the double-release collision.
    constexpr int kThreads = 24;
    constexpr int kRounds = 4;
    for (int round = 0; round < kRounds; ++round) {
        std::vector<std::thread> aborters;
        aborters.reserve(kThreads);
        for (int i = 0; i < kThreads; ++i) {
            aborters.emplace_back([&] {
                const int c =
                    locus::pipeline::connect_to("127.0.0.1", ts.port, 2000);
                if (c < 0) {
                    return;
                }
                const json req{{"prompt", "Once upon a time"},
                               {"max_tokens", 8},
                               {"stream", true}};
                if (send_all(c, make_post("/v1/completions", req.dump()))) {
                    // Read briefly so the stream has started, then abort.
                    locus::pipeline::set_recv_timeout(c, 200);
                    char buf[64];
                    (void)::recv(c, buf, sizeof(buf), 0);
                }
                ::close(c);  // hard close -> server send() fails
            });
        }
        for (auto& t : aborters) {
            t.join();
        }
    }

    // Premise check (PR#92 review finding): the aborts actually reached
    // the streaming route. locus_requests_total is incremented in the
    // completions/messages route handler BEFORE submit, so a 404, a
    // rejected request, or a refactor that stopped routing to the stream
    // would NOT move it. Without this the whole case would pass even if
    // every abort missed the streaming path (verified: pointing the
    // aborters at a 404 route still passed), asserting only that an idle
    // server is healthy. Each of the kThreads x kRounds aborters sends
    // exactly one request, so the counter must have grown by at least
    // that many.
    const long reqs_after = scrape_metric(ts.port, "locus_requests_total");
    REQUIRE(reqs_after - reqs_before >= kThreads * kRounds);

    // 1. The acceptor/fast path is still responsive (a wedged worker or
    //    acceptor would stall this).
    {
        const int c =
            locus::pipeline::connect_to("127.0.0.1", ts.port, 2000);
        REQUIRE(c >= 0);
        locus::pipeline::set_recv_timeout(c, 5000);
        REQUIRE(send_all(c, make_get("/health")));
        const std::string r = recv_to_eof(c);
        ::close(c);
        REQUIRE(status_of(r) == 200);
    }

    // 2. The engine is not wedged: a fresh non-streaming completion still
    //    runs to completion and returns real content.
    {
        const int c =
            locus::pipeline::connect_to("127.0.0.1", ts.port, 2000);
        REQUIRE(c >= 0);
        locus::pipeline::set_recv_timeout(c, 20000);
        const json req{{"prompt", "Once upon a time"}, {"max_tokens", 4}};
        REQUIRE(send_all(c, make_post("/v1/completions", req.dump())));
        const std::string r = recv_to_eof(c);
        ::close(c);
        REQUIRE(status_of(r) == 200);
    }

    // 3. No KV leak: every aborted generation released its blocks, so
    //    free returns to the baseline. Release is async behind the
    //    worker, so poll (bounded) rather than reading once.
    long free_after = -1;
    for (int i = 0; i < 400; ++i) {
        free_after = scrape_metric(ts.port, "locus_kv_blocks_free");
        if (free_after >= free_before) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    REQUIRE(free_after == free_before);
}

TEST_CASE("audio speech 503 when no decoder configured",
          "[server][audio]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    TestServer ts(model, tok);  // no audio_decoder_path in Options.
    httplib::Client client("127.0.0.1", ts.port);
    auto res = client.Post("/v1/audio/speech",
                           json{{"input", "hi"}}.dump(),
                           "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 503);
}

namespace {
std::string orpheus_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/orpheus-3b-0.1-ft-Q4_K_M.gguf";
}
std::string snac_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/snac_24khz_decoder.gguf";
}
}  // namespace

namespace {

/** A real Orpheus + SNAC server on an ephemeral port, for audio e2e. */
struct AudioServer {
    explicit AudioServer(std::uint32_t audio_max_tokens)
        : g(locus::gguf::GgufFile::open(orpheus_path())),
          model(locus::model::TransformerModel::load(g)),
          tok(locus::tok::tokenizer_from_gguf(g)),
          server(model, *tok, [&] {
              locus::server::OpenAiServer::Options o;
              o.audio_decoder_path = snac_path();
              o.audio_max_tokens = audio_max_tokens;
              return o;
          }()) {
        port = server.bind_any_port("127.0.0.1");
        REQUIRE(port > 0);
        th = std::thread([this] { server.listen_after_bind(); });
    }
    ~AudioServer() {
        server.stop();
        th.join();
    }
    httplib::Result post(const json& req) {
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(600, 0);
        return c.Post("/v1/audio/speech", req.dump(), "application/json");
    }

    locus::gguf::GgufFile g;
    locus::model::TransformerModel model;
    std::unique_ptr<locus::tok::Tokenizer> tok;
    locus::server::OpenAiServer server;
    int port = -1;
    std::thread th;
};

/** REQUIRE that a request got a response; on failure, report why
 * (connection refused, read timeout, ...) instead of a bare false. */
#define REQUIRE_RES(r)                                                \
    do {                                                              \
        INFO("httplib error: " << httplib::to_string((r).error()));   \
        REQUIRE(r);                                                   \
    } while (0)

/** RMS of a 16-bit mono WAV body (after the 44-byte header). */
double wav_rms(const std::string& body) {
    double sum = 0.0;
    std::size_t n = 0;
    for (std::size_t i = 44; i + 1 < body.size(); i += 2) {
        std::int16_t s;
        std::memcpy(&s, body.data() + i, 2);
        const double v = s / 32768.0;
        sum += v * v;
        ++n;
    }
    return n ? std::sqrt(sum / n) : 0.0;
}

}  // namespace

// End-to-end wiring on a real Orpheus + SNAC: the HTTP path (prompt
// wrapping -> sampling -> codec sampler -> decoder -> WAV), with a
// 4-frame cap so each request stays quick. The decode math is pinned
// bit-exact in test_snac.cpp.
TEST_CASE("audio speech returns a WAV end to end",
          "[server][audio][e2e]") {
    if (!std::filesystem::exists(orpheus_path()) ||
        !std::filesystem::exists(snac_path())) {
        SKIP("orpheus model or snac decoder absent");
    }
    AudioServer as(28);  // 4 frames.
    const json base{{"input", "The quick brown fox."}, {"voice", "tara"}};

    // Default request: sampled (Orpheus reference settings), a RIFF WAV,
    // and generation bounded by the server cap.
    auto res = as.post(base);
    REQUIRE_RES(res);
    REQUIRE(res->status == 200);
    REQUIRE(res->get_header_value("Content-Type") == "audio/wav");
    REQUIRE(res->body.substr(0, 4) == "RIFF");
    REQUIRE(std::stoul(res->get_header_value("X-Audio-Tokens")) <= 28);

    // A seedless request reports the seed the server drew; replaying it
    // reproduces the audio byte for byte.
    REQUIRE(res->has_header("X-Audio-Seed"));
    // Where the time went (i#173): one entry per stage, in order.
    const auto timing = res->get_header_value("Server-Timing");
    INFO(timing);
    const auto p = timing.find("prefill;dur=");
    const auto g = timing.find("generate;dur=");
    const auto d = timing.find("decode;dur=");
    REQUIRE(p == 0);
    REQUIRE(g != std::string::npos);
    REQUIRE(d != std::string::npos);
    REQUIRE((p < g && g < d));
    json replay = base;
    replay["seed"] = static_cast<std::int64_t>(
        std::stoull(res->get_header_value("X-Audio-Seed")));
    auto rp = as.post(replay);
    REQUIRE_RES(rp);
    REQUIRE(rp->body == res->body);

    // A fixed seed reproduces the audio byte for byte.
    json seeded = base;
    seeded["seed"] = 7;
    auto s1 = as.post(seeded);
    auto s2 = as.post(seeded);
    REQUIRE_RES(s1);
    REQUIRE_RES(s2);
    REQUIRE(s1->body == s2->body);
    // The seed changes the audio, so the replay check above is not empty
    // (a server-drawn seed equal to 7 is a 1 in 2^64 chance).
    REQUIRE(s1->body != res->body);

    // True greedy (temperature 0 AND no repetition penalty; temperature
    // 0 alone is argmax after the 1.3 penalty): deterministic without a
    // seed.
    json greedy = base;
    greedy["temperature"] = 0;
    greedy["repetition_penalty"] = 1;
    auto g1 = as.post(greedy);
    auto g2 = as.post(greedy);
    REQUIRE_RES(g1);
    REQUIRE_RES(g2);
    REQUIRE(g1->body == g2->body);

    // A request's max_tokens is clamped to the server cap; a smaller one
    // is honoured (7 tokens = at most one frame = 2048 samples).
    json big = base;
    big["max_tokens"] = 1000;
    auto b = as.post(big);
    REQUIRE_RES(b);
    REQUIRE(std::stoul(b->get_header_value("X-Audio-Tokens")) <= 28);
    json small = base;
    small["max_tokens"] = 7;
    auto sm = as.post(small);
    REQUIRE_RES(sm);
    REQUIRE(std::stoul(sm->get_header_value("X-Audio-Tokens")) <= 7);
    REQUIRE(sm->body.size() <= 44 + 2048 * 2);

    // Out-of-range sampling parameters are a 400, not a silent clamp.
    json bad = base;
    bad["top_p"] = 0;
    auto r = as.post(bad);
    REQUIRE_RES(r);
    REQUIRE(r->status == 400);
}

// i#173 streaming: "stream": true sends the audio as a chunked body,
// frame ranges going out as their lookahead arrives. A 7-frame cap is
// past the first emit (1 frame once 5 more exist), so the stream has an
// early range and a final flush. Same seed, same bytes as the buffered
// response (the decode ranges are bit-exact, test_snac.cpp); a WAV
// stream differs only in its header's unknown-length sizes.
TEST_CASE("audio speech streams the same audio as the buffered response",
          "[server][audio][e2e]") {
    if (!std::filesystem::exists(orpheus_path()) ||
        !std::filesystem::exists(snac_path())) {
        SKIP("orpheus model or snac decoder absent");
    }
    AudioServer as(49);  // 7 frames.
    json req{{"input", "The quick brown fox."}, {"voice", "tara"},
             {"seed", 9}, {"response_format", "pcm"}};
    auto buffered = as.post(req);
    REQUIRE_RES(buffered);
    REQUIRE(buffered->status == 200);
    REQUIRE(!buffered->body.empty());

    req["stream"] = true;
    auto streamed = as.post(req);
    REQUIRE_RES(streamed);
    REQUIRE(streamed->status == 200);
    REQUIRE(streamed->get_header_value("Transfer-Encoding") == "chunked");
    REQUIRE(streamed->get_header_value("X-Audio-Seed") == "9");
    REQUIRE(streamed->body == buffered->body);

    req["response_format"] = "wav";
    auto wav = as.post(req);
    REQUIRE_RES(wav);
    REQUIRE(wav->status == 200);
    REQUIRE(wav->body.size() == 44 + buffered->body.size());
    REQUIRE(wav->body.substr(0, 4) == "RIFF");
    REQUIRE(wav->body.substr(4, 4) == std::string(4, '\xFF'));
    REQUIRE(wav->body.substr(40, 4) == std::string(4, '\xFF'));
    REQUIRE(wav->body.substr(44) == buffered->body);

    json bad = req;
    bad["stream"] = "yes";
    auto r = as.post(bad);
    REQUIRE_RES(r);
    REQUIRE(r->status == 400);
}

// The point of sampling: greedy decoding never ends the utterance and
// decodes to near-silence (RMS ~0.001), while sampling ends speech on
// its own with audible output. Takes about 90 s on m2, so it is opt-in.
TEST_CASE("audio speech sampling ends the utterance with real audio",
          "[server][audio][e2e]") {
    if (!std::filesystem::exists(orpheus_path()) ||
        !std::filesystem::exists(snac_path())) {
        SKIP("orpheus model or snac decoder absent");
    }
    if (std::getenv("LOCUS_HEAVY_TESTS") == nullptr) {
        SKIP("full-utterance TTS takes ~90 s; set LOCUS_HEAVY_TESTS=1");
    }
    AudioServer as(1024);
    auto res = as.post({{"input",
                         "The quick brown fox jumps over the lazy dog."},
                        {"voice", "tara"},
                        {"seed", 42}});
    REQUIRE_RES(res);
    REQUIRE(res->status == 200);
    const long stop = std::stol(res->get_header_value("X-Audio-Stop-Token"));
    const unsigned long n =
        std::stoul(res->get_header_value("X-Audio-Tokens"));
    INFO("stop token " << stop << ", " << n << " audio tokens, rms "
                       << wav_rms(res->body));
    REQUIRE(stop >= 0);     // the model ended speech itself...
    REQUIRE(n < 1024);      // ...before the cap,
    REQUIRE(wav_rms(res->body) > 0.02);  // with audible audio.
}

namespace {

std::string janus_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/janus-pro-1b-bf16.gguf";
}
std::string vq16_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/janus-vq16-decoder.gguf";
}

/** Decodes standard base64 (the test's side of b64_json). */
std::vector<std::byte> unbase64(const std::string& s) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    REQUIRE(s.size() % 4 == 0);
    std::vector<std::byte> out;
    for (std::size_t i = 0; i < s.size(); i += 4) {
        std::uint32_t v = 0;
        int pad = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s[i + k];
            if (c == '=') {
                ++pad;
                v <<= 6;
                continue;
            }
            const int d = val(c);
            REQUIRE(d >= 0);
            v = (v << 6) | static_cast<std::uint32_t>(d);
        }
        out.push_back(static_cast<std::byte>(v >> 16));
        if (pad < 2) out.push_back(static_cast<std::byte>((v >> 8) & 0xFF));
        if (pad < 1) out.push_back(static_cast<std::byte>(v & 0xFF));
    }
    return out;
}

}  // namespace

// i#243: a JSON parse error quotes the bad input in what(); when that
// input is not valid UTF-8, the strict dump() of the error body threw
// inside the handler's catch, and the escaped exception terminated
// locus-server. Now the error body is lossy (U+FFFD) and the server
// keeps serving.
TEST_CASE("a body that is not UTF-8 gets a 400 and the server lives",
          "[server]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    TestServer ts(model, tok);
    httplib::Client client("127.0.0.1", ts.port);
    for (const char* path : {"/v1/completions", "/v1/chat/completions",
                             "/v1/messages", "/v1/embeddings"}) {
        INFO(path);
        auto res = client.Post(path, std::string("\xff\xfe{", 3),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 400);
        // The body is valid JSON with the bad bytes replaced.
        REQUIRE(json::parse(res->body).contains("error"));
    }
    auto health = client.Get("/health");
    REQUIRE(health);
    REQUIRE(health->status == 200);
}

TEST_CASE("images 503 unless a decoder and a Janus model are loaded",
          "[server][image]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const json req{{"prompt", "an apple"}};
    {
        TestServer ts(model, tok);  // no image_decoder_path
        httplib::Client client("127.0.0.1", ts.port);
        auto res = client.Post("/v1/images/generations", req.dump(),
                               "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 503);
    }
    if (!std::filesystem::exists(vq16_path())) {
        SKIP("janus-vq16-decoder.gguf absent");
    }
    // A decoder but a model without janus.* tensors: still 503.
    locus::server::OpenAiServer::Options o;
    o.image_decoder_path = vq16_path();
    o.model_path = model_path();
    TestServer ts(model, tok, o);
    httplib::Client client("127.0.0.1", ts.port);
    auto res = client.Post("/v1/images/generations", req.dump(),
                           "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 503);
}

// The whole HTTP path on Janus-Pro: request checks, prompt wrapping,
// CFG generation, VQ-16 decode, PNG and base64, against the reference
// image for the golden prompt (argmax, so deterministic). The codes are
// pinned in test_janus.cpp and the decode in test_vq16.cpp; this checks
// the wiring.
TEST_CASE("images endpoint returns the reference image end to end",
          "[server][image][e2e]") {
    const std::string golden =
        std::string(LOCUS_SOURCE_DIR) + "/tests/golden/janus_decode.png";
    if (!std::filesystem::exists(janus_path()) ||
        !std::filesystem::exists(vq16_path()) ||
        !std::filesystem::exists(golden)) {
        SKIP("janus model, vq16 decoder or janus_decode.png absent");
    }
    auto g = locus::gguf::GgufFile::open(janus_path());
    auto model = locus::model::TransformerModel::load(g);
    const auto* neon = locus::backend::find_backend("neon");
    if (neon != nullptr && neon->available) model.use_backend(*neon);
    auto tok = locus::tok::tokenizer_from_gguf(g);
    locus::server::OpenAiServer::Options o;
    o.image_decoder_path = vq16_path();
    o.model_path = janus_path();
    locus::server::OpenAiServer server(model, *tok, o);
    const int port = server.bind_any_port("127.0.0.1");
    REQUIRE(port > 0);
    std::thread th([&] { server.listen_after_bind(); });
    struct Join {
        locus::server::OpenAiServer& s;
        std::thread& t;
        ~Join() {
            s.stop();
            t.join();
        }
    } join{server, th};
    httplib::Client c("127.0.0.1", port);
    c.set_read_timeout(1800, 0);
    auto post = [&](const json& j) {
        return c.Post("/v1/images/generations", j.dump(),
                      "application/json");
    };

    const std::string prompt = "A red apple on a wooden table.";
    // Refused before any generation, so these are quick.
    for (const json& bad :
         {json{{"prompt", ""}}, json{{"prompt", prompt}, {"n", 2}},
          json{{"prompt", prompt}, {"size", "512x512"}},
          json{{"prompt", prompt}, {"response_format", "url"}},
          json{{"prompt", prompt}, {"cfg", -1}},
          json{{"prompt", prompt}, {"temperature", "hot"}},
          json{{"prompt", prompt}, {"seed", 1.5}}}) {
        INFO(bad.dump());
        auto res = post(bad);
        REQUIRE_RES(res);
        REQUIRE(res->status == 400);
    }

    if (std::getenv("LOCUS_HEAVY_TESTS") == nullptr) {
        SKIP("a full image takes minutes on a CPU; set LOCUS_HEAVY_TESTS=1");
    }
    auto res = post({{"prompt", prompt},
                     {"n", 1},
                     {"size", "384x384"},
                     {"response_format", "b64_json"},
                     {"temperature", 0}});
    REQUIRE_RES(res);
    INFO(res->body.substr(0, 200));
    REQUIRE(res->status == 200);
    INFO("Server-Timing: " << res->get_header_value("Server-Timing"));
    REQUIRE(res->has_header("X-Image-Seed"));
    const json body = json::parse(res->body);
    REQUIRE(body["data"].size() == 1);
    const auto got = png_test::parse_stored_png(
        unbase64(body["data"][0]["b64_json"].get<std::string>()));

    std::ifstream f(golden, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
    std::vector<std::byte> bytes(raw.size());
    std::memcpy(bytes.data(), raw.data(), raw.size());
    const auto want = png_test::parse_stored_png(bytes);
    REQUIRE(got.w == want.w);
    REQUIRE(got.h == want.h);
    // The decode matches the reference to about 1.5e-5 (test_vq16), so
    // a value that lands on a truncation edge may differ by one.
    std::size_t differ = 0;
    int worst = 0;
    for (std::size_t i = 0; i < want.rgb.size(); ++i) {
        const int d = std::abs(int(got.rgb[i]) - int(want.rgb[i]));
        differ += d != 0;
        worst = std::max(worst, d);
    }
    INFO(differ << " of " << want.rgb.size() << " values differ, worst "
                << worst);
    REQUIRE(worst <= 1);
    REQUIRE(differ * 1000 <= want.rgb.size());
}
