// locus-router (i#388 step 2): request kinds, picking an instance, and
// proxying buffered and streamed replies, against fake instances
// (locus's own http::Server) and a real locus-server.

#include <csignal>

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/http/client.hpp"
#include "locus/http/server.hpp"
#include "locus/model/transformer.hpp"
#include "locus/router/router.hpp"
#include "locus/server/server.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::http::Client;
using locus::http::ClientRequest;
using locus::router::Kind;
using locus::router::Router;
using locus::router::RouterConfig;
using locus::router::classify;
using nlohmann::json;

namespace {

const int kIgnoreSigpipe = [] {
    std::signal(SIGPIPE, SIG_IGN);
    return 0;
}();

/** A fake locus instance: its /v1/models lists `kinds` and `model`;
 * other paths echo "<name> <path> <body>" and record the headers. */
struct FakeInstance {
    std::string name;
    std::mutex mu;
    std::vector<std::pair<std::string, std::string>> last_headers;
    std::unique_ptr<locus::http::Server> server;
    int port = 0;

    FakeInstance(std::string n, std::vector<std::string> kinds,
                 std::string model)
        : name(std::move(n)) {
        locus::http::ServerConfig c;
        c.host = "127.0.0.1";
        c.port = 0;
        c.worker_threads = 4;
        server = std::make_unique<locus::http::Server>(
            c, [this, kinds, model](const locus::http::Request& req,
                                    locus::http::ServerResponse& res,
                                    const locus::http::RequestContext&) {
                if (req.path == "/v1/models") {
                    res.set_content(
                        json{{"object", "list"},
                             {"data", json::array({{{"id", model},
                                                    {"object", "model"}}})},
                             {"locus_kinds", kinds}}
                            .dump(),
                        "application/json");
                    return;
                }
                {
                    std::lock_guard lk(mu);
                    last_headers = req.headers;
                }
                if (req.path == "/v1/chat/completions" &&
                    req.body.find("\"stream\":true") != std::string::npos) {
                    res.set_chunked_content_provider(
                        "text/event-stream", [this](locus::http::Sink& s) {
                            for (int i = 0; i < 3; ++i) {
                                s.write("data: " + name + std::to_string(i) +
                                        "\n\n");
                            }
                            s.write("data: [DONE]\n\n");
                        });
                    return;
                }
                res.status = req.path == "/v1/audio/speech" ? 201 : 200;
                res.set_header("X-Instance", name);
                res.set_content(name + " " + req.path + " " + req.body,
                                "text/plain");
            });
        port = server->start();
        REQUIRE(port > 0);
    }
    std::string addr() const { return "127.0.0.1:" + std::to_string(port); }
    std::string header(const std::string& k) {
        std::lock_guard lk(mu);
        for (const auto& [n, v] : last_headers) {
            if (n == k) return v;
        }
        return "";
    }
};

struct RouterFixture {
    std::unique_ptr<Router> router;
    int port = 0;
    explicit RouterFixture(std::vector<std::string> instances) {
        RouterConfig cfg;
        cfg.host = "127.0.0.1";
        cfg.port = 0;
        cfg.instances = std::move(instances);
        cfg.refresh_ms = 60000;
        router = std::make_unique<Router>(cfg);
        port = router->start();
        REQUIRE(port > 0);
    }
    locus::http::ClientResponse post(const std::string& path,
                                     const json& body,
                                     std::vector<std::pair<std::string,
                                                           std::string>>
                                         headers = {}) {
        Client c("127.0.0.1", port);
        headers.emplace_back("Content-Type", "application/json");
        return c.send(ClientRequest{"POST", path, headers, body.dump()});
    }
};

}  // namespace

TEST_CASE("router: request kinds by path and body", "[router]") {
    const auto kind = [](std::string_view path, std::string_view body = "") {
        const auto r = classify(path, body);
        REQUIRE(r);
        return r->kind;
    };
    CHECK(kind("/v1/completions") == Kind::kText);
    CHECK(kind("/v1/embeddings") == Kind::kText);
    CHECK(kind("/v1/chat/completions",
               R"({"messages":[{"role":"user","content":"hi"}]})") ==
          Kind::kText);
    CHECK(kind("/v1/chat/completions",
               R"({"messages":[{"role":"user","content":[{"type":"text",
                  "text":"what"},{"type":"image_url","image_url":
                  {"url":"data:image/png;base64,AA"}}]}]})") ==
          Kind::kImageToText);
    CHECK(kind("/v1/messages",
               R"({"messages":[{"role":"user","content":[{"type":"image",
                  "source":{"type":"base64","data":"AA"}}]}]})") ==
          Kind::kImageToText);
    CHECK(kind("/v1/responses",
               R"({"input":[{"role":"user","content":[{"type":
                  "input_image","image_url":"x"}]}]})") ==
          Kind::kImageToText);
    // An image word in a tool or the system prompt is not an image part.
    CHECK(kind("/v1/chat/completions",
               R"({"messages":[{"role":"user","content":"type image"}],
                  "tools":[{"type":"image"}]})") == Kind::kText);
    CHECK(kind("/v1/images/generations") == Kind::kImage);
    CHECK(kind("/v1/images/edits") == Kind::kImage);
    CHECK(kind("/v1/videos") == Kind::kVideo);
    CHECK(kind("/v1/videos/video_1/content") == Kind::kVideo);
    CHECK(kind("/v1/audio/speech") == Kind::kSpeech);
    CHECK_FALSE(classify("/v1/files", ""));
    CHECK(classify("/v1/completions", R"({"model":"m1"})")->model == "m1");
    // A body that is not JSON still gets a kind; the instance answers.
    CHECK(kind("/v1/chat/completions", "{not json") == Kind::kText);
}

TEST_CASE("router: proxies each kind to its instance", "[router]") {
    FakeInstance text("textA", {"text"}, "llama");
    FakeInstance text2("textB", {"text"}, "qwen");
    FakeInstance media("media", {"image", "speech"}, "janus");
    RouterFixture rf({text.addr(), text2.addr(), media.addr()});

    SECTION("by kind, with the status, headers and body") {
        auto r = rf.post("/v1/completions", {{"prompt", "x"}});
        REQUIRE(r.ok);
        CHECK(r.head.status == 200);
        CHECK(r.body.rfind("text", 0) == 0);
        r = rf.post("/v1/images/generations", {{"prompt", "a cat"}});
        REQUIRE(r.ok);
        CHECK(r.body.rfind("media /v1/images/generations", 0) == 0);
        CHECK(r.head.get_header_value("X-Instance") == "media");
        r = rf.post("/v1/audio/speech", {{"input", "hi"}});
        REQUIRE(r.ok);
        CHECK(r.head.status == 201);
    }
    SECTION("by model, when an instance lists it") {
        auto r = rf.post("/v1/completions", {{"model", "qwen"}, {"prompt", "x"}});
        REQUIRE(r.ok);
        CHECK(r.body.rfind("textB", 0) == 0);
        r = rf.post("/v1/completions", {{"model", "llama"}, {"prompt", "x"}});
        CHECK(r.body.rfind("textA", 0) == 0);
        // A model no one lists still goes to a text instance.
        r = rf.post("/v1/completions", {{"model", "gpt-4o"}, {"prompt", "x"}});
        REQUIRE(r.ok);
        CHECK(r.body.rfind("text", 0) == 0);
    }
    SECTION("a stream passes through as it arrives") {
        Client c("127.0.0.1", rf.port);
        std::string sse;
        int status = 0;
        const auto r = c.stream(
            ClientRequest{"POST", "/v1/chat/completions",
                          {{"Content-Type", "application/json"}},
                          json{{"model", "qwen"}, {"stream", true},
                               {"messages", json::array()}}
                              .dump()},
            [&](const locus::http::ResponseHead& h) {
                status = h.status;
                CHECK(h.get_header_value("Content-Type") ==
                      "text/event-stream");
                return true;
            },
            [&](std::string_view d) {
                sse.append(d);
                return true;
            });
        REQUIRE(r.ok);
        CHECK(status == 200);
        CHECK(sse == "data: textB0\n\ndata: textB1\n\ndata: textB2\n\n"
                     "data: [DONE]\n\n");
    }
    SECTION("headers: the client's own pass, connection ones do not") {
        rf.post("/v1/completions", {{"model", "llama"}, {"prompt", "x"}},
                {{"Authorization", "Bearer k1"}, {"Keep-Alive", "timeout=5"}});
        CHECK(text.header("Authorization") == "Bearer k1");
        CHECK(text.header("Keep-Alive").empty());
    }
    SECTION("no instance of the kind, and unknown paths") {
        auto r = rf.post("/v1/videos", {{"prompt", "x"}});
        REQUIRE(r.ok);
        CHECK(r.head.status == 503);
        CHECK(r.head.get_header_value("Retry-After") == "5");
        r = rf.post("/v1/files", json::object());
        CHECK(r.head.status == 404);
    }
    SECTION("/v1/models is the union over the instances") {
        Client c("127.0.0.1", rf.port);
        const auto r = c.send(ClientRequest{"GET", "/v1/models", {}, ""});
        REQUIRE(r.ok);
        const json j = json::parse(r.body);
        std::vector<std::string> ids;
        for (const auto& m : j["data"]) ids.push_back(m["id"]);
        CHECK(ids == std::vector<std::string>{"llama", "qwen", "janus"});
        CHECK(j["locus_kinds"] ==
              json::array({"text", "image", "speech"}));
    }
}

TEST_CASE("router: a dead instance is skipped", "[router]") {
    FakeInstance live("live", {"text"}, "m");
    std::string dead_addr;
    {
        FakeInstance gone("gone", {"text"}, "m");
        dead_addr = gone.addr();
        gone.server->stop();
    }
    RouterFixture rf({dead_addr, live.addr()});
    const auto inst = rf.router->instances();
    REQUIRE(inst.size() == 2);
    CHECK_FALSE(inst[0].healthy);
    CHECK(inst[1].healthy);
    for (int i = 0; i < 3; ++i) {
        const auto r = rf.post("/v1/completions", {{"prompt", "x"}});
        REQUIRE(r.ok);
        CHECK(r.body.rfind("live", 0) == 0);
    }
}

TEST_CASE("router: a bad instance address is refused", "[router]") {
    RouterConfig cfg;
    cfg.instances = {"no-port"};
    CHECK_THROWS(Router(cfg));
}

TEST_CASE("router: in front of a real locus-server", "[router][e2e]") {
    const std::string path =
        std::string(LOCUS_SOURCE_DIR) + "/tests/models/stories260K.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP("stories260K.gguf absent");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    locus::server::OpenAiServer server(model, tok, {});
    const int sport = server.bind_any_port("127.0.0.1");
    REQUIRE(sport > 0);
    std::thread th([&] { server.listen_after_bind(); });
    {
        RouterFixture rf({"127.0.0.1:" + std::to_string(sport)});
        const auto inst = rf.router->instances();
        REQUIRE(inst.size() == 1);
        REQUIRE(inst[0].healthy);
        CHECK(inst[0].kinds == std::vector<Kind>{Kind::kText});

        const json req{{"prompt", "Once upon a time"},
                       {"max_tokens", 8},
                       {"temperature", 0}};
        Client direct("127.0.0.1", sport);
        const auto want =
            direct.send(ClientRequest{"POST", "/v1/completions",
                                      {{"Content-Type", "application/json"}},
                                      req.dump()});
        REQUIRE(want.ok);
        const auto got = rf.post("/v1/completions", req);
        REQUIRE(got.ok);
        CHECK(got.head.status == 200);
        CHECK(json::parse(got.body)["choices"][0]["text"] ==
              json::parse(want.body)["choices"][0]["text"]);

        // A streamed chat reply comes through chunk by chunk and ends
        // with [DONE].
        Client c("127.0.0.1", rf.port);
        std::string sse;
        const auto r = c.stream(
            ClientRequest{"POST", "/v1/chat/completions",
                          {{"Content-Type", "application/json"}},
                          json{{"stream", true},
                               {"max_tokens", 6},
                               {"messages",
                                json::array({{{"role", "user"},
                                              {"content", "hi"}}})}}
                              .dump()},
            nullptr, [&](std::string_view d) {
                sse.append(d);
                return true;
            });
        REQUIRE(r.ok);
        CHECK(sse.find("chat.completion.chunk") != std::string::npos);
        CHECK(sse.size() >= 14);
        CHECK(sse.substr(sse.size() - 14) == "data: [DONE]\n\n");
    }
    server.stop();
    th.join();
}
