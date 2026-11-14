// locus-router (i#388 step 2): request kinds, picking an instance, and
// proxying buffered and streamed replies, against fake instances
// (locus's own http::Server) and a real locus-server.

#include <csignal>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <map>
#include <set>
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
#include "locus/pipeline/net.hpp"
#include "locus/router/router.hpp"
#include "locus/server/server.hpp"
#include "locus/tok/tokenizer.hpp"
#include "router_cli.hpp"

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
    /** Each proxied request waits this long before answering. */
    std::atomic<int> delay_ms{0};
    /** Answer 503 with Retry-After, as a busy media instance does. */
    std::atomic<bool> busy{false};
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
                if (busy) {
                    res.status = 503;
                    res.set_header("Retry-After", "7");
                    res.set_content(name + " busy", "text/plain");
                    return;
                }
                if (delay_ms > 0) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(delay_ms.load()));
                }
                if (req.path == "/v1/chat/completions" &&
                    req.body.find("\"stream\":true") != std::string::npos) {
                    // `chunks` pieces, `gap_ms` apart.
                    const int chunks =
                        req.body.find("\"long\":true") != std::string::npos
                            ? 50
                            : 3;
                    res.set_chunked_content_provider(
                        "text/event-stream",
                        [this, chunks](locus::http::Sink& s) {
                            for (int i = 0; i < chunks; ++i) {
                                if (!s.write("data: " + name +
                                             std::to_string(i) + "\n\n")) {
                                    return;
                                }
                                if (chunks > 3) {
                                    std::this_thread::sleep_for(
                                        std::chrono::milliseconds(20));
                                }
                            }
                            s.write("data: [DONE]\n\n");
                        });
                    return;
                }
                // Video jobs: POST makes one this instance knows by its
                // own id; GET finds only its own.
                const std::string own = "video_" + name + "1";
                if (req.path == "/v1/videos" && req.method == "POST") {
                    res.set_content(json{{"id", own}, {"object", "video"},
                                         {"status", "queued"}}
                                        .dump(),
                                    "application/json");
                    return;
                }
                if (req.path.rfind("/v1/videos/", 0) == 0) {
                    const std::string rest = req.path.substr(11);
                    const std::string id = rest.substr(0, rest.find('/'));
                    if (id != own) {
                        res.status = 404;
                        res.set_content("no video " + id, "text/plain");
                    } else if (rest.find("/content") != std::string::npos) {
                        res.set_content("MP4 of " + name, "video/mp4");
                    } else {
                        res.set_content(json{{"id", own}, {"owner", name},
                                             {"status", "completed"}}
                                            .dump(),
                                        "application/json");
                    }
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
    explicit RouterFixture(std::vector<std::string> instances,
                           RouterConfig cfg = {}) {
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
    CHECK(kind("/v1/images/edits") == Kind::kImageEdit);
    CHECK(kind("/v1/videos") == Kind::kVideo);
    CHECK(kind("/v1/videos/video_1/content") == Kind::kVideo);
    CHECK(kind("/v1/audio/speech") == Kind::kSpeech);
    CHECK(kind("/v1/rerank") == Kind::kRerank);
    CHECK(kind("/rerank") == Kind::kRerank);
    CHECK(locus::router::kind_name(Kind::kRerank) == "rerank");
    CHECK(locus::router::parse_kind("rerank") == Kind::kRerank);
    CHECK(kind("/v1/classify") == Kind::kClassify);
    CHECK(locus::router::kind_name(Kind::kClassify) == "classify");
    CHECK(locus::router::parse_kind("classify") == Kind::kClassify);
    CHECK(locus::router::kind_name(Kind::kImageEdit) == "image-edit");
    CHECK(locus::router::parse_kind("image-edit") == Kind::kImageEdit);
    // Embeddings stay text; the flag lets a classify instance take them.
    CHECK(kind("/v1/embeddings") == Kind::kText);
    CHECK(classify("/v1/embeddings", "")->embeddings);
    CHECK_FALSE(classify("/v1/completions", "")->embeddings);
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

// i#633: an edit-only instance (Qwen-Image-Edit) lists image-edit, a
// generation-only one (PixArt) image, and each route goes to its own;
// a fleet of servers from before image-edit sends edits to "image".
TEST_CASE("router: image edits and generations go apart", "[router]") {
    SECTION("an edit-only and a generation-only instance") {
        FakeInstance edit("edit", {"text", "image-edit"}, "qwen-edit");
        FakeInstance gen("gen", {"text", "image"}, "pixart");
        RouterFixture rf({edit.addr(), gen.addr()});
        for (int k = 0; k < 4; ++k) {  // not one lucky pick
            auto r = rf.post("/v1/images/edits", {{"prompt", "purple"}});
            REQUIRE(r.ok);
            CHECK(r.body.rfind("edit /v1/images/edits", 0) == 0);
            r = rf.post("/v1/images/generations", {{"prompt", "a cat"}});
            REQUIRE(r.ok);
            CHECK(r.body.rfind("gen /v1/images/generations", 0) == 0);
        }
    }
    SECTION("servers that list only image take edits too") {
        FakeInstance old("old", {"text", "image"}, "flux");
        RouterFixture rf({old.addr()});
        auto r = rf.post("/v1/images/edits", {{"prompt", "purple"}});
        REQUIRE(r.ok);
        CHECK(r.head.status == 200);
        CHECK(r.body.rfind("old /v1/images/edits", 0) == 0);
    }
    SECTION("once one lists image-edit, image-only ones get no edits") {
        FakeInstance edit("edit", {"image-edit"}, "qwen-edit");
        FakeInstance gen("gen", {"image"}, "pixart");
        RouterFixture rf({gen.addr(), edit.addr()});
        for (int k = 0; k < 4; ++k) {
            const auto r = rf.post("/v1/images/edits", {{"prompt", "x"}});
            REQUIRE(r.ok);
            CHECK(r.body.rfind("edit /v1/images/edits", 0) == 0);
        }
    }
}

// i#583: a CNN classifier instance (locus-server --image-classifier)
// takes /v1/classify, and /v1/embeddings under its model's name.
TEST_CASE("router: classify and image embeddings go to a classify instance",
          "[router]") {
    FakeInstance text("text", {"text"}, "llama");
    FakeInstance cnn("cnn", {"classify"}, "resnet50.tv_in1k");
    RouterFixture rf({text.addr(), cnn.addr()});

    auto r = rf.post("/v1/classify", {{"image", "AA"}});
    REQUIRE(r.ok);
    CHECK(r.body.rfind("cnn /v1/classify", 0) == 0);
    r = rf.post("/v1/embeddings",
                {{"model", "resnet50.tv_in1k"}, {"input", "AA"}});
    REQUIRE(r.ok);
    CHECK(r.body.rfind("cnn /v1/embeddings", 0) == 0);
    // Text embeddings, by another model or none, stay on text.
    r = rf.post("/v1/embeddings", {{"model", "llama"}, {"input", "hi"}});
    REQUIRE(r.ok);
    CHECK(r.body.rfind("text", 0) == 0);
    r = rf.post("/v1/embeddings", {{"input", "hi"}});
    REQUIRE(r.ok);
    CHECK(r.body.rfind("text", 0) == 0);
    // A completion that names the classifier's model is not its kind.
    r = rf.post("/v1/completions",
                {{"model", "resnet50.tv_in1k"}, {"prompt", "x"}});
    REQUIRE(r.ok);
    CHECK(r.body.rfind("text", 0) == 0);
}

// i#605: a buffered image sends nothing until it is done, so the media
// kinds get their own read timeout. With the text one shorter than the
// instance's answer and the media one longer, the text request times
// out and the image request is served.
TEST_CASE("router: media requests get their own read timeout", "[router]") {
    FakeInstance slow("slow", {"text", "image", "speech"}, "m");
    slow.delay_ms = 400;
    RouterConfig cfg;
    cfg.text_read_timeout_ms = 150;
    cfg.media_read_timeout_ms = 3000;
    RouterFixture rf({slow.addr()}, cfg);

    auto r = rf.post("/v1/completions", {{"prompt", "x"}});
    REQUIRE(r.ok);
    CHECK(r.head.status != 200);
    r = rf.post("/v1/images/generations", {{"prompt", "a cat"}});
    REQUIRE(r.ok);
    CHECK(r.head.status == 200);
    CHECK(r.body.rfind("slow /v1/images/generations", 0) == 0);
    // The pooled connection's timeout follows each request's kind.
    r = rf.post("/v1/completions", {{"prompt", "x"}});
    REQUIRE(r.ok);
    CHECK(r.head.status != 200);
}

// i#610: a text reply can be as slow as an image (a buffered chat on a
// CPU, a long prefill), so text has its own read timeout, long by
// default; upstream's (the polls' base options) no longer applies.
TEST_CASE("router: slow text replies get the text read timeout",
          "[router]") {
    FakeInstance slow("slow", {"text"}, "m");
    slow.delay_ms = 400;
    RouterConfig cfg;
    cfg.upstream.read_timeout_ms = 150;
    REQUIRE(cfg.text_read_timeout_ms == 30 * 60 * 1000);
    RouterFixture rf({slow.addr()}, cfg);
    auto r = rf.post("/v1/chat/completions",
                     {{"messages", json::array({{{"role", "user"},
                                                 {"content", "hi"}}})}});
    REQUIRE(r.ok);
    CHECK(r.head.status == 200);
    CHECK(r.body.rfind("slow /v1/chat/completions", 0) == 0);
    r = rf.post("/v1/completions", {{"prompt", "x"}});
    REQUIRE(r.ok);
    CHECK(r.head.status == 200);
}

// The two read timeouts' flags: seconds, 0 to 86400.
TEST_CASE("router: --text-timeout and --media-timeout parse", "[router]") {
    const auto parse = [](std::vector<std::string> args,
                          RouterConfig& cfg) {
        args.push_back("127.0.0.1:9");  // an instance, required
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(a.data());
        bool help = false;
        return locus_tools::parse_router_args(static_cast<int>(argv.size()),
                                              argv.data(), 0, cfg, help);
    };
    RouterConfig cfg;
    CHECK(parse({"--text-timeout", "90", "--media-timeout", "0"}, cfg) ==
          "");
    CHECK(cfg.text_read_timeout_ms == 90 * 1000);
    CHECK(cfg.media_read_timeout_ms == 0);
    CHECK(parse({"--text-timeout", "86400"}, cfg) == "");
    CHECK(cfg.text_read_timeout_ms == 86400 * 1000);
    CHECK(parse({"--text-timeout", "86401"}, cfg) ==
          "bad --text-timeout: 86401");
    CHECK(parse({"--media-timeout", "-1"}, cfg) ==
          "bad --media-timeout: -1");
    CHECK(parse({"--text-timeout", "x"}, cfg) == "bad --text-timeout: x");
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

TEST_CASE("router: session keys", "[router]") {
    using locus::router::session_key;
    CHECK(session_key("abc", "{}") == "h:abc");
    CHECK(session_key("", R"({"prompt_cache_key":"pk","messages":[]})") ==
          "k:pk");
    CHECK(session_key("", R"({"metadata":{"user_id":"u7"}})") == "u:u7");
    // A conversation keeps its key as it grows: the system prompt and
    // the first user turn decide it.
    const std::string turn1 = R"({"messages":[
        {"role":"system","content":"Be brief."},
        {"role":"user","content":"Hello"}]})";
    const std::string turn2 = R"({"messages":[
        {"role":"system","content":"Be brief."},
        {"role":"user","content":"Hello"},
        {"role":"assistant","content":"Hi"},
        {"role":"user","content":"And?"}]})";
    const std::string other = R"({"messages":[
        {"role":"system","content":"Be brief."},
        {"role":"user","content":"Goodbye"}]})";
    const std::string k1 = session_key("", turn1);
    CHECK(k1.rfind("p:", 0) == 0);
    CHECK(session_key("", turn2) == k1);
    CHECK(session_key("", other) != k1);
    // Anthropic system field, Responses input, completions prompt.
    CHECK(session_key("", R"({"system":"S","messages":[{"role":"user",
              "content":[{"type":"text","text":"Q"}]}]})") ==
          session_key("", R"({"system":"S","messages":[{"role":"user",
              "content":"Q"}]})"));
    CHECK(!session_key("", R"({"input":"hi"})").empty());
    CHECK(!session_key("", R"({"prompt":"Once"})").empty());
    CHECK(session_key("", "").empty());
    CHECK(session_key("", "not json").empty());
}

TEST_CASE("router: rendezvous order moves only a leaving instance's keys",
          "[router]") {
    using locus::router::rendezvous_score;
    const auto winner = [](const std::string& key,
                           const std::vector<std::string>& insts) {
        std::string best;
        std::uint64_t top = 0;
        for (const auto& i : insts) {
            const auto s = rendezvous_score(key, i);
            if (best.empty() || s > top) {
                top = s;
                best = i;
            }
        }
        return best;
    };
    const std::vector<std::string> three{"a:1", "b:1", "c:1"};
    const std::vector<std::string> two{"a:1", "b:1"};
    std::map<std::string, int> share;
    int moved = 0;
    for (int k = 0; k < 3000; ++k) {
        const std::string key = "session-" + std::to_string(k);
        const std::string w3 = winner(key, three);
        ++share[w3];
        if (w3 != "c:1") {
            // Only keys that c:1 held may move when it leaves.
            CHECK(winner(key, two) == w3);
        } else {
            ++moved;
        }
    }
    // Each instance holds about a third of the keys.
    for (const auto& [inst, n] : share) {
        INFO(inst);
        CHECK(n > 800);
        CHECK(n < 1200);
    }
    CHECK(moved == share["c:1"]);
    // The same in every router: a pure function of its inputs.
    CHECK(rendezvous_score("k", "a:1") == rendezvous_score("k", "a:1"));
}

TEST_CASE("router: a session stays on one instance", "[router]") {
    FakeInstance a("A", {"text"}, "m"), b("B", {"text"}, "m"),
        c("C", {"text"}, "m");
    RouterFixture r1({a.addr(), b.addr(), c.addr()});
    RouterFixture r2({a.addr(), b.addr(), c.addr()});
    const auto where = [](RouterFixture& rf, const std::string& session) {
        const auto r = rf.post("/v1/completions", {{"prompt", "x"}},
                               {{"X-Session-Id", session}});
        REQUIRE(r.ok);
        return r.body.substr(0, 1);
    };
    const std::string home = where(r1, "s-1");
    for (int i = 0; i < 10; ++i) {
        CHECK(where(r1, "s-1") == home);
        CHECK(where(r2, "s-1") == home);  // the other router agrees
    }
    std::set<std::string> seen;
    for (int s = 0; s < 30; ++s) {
        seen.insert(where(r1, "spread-" + std::to_string(s)));
    }
    for (const auto& in : r1.router->instances()) {
        CHECK(in.healthy);
    }
    CHECK(seen.size() == 3);
}

TEST_CASE("router: a loaded or busy instance passes requests on",
          "[router]") {
    FakeInstance a("A", {"text"}, "m"), b("B", {"text"}, "m"),
        c("C", {"text"}, "m");
    FakeInstance* by_addr[] = {&a, &b, &c};
    RouterFixture rf({a.addr(), b.addr(), c.addr()});
    locus::router::Route route;
    route.session = "h:hot";
    const auto order = rf.router->candidates(route);
    REQUIRE(order.size() == 3);
    FakeInstance& first = *by_addr[order[0]];

    SECTION("over the load bound, the key's next instance answers") {
        first.delay_ms = 600;
        std::vector<std::future<std::string>> f;
        for (int i = 0; i < 3; ++i) {
            f.push_back(std::async(std::launch::async, [&] {
                const auto r = rf.post("/v1/completions", {{"prompt", "x"}},
                                       {{"X-Session-Id", "hot"}});
                return r.ok ? r.body.substr(0, 1) : std::string("!");
            }));
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::multiset<std::string> got;
        for (auto& x : f) got.insert(x.get());
        CHECK(got.count(first.name) >= 1);
        CHECK(got.count(first.name) < 3);  // the rest went elsewhere
        CHECK(got.count("!") == 0);
    }
    SECTION("a 503 with Retry-After moves to the next instance") {
        first.busy = true;
        const auto r = rf.post("/v1/completions", {{"prompt", "x"}},
                               {{"X-Session-Id", "hot"}});
        REQUIRE(r.ok);
        CHECK(r.head.status == 200);
        CHECK(r.body.substr(0, 1) == by_addr[order[1]]->name);
    }
    SECTION("all busy: the last 503 comes back with its Retry-After") {
        a.busy = b.busy = c.busy = true;
        const auto r = rf.post("/v1/completions", {{"prompt", "x"}});
        REQUIRE(r.ok);
        CHECK(r.head.status == 503);
        CHECK(r.head.get_header_value("Retry-After") == "7");
    }
}

// i#388: "srv:NAME" lists the lowest priority's SRV targets, looked up
// again on every refresh; a target the answer drops is never picked,
// an empty answer keeps the last list, and a target that comes back is
// picked again.
TEST_CASE("router: an SRV name lists instances and follows changes",
          "[router]") {
    using locus::pipeline::SrvTarget;
    FakeInstance a("A", {"text"}, "m"), b("B", {"text"}, "m"),
        c("C", {"text"}, "m");
    std::mutex mu;
    std::vector<SrvTarget> answer = {{"127.0.0.1", a.port, 10, 0},
                                     {"127.0.0.1", b.port, 10, 0},
                                     {"127.0.0.1", c.port, 20, 0}};
    RouterConfig cfg;
    cfg.srv_lookup = [&](const std::string& name) {
        CHECK(name == "_locus._tcp.test");
        std::lock_guard lk(mu);
        return answer;
    };
    RouterFixture rf({"srv:_locus._tcp.test"}, cfg);
    // The instances 20 completions reach (each its own prompt, so the
    // session keys differ and every instance gets some).
    const auto reached = [&] {
        std::set<std::string> seen;
        for (int i = 0; i < 20; ++i) {
            const auto r = rf.post(
                "/v1/completions",
                json{{"prompt", "p" + std::to_string(i)}, {"max_tokens", 1}});
            REQUIRE(r.ok);
            REQUIRE(r.head.status == 200);
            seen.insert(r.head.get_header_value("X-Instance"));
        }
        return seen;
    };
    const auto set = [&](std::vector<SrvTarget> v) {
        {
            std::lock_guard lk(mu);
            answer = std::move(v);
        }
        rf.router->refresh();
    };
    const auto gone = [&](int port) {
        for (const auto& in : rf.router->instances()) {
            if (in.port == port) return in.gone;
        }
        FAIL("no instance on port " << port);
        return false;
    };

    // Priority 10 is used; C (priority 20) is a fallback, not listed.
    CHECK(reached() == std::set<std::string>{"A", "B"});
    CHECK(rf.router->instances().size() == 2);

    // B leaves the record and C joins it at priority 10.
    set({{"127.0.0.1", a.port, 10, 0}, {"127.0.0.1", c.port, 10, 0}});
    CHECK(reached() == std::set<std::string>{"A", "C"});
    CHECK(gone(b.port));
    CHECK_FALSE(gone(c.port));

    // An empty answer (a DNS failure) keeps the last list.
    set({});
    CHECK(reached() == std::set<std::string>{"A", "C"});

    // B comes back; C leaves.
    set({{"127.0.0.1", a.port, 10, 0}, {"127.0.0.1", b.port, 10, 0}});
    CHECK(reached() == std::set<std::string>{"A", "B"});
    CHECK_FALSE(gone(b.port));
    CHECK(gone(c.port));
    CHECK(rf.router->instances().size() == 3);  // the list only grows

    RouterConfig bad;
    bad.instances = {"srv:"};
    CHECK_THROWS_AS(Router(bad), std::invalid_argument);
}

// i#492: a target two SRV names list is one entry, gone only when
// neither lists it; a static host:port is never doubled by a name.
TEST_CASE("router: a target two SRV names share is one instance",
          "[router]") {
    using locus::pipeline::SrvTarget;
    FakeInstance a("A", {"text"}, "m"), b("B", {"text"}, "m"),
        s("S", {"text"}, "m");
    std::mutex mu;
    std::map<std::string, std::vector<SrvTarget>> answers = {
        {"_x._tcp.test",
         {{"127.0.0.1", a.port, 0, 0}, {"127.0.0.1", b.port, 0, 0}}},
        {"_y._tcp.test",
         {{"127.0.0.1", a.port, 0, 0}, {"127.0.0.1", s.port, 0, 0}}}};
    RouterConfig cfg;
    cfg.srv_lookup = [&](const std::string& name) {
        std::lock_guard lk(mu);
        return answers[name];
    };
    RouterFixture rf({"srv:_x._tcp.test", "srv:_y._tcp.test", s.addr()},
                     cfg);
    const auto entries = [&](int port) {
        int n = 0;
        for (const auto& in : rf.router->instances()) n += in.port == port;
        return n;
    };
    const auto gone = [&](int port) {
        for (const auto& in : rf.router->instances()) {
            if (in.port == port) return in.gone;
        }
        return true;
    };
    CHECK(entries(a.port) == 1);  // listed by both names
    CHECK(entries(s.port) == 1);  // static, and listed by _y
    CHECK(rf.router->instances().size() == 3);

    // _x drops A; _y still lists it, so it stays.
    {
        std::lock_guard lk(mu);
        answers["_x._tcp.test"] = {{"127.0.0.1", b.port, 0, 0}};
    }
    rf.router->refresh();
    CHECK_FALSE(gone(a.port));
    // _y drops A too: now it is gone. S, static, never is.
    {
        std::lock_guard lk(mu);
        answers["_y._tcp.test"] = {{"127.0.0.1", b.port, 0, 0}};
    }
    rf.router->refresh();
    CHECK(gone(a.port));
    CHECK_FALSE(gone(s.port));
    CHECK(entries(b.port) == 1);
}

// i#388 step 5: an instance registers itself with a key, is reached
// while its registration lasts, drops out when it runs out or on
// deregistration, and comes back with the next registration.
TEST_CASE("router: instances register themselves", "[router]") {
    using locus::router::Registrar;
    FakeInstance a("A", {"text"}, "m"), b("B", {"text"}, "m");
    RouterConfig cfg;
    cfg.register_keys = {"k1", "k2"};
    RouterFixture rf({b.addr()}, cfg);  // B is static
    const auto reached = [&] {
        std::set<std::string> seen;
        for (int i = 0; i < 20; ++i) {
            const auto r = rf.post(
                "/v1/completions",
                json{{"prompt", "p" + std::to_string(i)}, {"max_tokens", 1}});
            REQUIRE(r.ok);
            REQUIRE(r.head.status == 200);
            seen.insert(r.head.get_header_value("X-Instance"));
        }
        return seen;
    };
    CHECK(reached() == std::set<std::string>{"B"});

    Registrar::Config rc;
    rc.router_host = "127.0.0.1";
    rc.router_port = rf.port;
    rc.address = a.addr();
    rc.key = "k2";
    rc.ttl_s = 1;
    Registrar reg(rc);
    REQUIRE(reg.register_once());
    CHECK(reached() == std::set<std::string>{"A", "B"});  // polled at once

    // Its one-second registration runs out: never picked after that.
    std::this_thread::sleep_for(std::chrono::milliseconds(1300));
    CHECK(reached() == std::set<std::string>{"B"});
    REQUIRE(reg.register_once());
    CHECK(reached() == std::set<std::string>{"A", "B"});
    REQUIRE(reg.deregister());
    CHECK(reached() == std::set<std::string>{"B"});

    // A static instance registering stays one entry.
    Registrar::Config rb = rc;
    rb.address = b.addr();
    REQUIRE(Registrar(rb).register_once());
    CHECK(rf.router->instances().size() == 2);

    // The heartbeat thread registers, and stop() deregisters.
    rc.ttl_s = 3;
    rc.first_delay_ms = 0;
    {
        Registrar beat(rc);
        beat.start();
        bool listed = false;
        for (int i = 0; i < 50 && !listed; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            for (const auto& in : rf.router->instances()) {
                listed |= in.port == a.port && !in.gone && in.healthy;
            }
        }
        CHECK(listed);
        beat.stop();
    }
    CHECK(reached() == std::set<std::string>{"B"});

    // Keys: none, or a wrong one, is a 401.
    Client c("127.0.0.1", rf.port);
    const std::string body = json{{"address", a.addr()}}.dump();
    auto r = c.send(ClientRequest{"POST", "/locus/register", {}, body});
    CHECK(r.head.status == 401);
    r = c.send(ClientRequest{"POST", "/locus/register",
                             {{"Authorization", "Bearer k3"}}, body});
    CHECK(r.head.status == 401);
    r = c.send(ClientRequest{"POST", "/locus/register",
                             {{"Authorization", "Bearer k1"}},
                             json{{"address", "nope"}}.dump()});
    CHECK(r.head.status == 400);
    r = c.send(ClientRequest{"POST", "/locus/register",
                             {{"Authorization", "Bearer k1"}},
                             json{{"address", a.addr()}, {"ttl_s", 0}}.dump()});
    CHECK(r.head.status == 400);
}

// A register key must not be a way to collect the router's keys: an
// instance found by registration (or SRV) gets an instance key only
// from a line that names its address, never the "*" one (review of
// #496). And a router refuses a request another router sent, so a
// registration at a router's own address cannot loop.
TEST_CASE("router: a registered instance never gets the wildcard key",
          "[router]") {
    using locus::router::Registrar;
    using UA = RouterConfig::UpstreamAuth;
    FakeInstance a("A", {"text"}, "m"), b("B", {"text"}, "m");
    RouterConfig cfg;
    cfg.register_keys = {"rk"};
    cfg.upstream_auth = UA::kKey;
    cfg.instance_keys = {{"*", "sk-wild"}};
    RouterFixture rf({b.addr()}, cfg);  // B is static
    Registrar::Config rc;
    rc.router_host = "127.0.0.1";
    rc.router_port = rf.port;
    rc.address = a.addr();
    rc.key = "rk";
    REQUIRE(Registrar(rc).register_once());
    std::set<std::string> seen;
    for (int i = 0; i < 20; ++i) {
        const auto r = rf.post(
            "/v1/completions",
            json{{"prompt", "p" + std::to_string(i)}, {"max_tokens", 1}});
        REQUIRE(r.ok);
        REQUIRE(r.head.status == 200);
        seen.insert(r.head.get_header_value("X-Instance"));
    }
    REQUIRE(seen == std::set<std::string>{"A", "B"});
    CHECK(b.header("Authorization") == "Bearer sk-wild");
    CHECK(a.header("Authorization").empty());
    // Every proxied request carries the router's loop mark.
    CHECK(a.header("X-Locus-Router") == "1");

    // A request with that mark is refused, not proxied.
    Client c("127.0.0.1", rf.port);
    const auto r = c.send(ClientRequest{
        "POST", "/v1/completions",
        {{"Content-Type", "application/json"}, {"X-Locus-Router", "1"}},
        json{{"prompt", "x"}, {"max_tokens", 1}}.dump()});
    CHECK(r.head.status == 508);
}

TEST_CASE("router: a registered instance gets the key that names it",
          "[router]") {
    using locus::router::Registrar;
    FakeInstance a("A", {"text"}, "m");
    RouterConfig cfg;
    cfg.register_keys = {"rk"};
    cfg.upstream_auth = RouterConfig::UpstreamAuth::kKey;
    cfg.instance_keys = {{a.addr(), "sk-a"}, {"*", "sk-wild"}};
    RouterFixture rf({}, cfg);
    Registrar::Config rc;
    rc.router_host = "127.0.0.1";
    rc.router_port = rf.port;
    rc.address = a.addr();
    rc.key = "rk";
    REQUIRE(Registrar(rc).register_once());
    const auto r = rf.post("/v1/completions",
                           json{{"prompt", "x"}, {"max_tokens", 1}});
    REQUIRE(r.ok);
    REQUIRE(r.head.status == 200);
    CHECK(a.header("Authorization") == "Bearer sk-a");
}

// i#524: "mdns:SERVICE" lists the instances a DNS-SD browse finds, on
// every refresh; an empty browse keeps them (lost packets), one the
// browse stops listing drops out, and a found instance gets no "*" key.
// i#537: mdns: and scan: with the client's key forwarded print a
// warning at startup; a fixed instance list or another upstream auth
// does not.
TEST_CASE("router: discovery with forwarded client keys warns",
          "[router]") {
    using UA = RouterConfig::UpstreamAuth;
    RouterConfig cfg;
    cfg.instances = {"127.0.0.1:8081"};
    CHECK(locus::router::discovery_auth_warning(cfg).empty());
    for (const char* src : {"mdns:_locus._tcp", "scan:10.0.0.0/30:8080"}) {
        cfg.instances = {"127.0.0.1:8081", src};
        cfg.upstream_auth = UA::kClient;
        CHECK(locus::router::discovery_auth_warning(cfg).find(
                  "--upstream-auth key") != std::string::npos);
        cfg.upstream_auth = UA::kKey;
        CHECK(locus::router::discovery_auth_warning(cfg).empty());
        cfg.upstream_auth = UA::kNone;
        CHECK(locus::router::discovery_auth_warning(cfg).empty());
    }
}

TEST_CASE("router: an mdns service lists the instances that answer",
          "[router]") {
    using locus::pipeline::SrvTarget;
    using UA = RouterConfig::UpstreamAuth;
    FakeInstance a("A", {"text"}, "m"), b("B", {"text"}, "m");
    std::mutex mu;
    std::vector<SrvTarget> answer = {{"127.0.0.1", a.port, 0, 0},
                                     {"127.0.0.1", b.port, 0, 0}};
    RouterConfig cfg;
    cfg.upstream_auth = UA::kKey;
    cfg.instance_keys = {{"*", "sk-wild"}};
    cfg.mdns_lookup = [&](const std::string& service) {
        CHECK(service == "_locus._tcp");
        std::lock_guard lk(mu);
        return answer;
    };
    RouterFixture rf({"mdns:_locus._tcp"}, cfg);
    const auto reached = [&] {
        std::set<std::string> seen;
        for (int i = 0; i < 20; ++i) {
            const auto r = rf.post(
                "/v1/completions",
                json{{"prompt", "p" + std::to_string(i)}, {"max_tokens", 1}});
            REQUIRE(r.ok);
            REQUIRE(r.head.status == 200);
            seen.insert(r.head.get_header_value("X-Instance"));
        }
        return seen;
    };
    CHECK(reached() == std::set<std::string>{"A", "B"});
    CHECK(a.header("Authorization").empty());  // found: no "*" key
    const auto set = [&](std::vector<SrvTarget> v) {
        {
            std::lock_guard lk(mu);
            answer = std::move(v);
        }
        rf.router->refresh();
    };
    set({});  // no answer this time: keep both
    CHECK(reached() == std::set<std::string>{"A", "B"});
    set({{"127.0.0.1", b.port, 0, 0}});  // A stopped answering
    CHECK(reached() == std::set<std::string>{"B"});
    for (const char* bad : {"mdns:", "mdns:a..b"}) {
        RouterConfig c;
        c.instances = {bad};
        CHECK_THROWS(Router(c));
    }
}

// i#443: the router's default host, "", serves IPv6 and IPv4 clients on
// one socket, as locus-server does (i#139).
TEST_CASE("router: the default host serves IPv4 and IPv6 clients",
          "[router]") {
    FakeInstance a("A", {"text"}, "m");
    RouterConfig cfg;
    REQUIRE(cfg.host.empty());  // the default
    cfg.port = 0;
    cfg.instances = {a.addr()};
    cfg.refresh_ms = 60000;
    Router router(cfg);
    const int port = router.start();
    REQUIRE(port > 0);
    Client v4("127.0.0.1", port);
    const auto r4 = v4.send(ClientRequest{"GET", "/health", {}, ""});
    REQUIRE(r4.ok);
    CHECK(r4.head.status == 200);
    Client v6("::1", port);
    const auto r6 = v6.send(ClientRequest{"GET", "/health", {}, ""});
    if (!r6.connected) {
        WARN("no IPv6 loopback here: the ::1 leg is skipped");
    } else {
        REQUIRE(r6.ok);
        CHECK(r6.head.status == 200);
    }
}

// i#388 step 6: "scan:CIDR:PORT" probes the range's addresses for a
// locus /v1/models; the ones that answer are its instances (found, so
// no "*" key), and a scan that no longer finds one drops it.
TEST_CASE("router: a scan finds the instances on a range", "[router]") {
    using UA = RouterConfig::UpstreamAuth;
    auto a = std::make_unique<FakeInstance>(
        "A", std::vector<std::string>{"text"}, "m");
    FakeInstance b("B", {"text"}, "m");
    RouterConfig cfg;
    cfg.scan_every = 1;  // every refresh, for the test
    cfg.upstream_auth = UA::kKey;
    cfg.instance_keys = {{"*", "sk-wild"}};
    const int a_port = a->port;
    RouterFixture rf({"scan:127.0.0.1/32:" + std::to_string(a_port), b.addr()},
                     cfg);
    std::set<std::string> seen;
    for (int i = 0; i < 20; ++i) {
        const auto r = rf.post(
            "/v1/completions",
            json{{"prompt", "p" + std::to_string(i)}, {"max_tokens", 1}});
        REQUIRE(r.ok);
        REQUIRE(r.head.status == 200);
        seen.insert(r.head.get_header_value("X-Instance"));
    }
    CHECK(seen == std::set<std::string>{"A", "B"});
    CHECK(a->header("Authorization").empty());  // found: no "*" key
    CHECK(b.header("Authorization") == "Bearer sk-wild");

    // A stops answering: the next scan no longer lists it.
    a.reset();
    rf.router->refresh();
    bool gone = false;
    for (const auto& in : rf.router->instances()) {
        if (in.port == a_port) gone = in.gone;
    }
    CHECK(gone);
    const auto r = rf.post("/v1/completions",
                           json{{"prompt", "x"}, {"max_tokens", 1}});
    REQUIRE(r.ok);
    CHECK(r.head.get_header_value("X-Instance") == "B");

    for (const char* bad : {"scan:10.0.0.0/16:80", "scan:::1/128:80",
                            "scan:10.0.0.0/24:0", "scan:10.0.0.0/24:x",
                            "scan:10.0.0.0/24"}) {
        CAPTURE(bad);
        RouterConfig c;
        c.instances = {bad};
        CHECK_THROWS_AS(Router(c), std::invalid_argument);
    }
}

TEST_CASE("router: registration is off without keys", "[router]") {
    FakeInstance b("B", {"text"}, "m");
    RouterFixture rf({b.addr()});
    Client c("127.0.0.1", rf.port);
    const auto r = c.send(ClientRequest{"POST", "/locus/register",
                                        {{"Authorization", "Bearer k1"}},
                                        json{{"address", b.addr()}}.dump()});
    CHECK(r.head.status == 404);
}

TEST_CASE("router: register keys and router URLs parse", "[router]") {
    using locus::router::parse_register_keys;
    using locus::router::parse_router_url;
    CHECK(parse_register_keys("# keys\n k1 \n\nk2\r\n") ==
          std::vector<std::string>{"k1", "k2"});
    CHECK(parse_register_keys("").empty());
    std::string h;
    int p = 0;
    REQUIRE(parse_router_url("http://10.0.0.2:8080", h, p));
    CHECK(h == "10.0.0.2");
    CHECK(p == 8080);
    CHECK(parse_router_url("http://r.lan:80/", h, p));
    CHECK_FALSE(parse_router_url("https://r.lan:80", h, p));
    CHECK_FALSE(parse_router_url("http://r.lan:80/v1", h, p));
    CHECK_FALSE(parse_router_url("r.lan:80", h, p));
}

TEST_CASE("router: video ids name their instance", "[router]") {
    using locus::router::router_video_id;
    using locus::router::split_video_id;
    const std::string rid =
        router_video_id("127.0.0.1:8311", "video_0123abcd");
    CHECK(rid.rfind("video_", 0) == 0);
    CHECK(rid.find('/') == std::string::npos);
    std::string inst, id;
    REQUIRE(split_video_id(rid, inst, id));
    CHECK(inst == "127.0.0.1:8311");
    CHECK(id == "video_0123abcd");
    // An instance's own id, or garbage, is not a router id.
    CHECK_FALSE(split_video_id("video_0123abcd", inst, id));
    CHECK_FALSE(split_video_id("video_zz_1", inst, id));
    CHECK_FALSE(split_video_id("clip_1", inst, id));
    CHECK_FALSE(split_video_id("video_3a__", inst, id));  // no id
    // Any id comes back unchanged, prefixed or not (i#452).
    for (const char* own : {"video_1", "job42", "video_", "video__x",
                            "vid_eo"}) {
        CAPTURE(own);
        REQUIRE(split_video_id(router_video_id("h:1", own), inst, id));
        CHECK(inst == "h:1");
        CHECK(id == own);
    }
    // The short form for locus's own ids is unchanged.
    CHECK(router_video_id("h:1", "video_ab") == "video_683a31_ab");

    FakeInstance v1("V1", {"video"}, "ltx"), v2("V2", {"video"}, "ltx");
    RouterFixture rf({v1.addr(), v2.addr()});
    Client c("127.0.0.1", rf.port);
    std::set<std::string> owners;
    // 20 jobs, so all of them landing on one of the two instances (a
    // 2 x 2^-20 chance) cannot flake the spread check below.
    for (int k = 0; k < 20; ++k) {
        // Different prompts spread the jobs (a prompt is the session
        // key); each id still finds its maker.
        const auto made = rf.post(
            "/v1/videos", {{"prompt", "a fox, take " + std::to_string(k)}});
        REQUIRE(made.ok);
        const std::string vid = json::parse(made.body)["id"];
        REQUIRE(split_video_id(vid, inst, id));
        const auto got =
            c.send(ClientRequest{"GET", "/v1/videos/" + vid, {}, ""});
        REQUIRE(got.ok);
        REQUIRE(got.head.status == 200);
        const json j = json::parse(got.body);
        CHECK(j["id"] == vid);  // the router's id comes back
        owners.insert(j["owner"].get<std::string>());
        const auto mp4 = c.send(
            ClientRequest{"GET", "/v1/videos/" + vid + "/content", {}, ""});
        REQUIRE(mp4.ok);
        CHECK(mp4.body == "MP4 of " + j["owner"].get<std::string>());
    }
    CHECK(owners.size() == 2);
    // A router id for an instance the router does not have: 404.
    const auto none = c.send(ClientRequest{
        "GET", "/v1/videos/" + router_video_id("10.9.9.9:1", "video_1"),
        {}, ""});
    REQUIRE(none.ok);
    CHECK(none.head.status == 404);
}

TEST_CASE("router: metrics per instance", "[router]") {
    FakeInstance a("A", {"text"}, "m");
    RouterFixture rf({a.addr()});
    for (int k = 0; k < 3; ++k) {
        REQUIRE(rf.post("/v1/completions", {{"prompt", "x"}}).ok);
    }
    // A client that leaves a long stream after the first piece.
    {
        Client c("127.0.0.1", rf.port);
        int got = 0;
        c.stream(ClientRequest{"POST", "/v1/chat/completions",
                               {{"Content-Type", "application/json"}},
                               json{{"stream", true}, {"long", true},
                                    {"messages", json::array()}}
                                   .dump()},
                 nullptr, [&](std::string_view) { return ++got < 1; });
    }
    const std::string series =
        "{instance=\"" + a.addr() + "\"} ";
    std::string text;
    // The router notices the leave on its next write to the client.
    for (int tries = 0; tries < 50; ++tries) {
        Client c("127.0.0.1", rf.port);
        text = c.send(ClientRequest{"GET", "/metrics", {}, ""}).body;
        if (text.find("locus_router_client_left_total" + series + "1") !=
            std::string::npos) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    CHECK(text.find("locus_router_instance_up" + series + "1") !=
          std::string::npos);
    CHECK(text.find("locus_router_requests_total" + series + "4") !=
          std::string::npos);
    CHECK(text.find("locus_router_client_left_total" + series + "1") !=
          std::string::npos);
    CHECK(text.find("locus_router_upstream_errors_total" + series + "0") !=
          std::string::npos);
    CHECK(text.find("locus_router_inflight" + series + "0") !=
          std::string::npos);
    CHECK(text.find("# TYPE locus_router_request_seconds_total counter") !=
          std::string::npos);
}

TEST_CASE("router: instance keys file", "[router]") {
    const auto k = locus::router::parse_instance_keys(
        "# keys\n10.0.0.1:8001 sk-a\n\n  10.0.0.2:8001\tsk-b  \r\n* sk-any\n");
    REQUIRE(k.size() == 3);
    CHECK(k.at("10.0.0.1:8001") == "sk-a");
    CHECK(k.at("10.0.0.2:8001") == "sk-b");
    CHECK(k.at("*") == "sk-any");
    CHECK_THROWS(locus::router::parse_instance_keys("10.0.0.1:8001\n"));
    CHECK_THROWS(locus::router::parse_instance_keys("h:1 two words\n"));
}

TEST_CASE("router: client keys checked by the auth helper", "[router]") {
    FakeInstance a("A", {"text"}, "m");
    RouterConfig cfg;
    cfg.auth_helper_argv = {LOCUS_AUTH_HELPER_BIN, "sk-ok"};
    RouterFixture rf({a.addr()}, cfg);
    const auto with = [&](std::vector<std::pair<std::string, std::string>> h) {
        return rf.post("/v1/completions", {{"prompt", "x"}}, std::move(h));
    };
    CHECK(with({}).head.status == 401);
    CHECK(with({{"Authorization", "Bearer sk-bad"}}).head.status == 401);
    const auto ok = with({{"Authorization", "Bearer sk-ok"}});
    CHECK(ok.head.status == 200);
    CHECK(ok.body.rfind("A ", 0) == 0);
    CHECK(with({{"x-api-key", "sk-ok"}}).head.status == 200);
    const auto denied = with({});
    CHECK(json::parse(denied.body)["error"]["type"] ==
          "authentication_error");
    // /health stays open; /v1/models needs a key.
    Client c("127.0.0.1", rf.port);
    CHECK(c.send(ClientRequest{"GET", "/health", {}, ""}).head.status == 200);
    CHECK(c.send(ClientRequest{"GET", "/v1/models", {}, ""}).head.status ==
          401);
}

TEST_CASE("router: the credential sent to the instances", "[router]") {
    FakeInstance a("A", {"text"}, "m"), b("B", {"text"}, "n");
    using UA = RouterConfig::UpstreamAuth;
    const std::vector<std::pair<std::string, std::string>> client_key{
        {"Authorization", "Bearer client-key"}, {"x-api-key", "client-x"}};
    SECTION("none: the client's key does not go on") {
        RouterConfig cfg;
        cfg.upstream_auth = UA::kNone;
        RouterFixture rf({a.addr()}, cfg);
        REQUIRE(rf.post("/v1/completions", {{"prompt", "x"}}, client_key).ok);
        CHECK(a.header("Authorization").empty());
        CHECK(a.header("x-api-key").empty());
    }
    SECTION("key: each instance gets its own, else the * key") {
        RouterConfig cfg;
        cfg.upstream_auth = UA::kKey;
        cfg.instance_keys = {{a.addr(), "sk-for-a"}, {"*", "sk-other"}};
        RouterFixture rf({a.addr(), b.addr()}, cfg);
        REQUIRE(rf.post("/v1/completions", {{"model", "m"}, {"prompt", "x"}},
                        client_key)
                    .ok);
        CHECK(a.header("Authorization") == "Bearer sk-for-a");
        CHECK(a.header("x-api-key").empty());
        REQUIRE(rf.post("/v1/completions", {{"model", "n"}, {"prompt", "x"}},
                        client_key)
                    .ok);
        CHECK(b.header("Authorization") == "Bearer sk-other");
    }
}

TEST_CASE("router: polls an instance that checks keys with its key",
          "[router]") {
    // A real locus-server-style check would 401 a keyless /v1/models;
    // this fake answers /v1/models only with the right key.
    locus::http::ServerConfig sc;
    sc.host = "127.0.0.1";
    sc.port = 0;
    sc.worker_threads = 2;
    locus::http::Server inst(
        sc, [](const locus::http::Request& req,
               locus::http::ServerResponse& res,
               const locus::http::RequestContext&) {
            if (req.get_header_value("Authorization") != "Bearer sk-inst") {
                res.status = 401;
                res.set_content("{}", "application/json");
                return;
            }
            res.set_content(
                R"({"object":"list","data":[{"id":"m"}],"locus_kinds":["text"]})",
                "application/json");
        });
    const int port = inst.start();
    REQUIRE(port > 0);
    const std::string addr = "127.0.0.1:" + std::to_string(port);
    {
        RouterFixture keyless({addr});
        CHECK_FALSE(keyless.router->instances()[0].healthy);
    }
    RouterConfig cfg;
    cfg.instance_keys = {{addr, "sk-inst"}};
    RouterFixture keyed({addr}, cfg);
    CHECK(keyed.router->instances()[0].healthy);
    inst.stop();
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
    std::jthread th([&] { server.listen_after_bind(); });
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

TEST_CASE("router: polls instances in parallel", "[router]") {
    // i#625 F13: two instances that take the connection and never
    // answer each hold their poll for the read timeout (5 s). One after
    // the other, a refresh took 10 s; in parallel it takes one timeout.
    int p1 = 0;
    int p2 = 0;
    const int l1 = locus::pipeline::listen_on("127.0.0.1", 0, &p1);
    const int l2 = locus::pipeline::listen_on("127.0.0.1", 0, &p2);
    REQUIRE(l1 >= 0);
    REQUIRE(l2 >= 0);
    FakeInstance a("A", {"text"}, "m");
    RouterFixture rf({"127.0.0.1:" + std::to_string(p1),
                      "127.0.0.1:" + std::to_string(p2), a.addr()});
    const auto t0 = std::chrono::steady_clock::now();
    rf.router->refresh();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    ::close(l1);
    ::close(l2);
    CHECK(ms < 8000);
    const auto inst = rf.router->instances();
    REQUIRE(inst.size() == 3);
    CHECK_FALSE(inst[0].healthy);
    CHECK_FALSE(inst[1].healthy);
    CHECK(inst[2].healthy);
}
