// locus-router (i#388 step 2): request kinds, picking an instance, and
// proxying buffered and streamed replies, against fake instances
// (locus's own http::Server) and a real locus-server.

#include <csignal>

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
