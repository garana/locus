#include "locus/router/router.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <set>
#include <stdexcept>

#include "json.hpp"
#include "locus/pipeline/net.hpp"

namespace locus::router {

using nlohmann::json;

namespace {

constexpr std::string_view kKindNames[] = {"text", "image-to-text", "image",
                                           "video", "speech"};

char lower(char c) {
    return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
}

bool iequals(std::string_view a, std::string_view b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(),
                      [](char x, char y) { return lower(x) == lower(y); });
}

/** Headers that belong to one connection, not the message (RFC 9110
 * section 7.6.1), plus the ones the client and server writer set. */
bool hop_by_hop(std::string_view name) {
    for (const char* h :
         {"Connection", "Keep-Alive", "Proxy-Connection", "TE", "Trailer",
          "Transfer-Encoding", "Upgrade", "Proxy-Authorization",
          "Proxy-Authenticate", "Host", "Content-Length"}) {
        if (iequals(name, h)) return true;
    }
    return false;
}

/** True if `j` holds, at any depth, an object whose "type" is an image
 * part: OpenAI `image_url`, Anthropic `image`, Responses `input_image`. */
bool has_image_part(const json& j, int depth = 0) {
    if (depth > 8) return false;
    if (j.is_object()) {
        if (const auto t = j.find("type");
            t != j.end() && t->is_string()) {
            const auto& s = t->get_ref<const std::string&>();
            if (s == "image_url" || s == "image" || s == "input_image") {
                return true;
            }
        }
        for (const auto& [k, v] : j.items()) {
            if (has_image_part(v, depth + 1)) return true;
        }
    } else if (j.is_array()) {
        for (const auto& v : j) {
            if (has_image_part(v, depth + 1)) return true;
        }
    }
    return false;
}

std::string error_body(const std::string& msg, const char* type) {
    return json{{"error", {{"message", msg}, {"type", type}}}}.dump();
}

}  // namespace

namespace {

constexpr std::uint64_t kFnvBasis = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

std::uint64_t fnv1a(std::string_view s, std::uint64_t h = kFnvBasis) {
    for (const char c : s) {
        h ^= static_cast<unsigned char>(c);
        h *= kFnvPrime;
    }
    return h;
}

/** The text of a message's content: a string, or the text parts. */
std::string content_text(const json& c) {
    if (c.is_string()) return c.get<std::string>();
    std::string out;
    if (c.is_array()) {
        for (const auto& p : c) {
            if (p.is_object() && p.contains("text") && p["text"].is_string()) {
                out += p["text"].get<std::string>();
            }
        }
    }
    return out;
}

}  // namespace

std::string session_key(std::string_view header, std::string_view body) {
    if (!header.empty()) return "h:" + std::string(header);
    const json j = json::parse(body, nullptr, /*allow_exceptions=*/false);
    if (!j.is_object()) return "";
    if (const auto k = j.find("prompt_cache_key");
        k != j.end() && k->is_string() && !k->get<std::string>().empty()) {
        return "k:" + k->get<std::string>();
    }
    if (const auto m = j.find("metadata"); m != j.end() && m->is_object()) {
        if (const auto u = m->find("user_id");
            u != m->end() && u->is_string() && !u->get<std::string>().empty()) {
            return "u:" + u->get<std::string>();
        }
    }
    // The prompt's start: what every later turn of the conversation
    // sends again. Capped, so a long first turn costs little to hash.
    constexpr std::size_t kMaxStart = 4096;
    std::string start;
    for (const char* key : {"system", "instructions"}) {
        if (const auto s = j.find(key); s != j.end()) {
            start += content_text(*s) + '\n';
        }
    }
    const auto first_user = [&](const json& list) {
        if (!list.is_array()) return;
        for (const auto& m : list) {
            if (!m.is_object()) continue;
            const std::string role = m.value("role", "");
            if (role == "system" || role == "developer") {
                start += content_text(m.value("content", json())) + '\n';
            } else if (role == "user") {
                start += content_text(m.value("content", json()));
                return;
            }
        }
    };
    if (const auto ms = j.find("messages"); ms != j.end()) first_user(*ms);
    if (const auto in = j.find("input"); in != j.end()) {
        if (in->is_string()) start += in->get<std::string>();
        else first_user(*in);
    }
    if (const auto p = j.find("prompt"); p != j.end() && p->is_string()) {
        start += p->get<std::string>();
    }
    if (start.empty()) return "";
    if (start.size() > kMaxStart) start.resize(kMaxStart);
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx",
                  static_cast<unsigned long long>(fnv1a(start)));
    return std::string("p:") + hex;
}

std::uint64_t rendezvous_score(std::string_view key,
                               std::string_view instance) {
    std::uint64_t h = fnv1a(key);
    h ^= 0xff;  // a separator byte, so "ab"+"c" differs from "a"+"bc"
    h *= kFnvPrime;
    h = fnv1a(instance, h);
    // Mix the bits (splitmix64 finalizer): FNV's high bits move little
    // for keys that differ only at the end.
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebull;
    h ^= h >> 31;
    return h;
}

std::string_view kind_name(Kind k) {
    return kKindNames[static_cast<int>(k)];
}

std::optional<Kind> parse_kind(std::string_view name) {
    for (int i = 0; i < 5; ++i) {
        if (kKindNames[i] == name) return static_cast<Kind>(i);
    }
    return std::nullopt;
}

std::optional<Route> classify(std::string_view path, std::string_view body) {
    Route r;
    json j;
    if (!body.empty()) {
        j = json::parse(body, nullptr, /*allow_exceptions=*/false);
        if (j.is_object()) {
            if (const auto m = j.find("model");
                m != j.end() && m->is_string()) {
                r.model = m->get<std::string>();
            }
        }
    }
    if (path == "/v1/completions" || path == "/v1/embeddings") {
        r.kind = Kind::kText;
    } else if (path == "/v1/chat/completions" || path == "/v1/messages" ||
               path == "/v1/responses") {
        // Only the conversation can hold image parts.
        bool image = false;
        if (j.is_object()) {
            for (const char* key : {"messages", "input"}) {
                if (const auto it = j.find(key);
                    it != j.end() && has_image_part(*it)) {
                    image = true;
                }
            }
        }
        r.kind = image ? Kind::kImageToText : Kind::kText;
    } else if (path == "/v1/images/generations" ||
               path == "/v1/images/edits") {
        r.kind = Kind::kImage;
    } else if (path == "/v1/videos" || path.rfind("/v1/videos/", 0) == 0) {
        r.kind = Kind::kVideo;
    } else if (path == "/v1/audio/speech") {
        r.kind = Kind::kSpeech;
    } else {
        return std::nullopt;
    }
    return r;
}

Router::Router(RouterConfig cfg) : cfg_(std::move(cfg)) {
    for (const auto& s : cfg_.instances) {
        Instance in;
        if (!pipeline::parse_hostport(s, in.host, in.port) ||
            in.host.empty() || in.port <= 0) {
            throw std::invalid_argument("bad instance address: " + s);
        }
        inst_.push_back(std::move(in));
        slots_.push_back(std::make_unique<Slot>());
    }
}

std::shared_ptr<http::Client> Router::take(std::size_t i) {
    Slot& s = *slots_[i];
    {
        std::lock_guard lk(s.mu);
        if (!s.idle.empty()) {
            auto c = std::move(s.idle.back());
            s.idle.pop_back();
            return c;
        }
    }
    std::string host;
    int port = 0;
    {
        std::lock_guard lk(mu_);
        host = inst_[i].host;
        port = inst_[i].port;
    }
    return std::make_shared<http::Client>(host, port, cfg_.upstream);
}

void Router::give(std::size_t i, std::shared_ptr<http::Client> c) {
    // A client whose server closed the connection reconnects on its
    // next request, so it is kept either way.
    Slot& s = *slots_[i];
    std::lock_guard lk(s.mu);
    if (static_cast<int>(s.idle.size()) < cfg_.pool_size) {
        s.idle.push_back(std::move(c));
    }
}

Router::~Router() { stop(); }

int Router::start() {
    refresh();
    http::ServerConfig sc;
    sc.host = cfg_.host;
    sc.port = cfg_.port;
    sc.worker_threads = cfg_.worker_threads;
    server_ = std::make_unique<http::Server>(
        sc, [this](const http::Request& req, http::ServerResponse& res,
                   const http::RequestContext&) { handle(req, res); });
    const int port = server_->start();
    if (port < 0) {
        server_.reset();
        return -1;
    }
    refresher_ = std::thread([this] {
        std::unique_lock lk(stop_mu_);
        while (!stopping_) {
            if (stop_cv_.wait_for(lk,
                                  std::chrono::milliseconds(cfg_.refresh_ms),
                                  [this] { return stopping_; })) {
                break;
            }
            lk.unlock();
            refresh();
            lk.lock();
        }
    });
    return port;
}

void Router::stop() {
    {
        std::lock_guard lk(stop_mu_);
        stopping_ = true;
    }
    stop_cv_.notify_all();
    if (refresher_.joinable()) refresher_.join();
    if (server_) server_->stop();
}

void Router::wait() {
    if (server_) server_->wait();
}

void Router::refresh() {
    std::vector<Instance> seen;
    {
        std::lock_guard lk(mu_);
        seen = inst_;
    }
    for (auto& in : seen) {
        http::ClientOptions o;
        o.connect_timeout_ms = 2000;
        o.read_timeout_ms = 5000;
        http::Client c(in.host, in.port, o);
        const auto r = c.send(http::ClientRequest{"GET", "/v1/models", {}, ""});
        const json j = r.ok && r.head.status == 200
                           ? json::parse(r.body, nullptr, false)
                           : json();
        in.healthy = j.is_object() && j.contains("data");
        in.kinds.clear();
        in.models.clear();
        in.models_json.clear();
        if (!in.healthy) continue;
        if (const auto k = j.find("locus_kinds"); k != j.end() && k->is_array()) {
            for (const auto& v : *k) {
                if (v.is_string()) {
                    if (const auto kind = parse_kind(v.get<std::string>())) {
                        in.kinds.push_back(*kind);
                    }
                }
            }
        } else {
            in.kinds.push_back(Kind::kText);  // a server without kinds
        }
        if (j["data"].is_array()) {
            for (const auto& m : j["data"]) {
                if (m.is_object() && m.contains("id") && m["id"].is_string()) {
                    in.models.push_back(m["id"].get<std::string>());
                }
            }
            in.models_json = j["data"].dump();
        }
    }
    std::lock_guard lk(mu_);
    inst_ = std::move(seen);
}

std::vector<Instance> Router::instances() const {
    std::lock_guard lk(mu_);
    return inst_;
}

std::vector<int> Router::candidates(const Route& route) const {
    std::vector<int> of_kind, of_model;
    {
        std::lock_guard lk(mu_);
        for (std::size_t i = 0; i < inst_.size(); ++i) {
            const Instance& in = inst_[i];
            if (!in.healthy ||
                std::find(in.kinds.begin(), in.kinds.end(), route.kind) ==
                    in.kinds.end()) {
                continue;
            }
            of_kind.push_back(static_cast<int>(i));
            if (!route.model.empty() &&
                std::find(in.models.begin(), in.models.end(), route.model) !=
                    in.models.end()) {
                of_model.push_back(static_cast<int>(i));
            }
        }
    }
    // A model no instance lists (clients often send their own name for
    // it) goes to any instance of the kind.
    std::vector<int> c = of_model.empty() ? of_kind : of_model;
    if (c.empty()) return c;
    if (!route.session.empty()) {
        // Rendezvous hashing: every router orders the same instances
        // the same way for a key, with no table to share.
        std::vector<std::pair<std::uint64_t, int>> scored;
        {
            std::lock_guard lk(mu_);
            for (const int i : c) {
                const Instance& in = inst_[static_cast<std::size_t>(i)];
                scored.emplace_back(
                    rendezvous_score(route.session,
                                     in.host + ":" + std::to_string(in.port)),
                    i);
            }
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        for (std::size_t k = 0; k < c.size(); ++k) c[k] = scored[k].second;
    } else {
        const std::size_t r = rr_.fetch_add(1) % c.size();
        std::rotate(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(r),
                    c.end());
    }
    // Bounded load: an instance at or over load_factor times the
    // average in-flight count (counting this request) goes last.
    int total = 0;
    for (const int i : c) {
        total += slots_[static_cast<std::size_t>(i)]->inflight.load();
    }
    const int cap = static_cast<int>(std::ceil(
        cfg_.load_factor * (total + 1) / static_cast<double>(c.size())));
    std::stable_partition(c.begin(), c.end(), [&](int i) {
        return slots_[static_cast<std::size_t>(i)]->inflight.load() < cap;
    });
    return c;
}

void Router::models(http::ServerResponse& res) const {
    json data = json::array();
    std::set<std::string> ids;
    std::set<int> kinds;
    {
        std::lock_guard lk(mu_);
        for (const auto& in : inst_) {
            if (!in.healthy) continue;
            for (const Kind k : in.kinds) kinds.insert(static_cast<int>(k));
            const json d = json::parse(in.models_json, nullptr, false);
            if (!d.is_array()) continue;
            for (const auto& m : d) {
                if (m.is_object() && m.contains("id") && m["id"].is_string() &&
                    ids.insert(m["id"].get<std::string>()).second) {
                    data.push_back(m);
                }
            }
        }
    }
    json k = json::array();
    for (const int i : kinds) k.push_back(kind_name(static_cast<Kind>(i)));
    res.set_content(
        json{{"object", "list"}, {"data", data}, {"locus_kinds", k}}.dump(),
        "application/json");
}

void Router::handle(const http::Request& req, http::ServerResponse& res) {
    if (req.path == "/health") {
        res.set_content(R"({"status":"ok"})", "application/json");
        return;
    }
    if (req.path == "/v1/models") {
        models(res);
        return;
    }
    const auto route = classify(req.path, req.body);
    if (!route) {
        res.status = 404;
        res.set_content(error_body("no route for " + req.path,
                                   "invalid_request_error"),
                        "application/json");
        return;
    }
    Route rt = *route;
    rt.session = session_key(req.get_header_value("X-Session-Id"), req.body);
    const std::vector<int> cands = candidates(rt);
    if (cands.empty()) {
        res.status = 503;
        res.set_header("Retry-After", "5");
        res.set_content(
            error_body("no healthy instance serves " +
                           std::string(kind_name(rt.kind)) + " requests",
                       "server_error"),
            "application/json");
        return;
    }

    http::ClientRequest up;
    up.method = req.method;
    up.target = req.target;
    up.body = req.body;
    for (const auto& [k, v] : req.headers) {
        if (!hop_by_hop(k)) up.headers.emplace_back(k, v);
    }
    const auto copy_head = [&](const http::ResponseHead& h) {
        res.status = h.status;
        for (const auto& [k, v] : h.headers) {
            if (!hop_by_hop(k) && !iequals(k, "Content-Type")) {
                res.set_header(k, v);
            }
        }
        return h.get_header_value("Content-Type");
    };
    // Try the candidates in order: a connect failure, or a 503 that
    // says when to retry (the instance is busy), moves to the next.
    std::string last_error;
    bool busy = false;
    std::string busy_type, busy_body;
    http::ResponseHead busy_head;
    for (const int ci : cands) {
        const auto i = static_cast<std::size_t>(ci);
        Slot& slot = *slots_[i];
        ++slot.inflight;
        // Released when the request is done here, or when the stream
        // provider that holds it is gone.
        std::shared_ptr<void> guard(nullptr,
                                    [&slot](void*) { --slot.inflight; });
        std::shared_ptr<http::Client> client = take(i);
        const auto head = client->begin(up);
        if (!head.ok) {
            // A refused or timed-out connect means the instance is down
            // until the next poll answers; any other failure (a reply
            // cut short, a bad head) only moves on to the next one.
            std::lock_guard lk(mu_);
            if (!head.connected) inst_[i].healthy = false;
            last_error = inst_[i].host + ":" + std::to_string(inst_[i].port) +
                         ": " + head.error;
            continue;
        }
        const auto framing = head.head.framing;
        const bool stream = framing == http::BodyFraming::kChunked ||
                            framing == http::BodyFraming::kClose;
        if (head.head.status == 503 &&
            !head.head.get_header_value("Retry-After").empty() && !stream) {
            std::string b, err;
            if (client->read_body(
                    [&](std::string_view d) {
                        b.append(d);
                        return true;
                    },
                    err)) {
                give(i, std::move(client));
            }
            busy = true;
            busy_head = head.head;
            busy_type = head.head.get_header_value("Content-Type");
            busy_body = std::move(b);
            continue;
        }
        const std::string content_type = copy_head(head.head);
        if (stream) {
            // A stream: pass each piece on as it arrives. A client that
            // goes away fails sink.write, which stops read_body and
            // closes the upstream connection, so the instance stops the
            // work.
            res.set_chunked_content_provider(
                content_type,
                [this, i, client, guard](http::Sink& sink) mutable {
                    std::string err;
                    if (client->read_body(
                            [&](std::string_view d) { return sink.write(d); },
                            err)) {
                        give(i, std::move(client));
                    }
                });
            return;
        }
        std::string body, err;
        if (!client->read_body(
                [&](std::string_view d) {
                    body.append(d);
                    return true;
                },
                err)) {
            res.status = 502;
            res.headers.clear();
            res.set_content(error_body("upstream body: " + err, "server_error"),
                            "application/json");
            return;
        }
        give(i, std::move(client));
        res.set_content(std::move(body), content_type);
        return;
    }
    if (busy) {
        // Every candidate was busy: pass the last 503 on, Retry-After
        // and all.
        copy_head(busy_head);
        res.set_content(std::move(busy_body), busy_type);
        return;
    }
    res.status = 502;
    res.set_content(error_body("upstream " + last_error, "server_error"),
                    "application/json");
}

}  // namespace locus::router
