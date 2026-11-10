#include "locus/router/router.hpp"

#include <algorithm>
#include <chrono>
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

int Router::pick(const Route& route) const {
    std::lock_guard lk(mu_);
    int any = -1;
    for (std::size_t i = 0; i < inst_.size(); ++i) {
        const Instance& in = inst_[i];
        if (!in.healthy ||
            std::find(in.kinds.begin(), in.kinds.end(), route.kind) ==
                in.kinds.end()) {
            continue;
        }
        if (route.model.empty() ||
            std::find(in.models.begin(), in.models.end(), route.model) !=
                in.models.end()) {
            return static_cast<int>(i);
        }
        if (any < 0) any = static_cast<int>(i);
    }
    // A model no instance lists (clients often send their own name for
    // it) goes to any instance of the kind.
    return any;
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
    const int idx = pick(*route);
    if (idx < 0) {
        res.status = 503;
        res.set_header("Retry-After", "5");
        res.set_content(
            error_body("no healthy instance serves " +
                           std::string(kind_name(route->kind)) + " requests",
                       "server_error"),
            "application/json");
        return;
    }
    Instance in;
    {
        std::lock_guard lk(mu_);
        in = inst_[static_cast<std::size_t>(idx)];
    }

    http::ClientRequest up;
    up.method = req.method;
    up.target = req.target;
    up.body = req.body;
    for (const auto& [k, v] : req.headers) {
        if (!hop_by_hop(k)) up.headers.emplace_back(k, v);
    }
    auto client = std::make_shared<http::Client>(in.host, in.port,
                                                 cfg_.upstream);
    const auto head = client->begin(up);
    if (!head.ok) {
        // A connect failure marks the instance down until the next poll.
        {
            std::lock_guard lk(mu_);
            for (auto& i : inst_) {
                if (i.host == in.host && i.port == in.port) i.healthy = false;
            }
        }
        res.status = 502;
        res.set_content(error_body("upstream " + in.host + ":" +
                                       std::to_string(in.port) + ": " +
                                       head.error,
                                   "server_error"),
                        "application/json");
        return;
    }
    res.status = head.head.status;
    std::string content_type = head.head.get_header_value("Content-Type");
    for (const auto& [k, v] : head.head.headers) {
        if (!hop_by_hop(k) && !iequals(k, "Content-Type")) {
            res.set_header(k, v);
        }
    }
    const auto framing = head.head.framing;
    if (framing == http::BodyFraming::kChunked ||
        framing == http::BodyFraming::kClose) {
        // A stream: pass each piece on as it arrives. A client that goes
        // away fails sink.write, which stops read_body and closes the
        // upstream connection, so the instance stops the work.
        res.set_chunked_content_provider(
            content_type, [client](http::Sink& sink) {
                std::string err;
                client->read_body(
                    [&](std::string_view d) { return sink.write(d); }, err);
            });
        return;
    }
    std::string body;
    std::string err;
    if (!client->read_body(
            [&](std::string_view d) {
                body.append(d);
                return true;
            },
            err)) {
        res.status = 502;
        res.set_content(error_body("upstream body: " + err, "server_error"),
                        "application/json");
        return;
    }
    res.set_content(std::move(body), content_type);
}

}  // namespace locus::router
