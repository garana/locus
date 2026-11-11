#include "locus/router/router.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <set>
#include <stdexcept>

#include "json.hpp"
#include "locus/auth/helper_spawn.hpp"
#include "locus/pipeline/net.hpp"

namespace locus::router {

using nlohmann::json;

namespace {

constexpr std::string_view kKindNames[] = {
    "text", "image-to-text", "image", "video", "speech", "rerank", "classify"};

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

std::string router_video_id(std::string_view instance, std::string_view id) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out = "video_";
    for (const char c : instance) {
        const auto u = static_cast<unsigned char>(c);
        out += kHex[u >> 4];
        out += kHex[u & 15];
    }
    out += '_';
    // "video_X" (X not starting with '_') keeps the short form, X;
    // any other id goes whole after a second '_', so split_video_id
    // gives every id back unchanged.
    if (id.rfind("video_", 0) == 0 && id.size() > 6 && id[6] != '_') {
        out += id.substr(6);
    } else {
        out += '_';
        out += id;
    }
    return out;
}

bool split_video_id(std::string_view rid, std::string& instance,
                    std::string& id) {
    if (rid.rfind("video_", 0) != 0) return false;
    rid.remove_prefix(6);
    const std::size_t sep = rid.find('_');
    if (sep == std::string_view::npos || sep == 0 || sep % 2 != 0 ||
        sep + 1 == rid.size()) {
        return false;
    }
    const auto nib = [](char c) {
        return c >= '0' && c <= '9'   ? c - '0'
               : c >= 'a' && c <= 'f' ? c - 'a' + 10
                                      : -1;
    };
    std::string inst;
    for (std::size_t k = 0; k < sep; k += 2) {
        const int hi = nib(rid[k]), lo = nib(rid[k + 1]);
        if (hi < 0 || lo < 0) return false;
        inst += static_cast<char>(hi * 16 + lo);
    }
    const std::string_view rest = rid.substr(sep + 1);
    if (rest[0] == '_') {
        if (rest.size() == 1) return false;  // no id after "__"
        id = std::string(rest.substr(1));  // the instance's id, whole
    } else {
        id = "video_" + std::string(rest);
    }
    instance = std::move(inst);
    return true;
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

std::string discovery_auth_warning(const RouterConfig& cfg) {
    if (cfg.upstream_auth != RouterConfig::UpstreamAuth::kClient) return "";
    const bool found = std::any_of(
        cfg.instances.begin(), cfg.instances.end(), [](const std::string& s) {
            return s.starts_with("mdns:") || s.starts_with("scan:");
        });
    if (!found) return "";
    return "warning: with --upstream-auth client, every client's key goes "
           "to the instances that mdns: and scan: find, so any host that "
           "answers on those networks receives the keys; prefer "
           "--upstream-auth key with --instance-keys, or none";
}

std::optional<Kind> parse_kind(std::string_view name) {
    for (std::size_t i = 0; i < std::size(kKindNames); ++i) {
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
        r.embeddings = path == "/v1/embeddings";
    } else if (path == "/v1/classify") {
        r.kind = Kind::kClassify;
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
    } else if (path == "/v1/rerank" || path == "/rerank") {
        r.kind = Kind::kRerank;
    } else {
        return std::nullopt;
    }
    return r;
}

Router::Router(RouterConfig cfg)
    : cfg_(std::move(cfg)), slots_(std::make_unique<Slot[]>(kMaxInstances)) {
    for (const auto& s : cfg_.instances) {
        if (s.rfind("srv:", 0) == 0) {
            if (s.size() == 4) {
                throw std::invalid_argument("srv: needs a DNS name");
            }
            srv_names_.push_back(s.substr(4));
            continue;
        }
        if (s.rfind("mdns:", 0) == 0) {
            // Validates the service name now (mdns_query throws).
            pipeline::mdns_query(s.substr(5));
            mdns_services_.push_back(s.substr(5));
            continue;
        }
        if (s.rfind("scan:", 0) == 0) {
            // scan:CIDR:PORT, IPv4, at most 1024 addresses.
            const std::size_t colon = s.rfind(':');
            const auto cidr = colon > 5 ? pipeline::Cidr::parse(
                                              s.substr(5, colon - 5))
                                        : std::nullopt;
            int port = 0;
            const std::string p = s.substr(colon + 1);
            const bool digits =
                !p.empty() && p.size() <= 5 &&
                std::all_of(p.begin(), p.end(), [](char c) {
                    return c >= '0' && c <= '9';
                });
            if (digits) port = std::stoi(p);
            if (!cidr || cidr->family != pipeline::AddrFamily::kV4 ||
                cidr->bits < 22 || port <= 0 || port > 65535) {
                throw std::invalid_argument(
                    "bad scan entry " + s +
                    ": scan:A.B.C.D/N:PORT, IPv4, N from 22 to 32");
            }
            const std::uint32_t net = (std::uint32_t{cidr->net[0]} << 24) |
                                      (std::uint32_t{cidr->net[1]} << 16) |
                                      (std::uint32_t{cidr->net[2]} << 8) |
                                      cidr->net[3];
            const std::uint32_t size = 1u << (32 - cidr->bits);
            // A /31 or /32 has no network and broadcast addresses to skip.
            const bool ends = cidr->bits < 31;
            scans_.push_back({s, net + (ends ? 1u : 0u),
                              size - (ends ? 2u : 0u), port});
            continue;
        }
        Instance in;
        if (!pipeline::parse_hostport(s, in.host, in.port) ||
            in.host.empty() || in.port <= 0) {
            throw std::invalid_argument("bad instance address: " + s);
        }
        if (inst_.size() == kMaxInstances) {
            throw std::invalid_argument("more than " +
                                        std::to_string(kMaxInstances) +
                                        " instances");
        }
        inst_.push_back(std::move(in));
    }
    if (!cfg_.auth_helper_argv.empty()) {
        std::vector<std::unique_ptr<auth::HelperConnection>> conns;
        for (std::size_t i = 0; i < std::max<std::size_t>(1, cfg_.auth_helpers);
             ++i) {
            std::string err;
            auto c = auth::spawn_helper(cfg_.auth_helper_argv, &err);
            if (!c) {
                throw std::runtime_error("auth helper spawn: " + err);
            }
            conns.push_back(std::move(c));
        }
        auth_ = std::make_unique<auth::AuthClient>(std::move(conns), clock_,
                                                   cfg_.auth_cache);
    }
}

std::map<std::string, std::string> parse_instance_keys(std::string_view text) {
    std::map<std::string, std::string> keys;
    std::size_t line_no = 0;
    while (!text.empty()) {
        const std::size_t nl = text.find('\n');
        std::string_view line = text.substr(0, nl);
        text.remove_prefix(nl == std::string_view::npos ? text.size() : nl + 1);
        ++line_no;
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' ||
                                 line.back() == '\t')) {
            line.remove_suffix(1);
        }
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) {
            line.remove_prefix(1);
        }
        if (line.empty() || line.front() == '#') continue;
        const std::size_t sp = line.find_first_of(" \t");
        std::string_view key =
            sp == std::string_view::npos ? std::string_view{} : line.substr(sp);
        while (!key.empty() && (key.front() == ' ' || key.front() == '\t')) {
            key.remove_prefix(1);
        }
        if (sp == std::string_view::npos || key.empty() ||
            key.find_first_of(" \t") != std::string_view::npos) {
            throw std::invalid_argument("instance keys, line " +
                                        std::to_string(line_no) +
                                        ": want \"host:port key\"");
        }
        keys[std::string(line.substr(0, sp))] = std::string(key);
    }
    return keys;
}

bool Router::reject(const http::Request& head, http::ServerResponse& res) {
    // Registration checks its own keys (register_keys).
    if (!auth_ || head.path == "/health" || head.path == "/locus/register" ||
        head.path == "/locus/deregister") {
        return false;
    }
    // "Authorization: Bearer <key>" or "x-api-key: <key>", as
    // locus-server takes them.
    std::string cred;
    if (head.has_header("Authorization")) {
        const std::string h = head.get_header_value("Authorization");
        cred = h.rfind("Bearer ", 0) == 0 ? h.substr(7) : h;
    } else if (head.has_header("x-api-key")) {
        cred = head.get_header_value("x-api-key");
    }
    if (!cred.empty() && auth_->resolve(cred).ok) return false;
    res.status = 401;
    res.set_content(error_body("invalid or missing credential",
                               "authentication_error"),
                    "application/json");
    return true;
}

const std::string* Router::instance_key(const Instance& in) const {
    auto it = cfg_.instance_keys.find(in.host + ":" + std::to_string(in.port));
    if (it == cfg_.instance_keys.end() && !in.from_srv && !in.from_reg) {
        it = cfg_.instance_keys.find("*");
    }
    return it == cfg_.instance_keys.end() ? nullptr : &it->second;
}

std::vector<std::pair<std::string, std::string>> Router::upstream_headers(
    const http::Request& req, const Instance& in) const {
    using UA = RouterConfig::UpstreamAuth;
    std::vector<std::pair<std::string, std::string>> h;
    for (const auto& [k, v] : req.headers) {
        if (hop_by_hop(k)) continue;
        // The client's credential goes on only in the kClient mode.
        if (cfg_.upstream_auth != UA::kClient &&
            (iequals(k, "Authorization") || iequals(k, "x-api-key"))) {
            continue;
        }
        h.emplace_back(k, v);
    }
    if (cfg_.upstream_auth == UA::kKey) {
        if (const std::string* key = instance_key(in)) {
            h.emplace_back("Authorization", "Bearer " + *key);
        }
    }
    // A router that receives this refuses the request (handle): an
    // instance registered at a router's own address, or a router
    // behind another, would otherwise send it round until a timeout.
    h.emplace_back("X-Locus-Router", "1");
    return h;
}

std::shared_ptr<http::Client> Router::take(std::size_t i) {
    Slot& s = slots_[i];
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
    Slot& s = slots_[i];
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
        sc,
        [this](const http::Request& req, http::ServerResponse& res,
               const http::RequestContext&) { handle(req, res); },
        // The key check on the head, before the body is read.
        [this](const http::Request& head, http::ServerResponse& res,
               http::RequestContext&) { return reject(head, res); });
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

namespace {

/** @returns Whether a found (not static) instance is listed now: by an
 *  SRV name, or by a registration that has not run out. */
bool listed_now(const Instance& in, std::chrono::steady_clock::time_point now) {
    return !in.srv.empty() || (in.from_reg && in.reg_until > now);
}

/** @returns Whether `in` was found (by SRV or registration) and is not
 *  listed any more, so it must not be picked. */
bool lapsed(const Instance& in, std::chrono::steady_clock::time_point now) {
    return (in.from_srv || in.from_reg) && !listed_now(in, now);
}

}  // namespace

void Router::expand_srv() {
    for (const auto& name : srv_names_) {
        auto all = cfg_.srv_lookup ? cfg_.srv_lookup(name)
                                   : pipeline::dns_srv_lookup(name);
        if (all.empty()) continue;  // keep what the name listed last
        // RFC 2782: the lowest priority is used while it answers; the
        // others are fallbacks. Weights are left to the router's own
        // spreading (rendezvous and the load bound).
        int lo = all.front().priority;
        for (const auto& t : all) lo = std::min(lo, t.priority);
        std::vector<std::pair<std::string, int>> picks;
        for (const auto& t : all) {
            if (t.priority == lo) picks.emplace_back(t.host, t.port);
        }
        merge_listing(name, picks);
    }
}

void Router::expand_mdns() {
    for (const auto& service : mdns_services_) {
        auto all = cfg_.mdns_lookup ? cfg_.mdns_lookup(service)
                                    : pipeline::mdns_browse(service, cfg_.mdns);
        if (all.empty()) continue;  // keep what it listed last
        std::vector<std::pair<std::string, int>> picks;
        for (const auto& t : all) picks.emplace_back(t.host, t.port);
        merge_listing("mdns:" + service, picks);
    }
}

void Router::merge_listing(
    const std::string& name,
    const std::vector<std::pair<std::string, int>>& targets) {
    const auto listed = [&](const std::string& host, int port) {
        return std::any_of(targets.begin(), targets.end(), [&](const auto& t) {
            return t.first == host && t.second == port;
        });
    };
    std::lock_guard lk(mu_);
    const auto now = std::chrono::steady_clock::now();
    // One entry per host:port: this source's targets add or drop it
    // from each entry's names (an entry that registered itself counts
    // as found too), and an entry neither a name nor a registration
    // lists any more is gone (i#492).
    for (auto& in : inst_) {
        if (!in.from_srv && !in.from_reg) continue;  // static
        auto it = std::lower_bound(in.srv.begin(), in.srv.end(), name);
        const bool had = it != in.srv.end() && *it == name;
        if (listed(in.host, in.port)) {
            if (!had) in.srv.insert(it, name);
            in.from_srv = true;
        } else if (had) {
            in.srv.erase(it);
        }
        in.gone = lapsed(in, now);
        if (in.gone) in.healthy = false;
    }
    for (const auto& [host, port] : targets) {
        const bool known = std::any_of(
            inst_.begin(), inst_.end(), [&](const Instance& in) {
                return in.host == host && in.port == port;
            });
        if (known) continue;  // a static entry, or one listed above
        if (inst_.size() == kMaxInstances) {
            std::fprintf(stderr,
                         "locus-router: %s lists more than %zu "
                         "instances; %s:%d left out\n",
                         name.c_str(), kMaxInstances, host.c_str(), port);
            continue;
        }
        Instance in;
        in.host = host;
        in.port = port;
        in.from_srv = true;
        in.srv = {name};
        inst_.push_back(std::move(in));
    }
}

void Router::scan() {
    for (const auto& sc : scans_) {
        // Probe every address with a short connect timeout, 64 at a
        // time: most addresses of a range answer nothing, and a scan
        // must not hold up the refresh for minutes.
        std::vector<char> hit(sc.count, 0);
        std::atomic<std::uint32_t> next{0};
        const auto worker = [&] {
            for (std::uint32_t i; (i = next.fetch_add(1)) < sc.count;) {
                const std::uint32_t a = sc.first + i;
                Instance probe;
                probe.host = std::to_string(a >> 24) + "." +
                             std::to_string((a >> 16) & 255) + "." +
                             std::to_string((a >> 8) & 255) + "." +
                             std::to_string(a & 255);
                probe.port = sc.port;
                probe.from_srv = true;  // found: no "*" key (instance_key)
                http::ClientOptions o;
                o.connect_timeout_ms = 300;
                o.read_timeout_ms = 2000;
                http::Client c(probe.host, probe.port, o);
                http::ClientRequest req{"GET", "/v1/models", {}, ""};
                if (const std::string* key = instance_key(probe)) {
                    req.headers.emplace_back("Authorization", "Bearer " + *key);
                }
                const auto r = c.send(req);
                if (r.ok && r.head.status == 200) {
                    const json j = json::parse(r.body, nullptr, false);
                    hit[i] = j.is_object() && j.contains("data");
                }
            }
        };
        std::vector<std::thread> pool;
        const std::uint32_t threads = std::min<std::uint32_t>(64, sc.count);
        for (std::uint32_t t = 0; t < threads; ++t) pool.emplace_back(worker);
        for (auto& t : pool) t.join();
        std::vector<std::pair<std::string, int>> found;
        for (std::uint32_t i = 0; i < sc.count; ++i) {
            if (!hit[i]) continue;
            const std::uint32_t a = sc.first + i;
            found.emplace_back(std::to_string(a >> 24) + "." +
                                   std::to_string((a >> 16) & 255) + "." +
                                   std::to_string((a >> 8) & 255) + "." +
                                   std::to_string(a & 255),
                               sc.port);
        }
        // An empty scan is an answer (nothing listens there now), not
        // a lookup failure, so it drops the entry's instances.
        merge_listing(sc.name, found);
    }
}

void Router::poll(Instance& in) const {
    http::ClientOptions o;
    o.connect_timeout_ms = 2000;
    o.read_timeout_ms = 5000;
    http::Client c(in.host, in.port, o);
    // An instance that checks keys needs one for its /v1/models too:
    // the instance's key from instance_keys, in any upstream mode.
    http::ClientRequest poll{"GET", "/v1/models", {}, ""};
    if (const std::string* key = instance_key(in)) {
        poll.headers.emplace_back("Authorization", "Bearer " + *key);
    }
    const auto r = c.send(poll);
    const json j = r.ok && r.head.status == 200
                       ? json::parse(r.body, nullptr, false)
                       : json();
    in.healthy = j.is_object() && j.contains("data");
    in.kinds.clear();
    in.models.clear();
    in.models_json.clear();
    if (!in.healthy) return;
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

void Router::refresh() {
    expand_srv();
    expand_mdns();
    if (!scans_.empty() &&
        refreshes_.fetch_add(1) % std::max(1, cfg_.scan_every) == 0) {
        scan();
    }
    std::vector<Instance> seen;
    {
        std::lock_guard lk(mu_);
        // A registration that ran out since the last refresh.
        const auto now = std::chrono::steady_clock::now();
        for (auto& in : inst_) {
            if (lapsed(in, now)) {
                in.gone = true;
                in.healthy = false;
            }
        }
        seen = inst_;
    }
    for (auto& in : seen) {
        if (!in.gone) poll(in);
    }
    // Field by field and by index: inst_ may have grown meanwhile (it
    // only grows), and a name may have dropped one of these since.
    std::lock_guard lk(mu_);
    for (std::size_t i = 0; i < seen.size(); ++i) {
        Instance& in = inst_[i];
        if (in.gone || seen[i].gone) continue;
        in.healthy = seen[i].healthy;
        in.kinds = std::move(seen[i].kinds);
        in.models = std::move(seen[i].models);
        in.models_json = std::move(seen[i].models_json);
    }
}

std::vector<Instance> Router::instances() const {
    std::lock_guard lk(mu_);
    return inst_;
}

std::vector<int> Router::candidates(const Route& route) const {
    std::vector<int> of_kind, of_model;
    {
        std::lock_guard lk(mu_);
        // A registration can run out between refreshes: checked here
        // too, so a lapsed instance gets no request meanwhile.
        const auto now = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < inst_.size(); ++i) {
            const Instance& in = inst_[i];
            if (in.gone || lapsed(in, now) || !in.healthy) {
                continue;
            }
            const auto has = [&](Kind k) {
                return std::find(in.kinds.begin(), in.kinds.end(), k) !=
                       in.kinds.end();
            };
            const bool listed =
                !route.model.empty() &&
                std::find(in.models.begin(), in.models.end(), route.model) !=
                    in.models.end();
            // Image embeddings: a classify instance serves
            // /v1/embeddings under its model's name.
            const bool of = has(route.kind);
            if (!of && !(route.embeddings && listed && has(Kind::kClassify))) {
                continue;
            }
            if (of) of_kind.push_back(static_cast<int>(i));
            if (listed) of_model.push_back(static_cast<int>(i));
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
        total += slots_[static_cast<std::size_t>(i)].inflight.load();
    }
    const int cap = static_cast<int>(std::ceil(
        cfg_.load_factor * (total + 1) / static_cast<double>(c.size())));
    std::stable_partition(c.begin(), c.end(), [&](int i) {
        return slots_[static_cast<std::size_t>(i)].inflight.load() < cap;
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

void Router::metrics(http::ServerResponse& res) const {
    // Prometheus text: one series per instance for each family.
    std::vector<std::pair<std::string, bool>> inst;
    {
        std::lock_guard lk(mu_);
        for (const auto& in : inst_) {
            inst.emplace_back(in.host + ":" + std::to_string(in.port),
                              in.healthy);
        }
    }
    std::string out;
    const auto family = [&](const char* name, const char* help,
                            const char* type, const auto& value) {
        out += std::string("# HELP ") + name + " " + help + "\n";
        out += std::string("# TYPE ") + name + " " + type + "\n";
        for (std::size_t i = 0; i < inst.size(); ++i) {
            out += std::string(name) + "{instance=\"" + inst[i].first +
                   "\"} " + value(i) + "\n";
        }
    };
    const auto u = [](std::uint64_t v) { return std::to_string(v); };
    family("locus_router_instance_up",
           "1 when the instance answered its last /v1/models poll.", "gauge",
           [&](std::size_t i) { return u(inst[i].second ? 1 : 0); });
    family("locus_router_requests_total",
           "Requests the instance answered through the router.", "counter",
           [&](std::size_t i) { return u(slots_[i].requests.load()); });
    family("locus_router_upstream_errors_total",
           "Requests that failed toward the instance (connect, a reply cut "
           "short).",
           "counter",
           [&](std::size_t i) { return u(slots_[i].errors.load()); });
    family("locus_router_client_left_total",
           "Streams the client left before the end; the upstream "
           "connection was closed, so the instance stopped the work.",
           "counter",
           [&](std::size_t i) { return u(slots_[i].client_left.load()); });
    family("locus_router_request_seconds_total",
           "Wall time of the requests sent to the instance.", "counter",
           [&](std::size_t i) {
               char buf[32];
               std::snprintf(buf, sizeof buf, "%.6f",
                             static_cast<double>(slots_[i].busy_ns.load()) /
                                 1e9);
               return std::string(buf);
           });
    family("locus_router_inflight",
           "Requests in flight to the instance now.", "gauge",
           [&](std::size_t i) {
               return u(static_cast<std::uint64_t>(
                   std::max(0, slots_[i].inflight.load())));
           });
    res.set_content(std::move(out), "text/plain; version=0.0.4");
}

namespace {

/** @returns a == b, in a time that does not depend on where they
 *  differ (a registration key check). */
bool same_key(std::string_view a, std::string_view b) {
    unsigned char d = a.size() == b.size() ? 0 : 1;
    const std::size_t n = std::max(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        const unsigned char x = i < a.size() ? a[i] : 0;
        const unsigned char y = i < b.size() ? b[i] : 0;
        d |= x ^ y;
    }
    return d == 0;
}

}  // namespace

void Router::registration(const http::Request& req,
                          http::ServerResponse& res) {
    const auto fail = [&](int status, const std::string& msg,
                          const char* type) {
        res.status = status;
        res.set_content(error_body(msg, type), "application/json");
    };
    if (cfg_.register_keys.empty()) {
        fail(404, "registration is off (no --register-keys)",
             "invalid_request_error");
        return;
    }
    const std::string h = req.get_header_value("Authorization");
    const std::string cred = h.rfind("Bearer ", 0) == 0 ? h.substr(7) : "";
    bool ok = false;
    for (const auto& k : cfg_.register_keys) {
        ok |= !cred.empty() && same_key(cred, k);
    }
    if (!ok) {
        fail(401, "invalid or missing registration key",
             "authentication_error");
        return;
    }
    if (req.method != "POST") {
        fail(405, "use POST", "invalid_request_error");
        return;
    }
    const bool remove = req.path == "/locus/deregister";
    const json body = json::parse(req.body, nullptr, false);
    Instance want;
    if (!body.is_object() || !body.contains("address") ||
        !body["address"].is_string() ||
        !pipeline::parse_hostport(body["address"].get<std::string>(),
                                  want.host, want.port) ||
        want.host.empty() || want.port <= 0) {
        fail(400, "address must be HOST:PORT", "invalid_request_error");
        return;
    }
    int ttl = 30;
    if (body.contains("ttl_s")) {
        const json& t = body["ttl_s"];
        if (!t.is_number_integer() || t.get<int>() < 1 || t.get<int>() > 300) {
            fail(400, "ttl_s must be an integer from 1 to 300",
                 "invalid_request_error");
            return;
        }
        ttl = body["ttl_s"].get<int>();
    }
    const auto now = std::chrono::steady_clock::now();
    std::size_t idx = 0;
    bool probe = false;  // a new or returning instance: poll it now
    std::string status = "registered";
    {
        std::lock_guard lk(mu_);
        auto it = std::find_if(inst_.begin(), inst_.end(), [&](const auto& in) {
            return in.host == want.host && in.port == want.port;
        });
        if (remove) {
            if (it != inst_.end() && it->from_reg) {
                // Expire the registration; from_reg stays, so the
                // entry is still a found one (gone unless a name lists
                // it), not a static one.
                it->reg_until = {};
                it->gone = lapsed(*it, now);
                if (it->gone) it->healthy = false;
            }
            status = "removed";
        } else if (it != inst_.end() && !it->from_srv && !it->from_reg) {
            status = "static";  // given as host:port: always listed
        } else if (it != inst_.end()) {
            probe = it->gone || !it->healthy;
            it->from_reg = true;
            it->reg_until = now + std::chrono::seconds(ttl);
            it->gone = false;
            idx = static_cast<std::size_t>(it - inst_.begin());
        } else if (inst_.size() == kMaxInstances) {
            fail(503, "the router keeps at most " +
                          std::to_string(kMaxInstances) + " instances",
                 "server_error");
            return;
        } else {
            want.from_reg = true;
            want.reg_until = now + std::chrono::seconds(ttl);
            inst_.push_back(want);
            idx = inst_.size() - 1;
            probe = true;
        }
    }
    // A new or returning instance takes requests once it answers its
    // /v1/models, so poll it now rather than at the next refresh; a
    // heartbeat of a healthy one does not poll.
    if (probe) {
        Instance seen;
        {
            std::lock_guard lk(mu_);
            seen = inst_[idx];
        }
        poll(seen);
        std::lock_guard lk(mu_);
        Instance& in = inst_[idx];
        if (!in.gone) {
            in.healthy = seen.healthy;
            in.kinds = std::move(seen.kinds);
            in.models = std::move(seen.models);
            in.models_json = std::move(seen.models_json);
        }
    }
    const std::string addr = want.host + ":" + std::to_string(want.port);
    res.set_content(
        json{{"address", addr}, {"status", status}, {"ttl_s", ttl}}.dump(),
        "application/json");
}

void Router::handle(const http::Request& req, http::ServerResponse& res) {
    if (req.path == "/health") {
        res.set_content(R"({"status":"ok"})", "application/json");
        return;
    }
    // A request a router sent: this router is one of its instances (a
    // registration at a router's address, or routers chained), so
    // proxying it would loop until a timeout.
    if (req.has_header("X-Locus-Router")) {
        res.status = 508;
        res.set_content(error_body("loop: this request came from a "
                                   "locus-router, and a router does not "
                                   "proxy to a router",
                                   "invalid_request_error"),
                        "application/json");
        return;
    }
    if (req.path == "/locus/register" || req.path == "/locus/deregister") {
        registration(req, res);
        return;
    }
    if (req.path == "/v1/models") {
        models(res);
        return;
    }
    if (req.path == "/metrics") {
        metrics(res);
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
    std::string target = req.target;
    std::vector<int> cands;
    // GET /v1/videos/{id}[/content]: the id names the instance that made
    // the job (router_video_id), so only that one can answer.
    constexpr std::string_view kVideos = "/v1/videos/";
    if (rt.kind == Kind::kVideo && req.path.rfind(kVideos, 0) == 0) {
        std::string_view rest(req.path);
        rest.remove_prefix(kVideos.size());
        const std::string rid(rest.substr(0, rest.find('/')));
        std::string inst, id;
        if (split_video_id(rid, inst, id)) {
            {
                std::lock_guard lk(mu_);
                for (std::size_t i = 0; i < inst_.size(); ++i) {
                    if (inst_[i].host + ":" + std::to_string(inst_[i].port) ==
                        inst) {
                        cands.push_back(static_cast<int>(i));
                    }
                }
            }
            if (cands.empty()) {
                res.status = 404;
                res.set_content(error_body("no video " + rid,
                                           "invalid_request_error"),
                                "application/json");
                return;
            }
            target.replace(kVideos.size(), rid.size(), id);
        }
    }
    if (cands.empty()) cands = candidates(rt);
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
    up.target = target;
    up.body = req.body;
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
        Slot& slot = slots_[i];
        ++slot.inflight;
        // Released when the request is done here, or when the stream
        // provider that holds it is gone; it also adds the wall time.
        const auto t0 = std::chrono::steady_clock::now();
        std::shared_ptr<void> guard(nullptr, [&slot, t0](void*) {
            --slot.inflight;
            slot.busy_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0)
                    .count());
        });
        std::shared_ptr<http::Client> client = take(i);
        const bool media = rt.kind == Kind::kImage ||
                           rt.kind == Kind::kSpeech ||
                           rt.kind == Kind::kVideo;
        client->set_read_timeout_ms(media ? cfg_.media_read_timeout_ms
                                          : cfg_.upstream.read_timeout_ms);
        {
            std::lock_guard lk(mu_);
            up.headers = upstream_headers(req, inst_[i]);
        }
        const auto head = client->begin(up);
        if (!head.ok) {
            // A refused or timed-out connect means the instance is down
            // until the next poll answers; any other failure (a reply
            // cut short, a bad head) only moves on to the next one.
            ++slot.errors;
            std::lock_guard lk(mu_);
            if (!head.connected) inst_[i].healthy = false;
            last_error = inst_[i].host + ":" + std::to_string(inst_[i].port) +
                         ": " + head.error;
            continue;
        }
        ++slot.requests;
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
                    bool left = false;
                    if (client->read_body(
                            [&](std::string_view d) {
                                if (!sink.write(d)) left = true;
                                return !left;
                            },
                            err)) {
                        give(i, std::move(client));
                    } else if (left) {
                        ++slots_[i].client_left;
                    } else {
                        ++slots_[i].errors;
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
            ++slot.errors;
            res.status = 502;
            res.headers.clear();
            res.set_content(error_body("upstream body: " + err, "server_error"),
                            "application/json");
            return;
        }
        give(i, std::move(client));
        if (rt.kind == Kind::kVideo &&
            content_type.find("json") != std::string::npos) {
            // A job: give it the router's id, which names this instance.
            json j = json::parse(body, nullptr, /*allow_exceptions=*/false);
            if (j.is_object() && j.contains("id") && j["id"].is_string()) {
                std::lock_guard lk(mu_);
                j["id"] = router_video_id(
                    inst_[i].host + ":" + std::to_string(inst_[i].port),
                    j["id"].get<std::string>());
                body = j.dump();
            }
        }
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

std::vector<std::string> parse_register_keys(std::string_view text) {
    std::vector<std::string> keys;
    while (!text.empty()) {
        const std::size_t nl = text.find('\n');
        std::string_view line = text.substr(0, nl);
        text = nl == std::string_view::npos ? std::string_view{}
                                            : text.substr(nl + 1);
        while (!line.empty() && std::isspace(static_cast<unsigned char>(
                                    line.front()))) {
            line.remove_prefix(1);
        }
        while (!line.empty() && std::isspace(static_cast<unsigned char>(
                                    line.back()))) {
            line.remove_suffix(1);
        }
        if (line.empty() || line.front() == '#') continue;
        keys.emplace_back(line);
    }
    return keys;
}

bool parse_router_url(std::string_view url, std::string& host, int& port) {
    constexpr std::string_view kScheme = "http://";
    if (url.rfind(kScheme, 0) != 0) return false;
    url.remove_prefix(kScheme.size());
    if (!url.empty() && url.back() == '/') url.remove_suffix(1);
    if (url.find('/') != std::string_view::npos) return false;
    return pipeline::parse_hostport(std::string(url), host, port) &&
           !host.empty() && port > 0;
}

Registrar::Registrar(Config cfg) : cfg_(std::move(cfg)) {}

Registrar::~Registrar() { stop(); }

bool Registrar::call(const std::string& path) {
    http::ClientOptions o;
    o.connect_timeout_ms = 2000;
    o.read_timeout_ms = 5000;
    http::Client c(cfg_.router_host, cfg_.router_port, o);
    json body{{"address", cfg_.address}};
    if (path == "/locus/register") body["ttl_s"] = cfg_.ttl_s;
    const auto r = c.send(http::ClientRequest{
        "POST",
        path,
        {{"Content-Type", "application/json"},
         {"Authorization", "Bearer " + cfg_.key}},
        body.dump()});
    return r.ok && r.head.status == 200;
}

bool Registrar::register_once() { return call("/locus/register"); }

bool Registrar::deregister() { return call("/locus/deregister"); }

void Registrar::start() {
    std::lock_guard lk(mu_);
    if (started_) return;
    started_ = true;
    thread_ = std::thread([this] {
        std::unique_lock lk(mu_);
        auto wait = std::chrono::milliseconds(cfg_.first_delay_ms);
        const auto every =
            std::chrono::milliseconds(std::max(1, cfg_.ttl_s) * 1000 / 3);
        while (!cv_.wait_for(lk, wait, [this] { return stopping_; })) {
            lk.unlock();
            if (!register_once()) {
                std::fprintf(stderr,
                             "locus: registration with %s:%d failed; "
                             "retrying\n",
                             cfg_.router_host.c_str(), cfg_.router_port);
            }
            lk.lock();
            wait = every;
        }
    });
}

void Registrar::stop() {
    {
        std::lock_guard lk(mu_);
        if (!started_ || stopping_) return;
        stopping_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    deregister();
}

}  // namespace locus::router
