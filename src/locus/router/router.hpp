#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "locus/auth/auth_client.hpp"
#include "locus/auth/token_cache.hpp"
#include "locus/http/client.hpp"
#include "locus/http/server.hpp"
#include "locus/pipeline/mdns.hpp"
#include "locus/pipeline/resolver.hpp"

namespace locus::router {

/**
 * locus-router (issue 388): one URL in front of dedicated locus
 * instances, one kind of request per instance. The router reads each
 * request's path (and, for chat, its body), picks a healthy instance
 * that serves that kind (and the requested model, when it names one),
 * and proxies the request, streaming SSE replies through as they
 * arrive.
 */

/** What a request asks for; an instance lists the kinds it serves in
 * its /v1/models (`locus_kinds`). */
enum class Kind { kText, kImageToText, kImage, kVideo, kSpeech, kRerank };

/** @returns the wire name: text, image-to-text, image, video, speech,
 *  rerank. */
std::string_view kind_name(Kind k);
/** @returns the kind for a wire name, or nullopt. */
std::optional<Kind> parse_kind(std::string_view name);

/** How the router sorts one request. */
struct Route {
    Kind kind = Kind::kText;
    /** The body's `model`, when it names one ("" otherwise). */
    std::string model;
    /** The session key (session_key()); "" when there is none. */
    std::string session;
};

/**
 * The key that sends one conversation to one instance, so its prefix
 * cache and hot experts are reused: the client's own key when it sends
 * one (the `X-Session-Id` header, OpenAI's `prompt_cache_key`,
 * Anthropic's `metadata.user_id`), else a hash of the prompt's start
 * (the system prompt and the first user turn, which every later turn
 * resends).
 * @param header The X-Session-Id header ("" when absent).
 * @param body The request body (JSON; anything else gives "").
 */
std::string session_key(std::string_view header, std::string_view body);

/**
 * The id the router gives a video job, so a later GET /v1/videos/{id}
 * reaches the instance that made it with no table: "video_", the
 * instance's host:port in hex, "_", then the instance's own id without
 * its "video_" prefix; an id without that prefix goes whole after a
 * second "_", so split_video_id gives back any id. It stays an id of
 * word characters when the instance's id is one.
 * @param instance The instance's host:port.
 * @param id The instance's job id (locus makes "video_" and hex
 *     digits).
 */
std::string router_video_id(std::string_view instance, std::string_view id);

/**
 * Splits an id from router_video_id().
 * @param[out] instance The instance's host:port.
 * @param[out] id The instance's own id.
 * @returns false for an id the router did not make.
 */
bool split_video_id(std::string_view rid, std::string& instance,
                    std::string& id);

/** @returns the rendezvous score of `instance` (host:port) for `key`:
 * a 64-bit FNV-1a hash of both, the same in every router. */
std::uint64_t rendezvous_score(std::string_view key,
                               std::string_view instance);

/**
 * Sorts a request by its path, and for chat, messages and responses by
 * whether the body carries an image part.
 * @param path The request path.
 * @param body The raw body (may be empty).
 * @returns nullopt for a path the router does not proxy.
 */
std::optional<Route> classify(std::string_view path, std::string_view body);

/** One locus instance behind the router, as last seen. */
struct Instance {
    std::string host;
    int port = 0;
    bool healthy = false;
    std::vector<Kind> kinds;
    std::vector<std::string> models;
    /** The /v1/models data entries, for the router's own listing. */
    std::string models_json;
    /** Found by a name: an SRV name or a scan entry (not given as
     *  host:port). One entry per host:port, whichever names list it. */
    bool from_srv = false;
    /** The SRV names and scan entries ("scan:CIDR:PORT") whose last
     *  answer lists it, sorted. */
    std::vector<std::string> srv;
    /** Registered itself (POST /locus/register). */
    bool from_reg = false;
    /** When its last registration runs out. */
    std::chrono::steady_clock::time_point reg_until{};
    /** Found by SRV or by registration, and neither lists it now (no
     *  name lists it, and its registration ran out or was dropped):
     *  never picked again unless it is listed once more. */
    bool gone = false;
};

/** The most instances a router keeps, SRV targets included. */
inline constexpr std::size_t kMaxInstances = 1024;

/** Router settings. */
struct RouterConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    /** Instances as host:port (a name is resolved on each connect), or
     * "srv:NAME": the targets of NAME's DNS SRV records (the lowest
     * priority's), looked up again on every refresh, so instances
     * join and leave with the record. An empty or failed answer keeps
     * the targets the name listed last. */
    std::vector<std::string> instances;
    /** The SRV lookup (tests set a fake); null: dns_srv_lookup. */
    std::function<std::vector<pipeline::SrvTarget>(const std::string&)>
        srv_lookup;
    /** An instance may also be "scan:CIDR:PORT" (IPv4, at most 1024
     * addresses, so /22 or longer): every address of the range is
     * probed on PORT for a locus /v1/models, once at start and then
     * every scan_every refreshes; the ones that answer are the
     * instances of that entry until a scan no longer finds them. */
    int scan_every = 6;
    /** An instance may also be "mdns:SERVICE" (for example
     * "mdns:_locus._tcp"): the instances that answer a DNS-SD browse
     * for SERVICE.local on the LAN, browsed again on every refresh.
     * An empty answer keeps the instances it listed last (mDNS
     * packets get lost); one it stops listing drops out. */
    pipeline::MdnsBrowse mdns;
    /** The browse (tests set a fake); null: pipeline::mdns_browse. */
    std::function<std::vector<pipeline::SrvTarget>(const std::string&)>
        mdns_lookup;
    /** Keys an instance may register itself with (POST /locus/register
     * and /locus/deregister, Authorization: Bearer KEY). Empty: registration
     * is off (404), since anyone who could register could take the
     * router's traffic. */
    std::vector<std::string> register_keys;
    /** How often /v1/models is fetched from each instance. */
    int refresh_ms = 10000;
    /** Load bound: an instance with more requests in flight than this
     * times the average over its kind's candidates passes the request
     * to the key's next instance. */
    double load_factor = 1.25;
    /** Idle kept-alive connections kept per instance. */
    int pool_size = 8;
    /** Toward the instances. Media requests can take minutes, so the
     * read deadline is per gap between reads, not per request. */
    http::ClientOptions upstream;
    int worker_threads = 16;

    /** Client keys: when set, every request but /health needs a key
     * this auth helper allows (Authorization: Bearer, or x-api-key),
     * else a 401. The argv of the helper, as locus-server takes it. */
    std::vector<std::string> auth_helper_argv;
    std::size_t auth_helpers = 1;
    auth::AuthConfig auth_cache;

    /** What the router sends the instances as a credential. */
    enum class UpstreamAuth {
        kClient,  ///< the client's own Authorization / x-api-key
        kNone,    ///< nothing: instances trust the router
        kKey,     ///< the key set for the instance (instance_keys)
    };
    UpstreamAuth upstream_auth = UpstreamAuth::kClient;
    /** For kKey: host:port to key; "*" is the key of any other. */
    std::map<std::string, std::string> instance_keys;
};

/**
 * Parses an instance-keys file: one "host:port key" (or "* key") per
 * line; blank lines and lines starting with '#' are skipped.
 * @throws std::invalid_argument on a malformed line.
 */
std::map<std::string, std::string> parse_instance_keys(std::string_view text);

/**
 * The router: an http::Server whose handler proxies each request to an
 * instance. Instances are polled for their kinds and models on start
 * and every refresh_ms; one that fails the poll is skipped until it
 * answers again.
 */
class Router {
  public:
    /** @throws std::invalid_argument on a malformed instance address. */
    explicit Router(RouterConfig cfg);
    ~Router();
    Router(const Router&) = delete;
    Router& operator=(const Router&) = delete;

    /** Polls the instances once, then serves. @returns the bound port,
     * or -1 if the bind failed. */
    int start();
    /** Stops serving and polling. Safe to call more than once. */
    void stop();
    /** Blocks until stop(). */
    void wait();

    /** Polls every instance now (also run by the refresh thread). */
    void refresh();
    /** @returns a copy of the instances as last polled. */
    std::vector<Instance> instances() const;

    /**
     * @returns the instance indexes that serve `route`, best first: the
     * ones that list its model (when any does), each group in
     * rendezvous order for the session key (or round robin without
     * one), with instances over the load bound moved after the rest.
     */
    std::vector<int> candidates(const Route& route) const;

  private:
    /** Per instance, kept across refreshes: requests in flight and
     * idle connections. */
    struct Slot {
        std::atomic<int> inflight{0};
        std::mutex mu;
        std::vector<std::shared_ptr<http::Client>> idle;
        // For /metrics: requests answered from this instance, the ones
        // that failed upstream, streams a client left, wall seconds.
        std::atomic<std::uint64_t> requests{0};
        std::atomic<std::uint64_t> errors{0};
        std::atomic<std::uint64_t> client_left{0};
        std::atomic<std::uint64_t> busy_ns{0};
    };

    void handle(const http::Request& req, http::ServerResponse& res);
    void models(http::ServerResponse& res) const;
    /** The router's own Prometheus metrics, per instance. */
    void metrics(http::ServerResponse& res) const;
    /** An idle connection to instance i, or a new client. */
    std::shared_ptr<http::Client> take(std::size_t i);
    /** Returns a client to instance i's pool (up to pool_size). */
    void give(std::size_t i, std::shared_ptr<http::Client> c);

    /** The head stage: the client key check (cfg_.auth_helper_argv).
     * @returns true to reject (res holds the 401). */
    bool reject(const http::Request& head, http::ServerResponse& res);
    /** The request's headers for instance `in`: the client's, minus
     * hop-by-hop ones, with the credential the upstream_auth mode
     * sends, and the router's loop mark (X-Locus-Router). */
    std::vector<std::pair<std::string, std::string>> upstream_headers(
        const http::Request& req, const Instance& in) const;
    /** The instance key for `in`: its own "HOST:PORT KEY" line, else
     * the "*" line, but "*" only for an instance given as host:port:
     * one found by SRV or registration gets a key only from a line
     * that names it, so a key goes only to an address the operator
     * named. @returns nullptr when there is none. */
    const std::string* instance_key(const Instance& in) const;

    RouterConfig cfg_;
    auth::WallClock clock_;
    std::unique_ptr<auth::AuthClient> auth_;  ///< null: no key check
    /** Adds the targets of every SRV name to inst_ and marks the ones
     *  a name no longer lists as gone (run first by refresh). */
    void expand_srv();
    /** Browses every mdns: service and merges its answer (run by
     *  refresh after expand_srv). */
    void expand_mdns();
    /** Probes every scan entry's addresses and merges the ones that
     *  answer (run by refresh every scan_every refreshes). */
    void scan();
    /** Merges one source's current targets (an SRV name's answer, or
     *  a scan's hits) into inst_: adds new host:ports, adds or drops
     *  `name` from each found entry's names, and marks an entry gone
     *  when no source lists it. Takes mu_. */
    void merge_listing(const std::string& name,
                       const std::vector<std::pair<std::string, int>>& targets);

    /** One "scan:CIDR:PORT" entry. */
    struct ScanSpec {
        std::string name;      ///< the entry as given, as its source name
        std::uint32_t first;   ///< the first address probed, host order
        std::uint32_t count;   ///< addresses probed
        int port;
    };
    std::vector<ScanSpec> scans_;
    std::atomic<std::uint64_t> refreshes_{0};  ///< for scan_every
    /** POST /locus/register and /locus/deregister
     *  (RouterConfig::register_keys). */
    void registration(const http::Request& req, http::ServerResponse& res);
    /** Polls one instance's /v1/models into `in` (no lock held). */
    void poll(Instance& in) const;

    mutable std::mutex mu_;
    /** Only grows (under mu_), so an index stays the same instance. */
    std::vector<Instance> inst_;
    /** One per instance, kMaxInstances allocated up front: adding an
     *  instance never moves a Slot a request is using. */
    std::unique_ptr<Slot[]> slots_;
    std::vector<std::string> srv_names_;
    /** The services of the "mdns:SERVICE" entries. */
    std::vector<std::string> mdns_services_;
    mutable std::atomic<std::uint64_t> rr_{0};
    std::unique_ptr<http::Server> server_;
    std::thread refresher_;
    std::mutex stop_mu_;
    std::condition_variable stop_cv_;
    bool stopping_ = false;
};

/**
 * Parses a --register-keys file: one key per line, surrounding spaces
 * trimmed; blank lines and lines starting with '#' are skipped.
 */
std::vector<std::string> parse_register_keys(std::string_view text);

/**
 * Parses a router URL for --register: "http://HOST:PORT", with an
 * optional trailing "/". @returns false for anything else (https is
 * not supported).
 */
bool parse_router_url(std::string_view url, std::string& host, int& port);

/**
 * Keeps one instance registered with a router (locus-server
 * --register): registers it once start() is called, again every
 * ttl_s / 3, and deregisters it on stop(). A failed call is retried on
 * the next tick; an instance that dies without stop() drops out of
 * the router when its last registration runs out.
 */
class Registrar {
  public:
    struct Config {
        std::string router_host;
        int router_port = 0;
        std::string address;  ///< HOST:PORT the router reaches this at
        std::string key;      ///< one of the router's register_keys
        int ttl_s = 30;
        /** Before the first registration, so the server is listening
         *  when the router polls it. */
        int first_delay_ms = 1000;
    };

    explicit Registrar(Config cfg);
    ~Registrar();
    Registrar(const Registrar&) = delete;
    Registrar& operator=(const Registrar&) = delete;

    /** Starts the heartbeat thread. */
    void start();
    /** Stops it and deregisters (idempotent). */
    void stop();

    /** One registration. @returns whether the router took it. */
    bool register_once();
    /** One deregistration. @returns whether the router took it. */
    bool deregister();

  private:
    /** POSTs this instance to `path` (/locus/register or
     *  /locus/deregister). @returns whether the router took it. */
    bool call(const std::string& path);

    Config cfg_;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stopping_ = false;
    bool started_ = false;
};

}  // namespace locus::router
