#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "locus/http/client.hpp"
#include "locus/http/server.hpp"

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
enum class Kind { kText, kImageToText, kImage, kVideo, kSpeech };

/** @returns the wire name: text, image-to-text, image, video, speech. */
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
};

/** Router settings. */
struct RouterConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    /** Instances as host:port; a name is resolved on each connect. */
    std::vector<std::string> instances;
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
};

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
    };

    void handle(const http::Request& req, http::ServerResponse& res);
    void models(http::ServerResponse& res) const;
    /** An idle connection to instance i, or a new client. */
    std::shared_ptr<http::Client> take(std::size_t i);
    /** Returns a client to instance i's pool (up to pool_size). */
    void give(std::size_t i, std::shared_ptr<http::Client> c);

    RouterConfig cfg_;
    mutable std::mutex mu_;
    std::vector<Instance> inst_;
    std::vector<std::unique_ptr<Slot>> slots_;  ///< one per instance
    mutable std::atomic<std::uint64_t> rr_{0};
    std::unique_ptr<http::Server> server_;
    std::thread refresher_;
    std::mutex stop_mu_;
    std::condition_variable stop_cv_;
    bool stopping_ = false;
};

}  // namespace locus::router
