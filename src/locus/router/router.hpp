#pragma once

#include <condition_variable>
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
};

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

  private:
    void handle(const http::Request& req, http::ServerResponse& res);
    /** @returns the index of an instance for `route`, or -1. */
    int pick(const Route& route) const;
    void models(http::ServerResponse& res) const;

    RouterConfig cfg_;
    mutable std::mutex mu_;
    std::vector<Instance> inst_;
    std::unique_ptr<http::Server> server_;
    std::thread refresher_;
    std::mutex stop_mu_;
    std::condition_variable stop_cv_;
    bool stopping_ = false;
};

}  // namespace locus::router
