#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "locus/sys/histogram.hpp"

namespace locus::http {

/**
 * Per-route request metrics of an http::Server (i#285): requests by
 * status code, request and response body sizes, and processing time
 * (head received to response written). Routes are a fixed list given
 * at construction; any other path counts as "other", so a client
 * cannot create series. Every write is a relaxed atomic add (no lock,
 * no allocation); /metrics reads them while requests run.
 */
class HttpMetrics {
  public:
    /** Status codes kept per route: 100 to 599. */
    static constexpr int kMinCode = 100, kMaxCode = 599;

    /**
     * @param routes Request paths (without the query), matched in
     *     order: exact, or, for a name ending in a slash, any longer
     *     path under it (for example "/v1/models/" for every model id).
     */
    explicit HttpMetrics(std::vector<std::string> routes);

    /** @returns The index of `path`, or other() when it is not a route. */
    std::size_t route(std::string_view path) const;
    /** @returns The index of the "other" route. */
    std::size_t other() const { return names_.size(); }
    /** @returns The route names, then "other". */
    std::vector<std::string> names() const;

    /**
     * Records one answered request.
     *
     * @param status The status written (a code outside 100..599 counts
     *     in Route::out_of_range).
     * @param request_bytes The request body size (for a 408, the body
     *     bytes counted for the upload-rate check: those read after the
     *     head, and only with a rate floor set).
     * @param response_bytes The response body bytes written (for a
     *     stream, the payload of every chunk).
     * @param ns Head received to response written.
     */
    void record(std::size_t route, int status, std::uint64_t request_bytes,
                std::uint64_t response_bytes, std::uint64_t ns);

    /** Requests between head received and response written. */
    std::atomic<std::int64_t> in_flight{0};

    /** One route's series. */
    struct Route {
        Route();
        /** Count per status code, index code - kMinCode; a code outside
         * the range counts in out_of_range. */
        std::array<std::atomic<std::uint64_t>, kMaxCode - kMinCode + 1> codes;
        std::atomic<std::uint64_t> out_of_range{0};
        sys::Histogram request_bytes, response_bytes, seconds;
    };
    /** @returns Route i (i <= other()). */
    const Route& at(std::size_t i) const { return *routes_[i]; }

  private:
    std::vector<std::string> names_;
    std::vector<std::unique_ptr<Route>> routes_;  ///< names_ then other
};

}  // namespace locus::http
