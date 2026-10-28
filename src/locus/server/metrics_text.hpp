#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "locus/engine/metrics.hpp"

namespace locus::server {

/**
 * Writes the Prometheus text format (version 0.0.4) with one fixed
 * label set on every series (the model and backend of this server).
 */
class MetricsText {
  public:
    /** A label name and its raw (unescaped) value. */
    using Label = std::pair<std::string, std::string>;

    /** @param labels The labels every series carries. */
    explicit MetricsText(const std::vector<Label>& labels);

    void counter(std::string_view name, std::string_view help,
                 std::uint64_t v);
    /** A counter in seconds (or any other fractional unit). */
    void counter(std::string_view name, std::string_view help, double v);
    void gauge(std::string_view name, std::string_view help,
               std::uint64_t v);
    /** `name_bucket{le=...}`, `name_sum` and `name_count`. */
    void histogram(std::string_view name, std::string_view help,
                   const engine::Histogram::Snapshot& s);
    /** An info series: value 1, the fixed labels plus `extra`. */
    void info(std::string_view name, std::string_view help,
              const std::vector<Label>& extra);

    const std::string& text() const { return out_; }

    /** @returns `v` escaped for a label value (\\, \", \n). */
    static std::string escape(std::string_view v);

  private:
    void header(std::string_view name, std::string_view help,
                std::string_view type);
    /** `name{labels[,extra]} value\n`. */
    void sample(std::string_view name, std::string_view extra,
                std::string_view value);

    std::string labels_;  ///< rendered, without braces
    std::string out_;
};

/** @returns `v` in the shortest form that reads back the same. */
std::string format_double(double v);

}  // namespace locus::server
