#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "locus/sys/histogram.hpp"

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
                   const sys::Histogram::Snapshot& s);
    /** An info series: value 1, the fixed labels plus `extra`. */
    void info(std::string_view name, std::string_view help,
              const std::vector<Label>& extra);

    /**
     * `# HELP` and `# TYPE` once for a family of labelled series; the
     * samples follow with value() or histogram_series().
     */
    void family(std::string_view name, std::string_view help,
                std::string_view type);
    /**
     * One sample with `extra` labels after the fixed ones.
     *
     * @param extra Rendered labels without braces (see label()).
     */
    void value(std::string_view name, std::string_view extra,
               std::uint64_t v);
    void value(std::string_view name, std::string_view extra, double v);
    /** One histogram's `_bucket`, `_sum` and `_count` with `extra`. */
    void histogram_series(std::string_view name, std::string_view extra,
                          const sys::Histogram::Snapshot& s);

    const std::string& text() const { return out_; }

    /** @returns `v` escaped for a label value (\\, \", \n). */
    static std::string escape(std::string_view v);
    /** @returns `key="v"` with v escaped. */
    static std::string label(std::string_view key, std::string_view v);

  private:
    /** `name{labels[,extra]} value\n`. */
    void sample(std::string_view name, std::string_view extra,
                std::string_view value);

    std::string labels_;  ///< rendered, without braces
    std::string out_;
};

/** @returns `v` in the shortest form that reads back the same. */
std::string format_double(double v);

}  // namespace locus::server
