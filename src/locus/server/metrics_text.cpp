#include "locus/server/metrics_text.hpp"

#include <charconv>

namespace locus::server {

namespace {

std::string render_labels(const std::vector<MetricsText::Label>& labels) {
    std::string out;
    for (const auto& [k, v] : labels) {
        if (!out.empty()) out += ',';
        out += k;
        out += "=\"";
        out += MetricsText::escape(v);
        out += '"';
    }
    return out;
}

}  // namespace

std::string format_double(double v) {
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, v);
    return std::string(buf, r.ptr);
}

MetricsText::MetricsText(const std::vector<Label>& labels)
    : labels_(render_labels(labels)) {}

std::string MetricsText::escape(std::string_view v) {
    std::string out;
    out.reserve(v.size());
    for (const char c : v) {
        if (c == '\\') {
            out += "\\\\";
        } else if (c == '"') {
            out += "\\\"";
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out += c;
        }
    }
    return out;
}

std::string MetricsText::label(std::string_view key, std::string_view v) {
    std::string out(key);
    out += "=\"";
    out += escape(v);
    out += '"';
    return out;
}

void MetricsText::family(std::string_view name, std::string_view help,
                         std::string_view type) {
    out_ += "# HELP ";
    out_ += name;
    out_ += ' ';
    out_ += help;
    out_ += "\n# TYPE ";
    out_ += name;
    out_ += ' ';
    out_ += type;
    out_ += '\n';
}

void MetricsText::sample(std::string_view name, std::string_view extra,
                         std::string_view value) {
    out_ += name;
    if (!labels_.empty() || !extra.empty()) {
        out_ += '{';
        out_ += labels_;
        if (!labels_.empty() && !extra.empty()) out_ += ',';
        out_ += extra;
        out_ += '}';
    }
    out_ += ' ';
    out_ += value;
    out_ += '\n';
}

void MetricsText::value(std::string_view name, std::string_view extra,
                        std::uint64_t v) {
    sample(name, extra, std::to_string(v));
}

void MetricsText::value(std::string_view name, std::string_view extra,
                        double v) {
    sample(name, extra, format_double(v));
}

void MetricsText::counter(std::string_view name, std::string_view help,
                          std::uint64_t v) {
    family(name, help, "counter");
    value(name, "", v);
}

void MetricsText::counter(std::string_view name, std::string_view help,
                          double v) {
    family(name, help, "counter");
    value(name, "", v);
}

void MetricsText::gauge(std::string_view name, std::string_view help,
                        std::uint64_t v) {
    family(name, help, "gauge");
    value(name, "", v);
}

void MetricsText::histogram_series(std::string_view name,
                                   std::string_view extra,
                                   const sys::Histogram::Snapshot& s) {
    const std::string bucket = std::string(name) + "_bucket";
    const std::string pre =
        extra.empty() ? std::string() : std::string(extra) + ",";
    for (std::size_t i = 0; i < s.cumulative.size(); ++i) {
        const std::string le =
            i < s.bounds.size() ? format_double(s.bounds[i]) : "+Inf";
        sample(bucket, pre + "le=\"" + le + "\"",
               std::to_string(s.cumulative[i]));
    }
    sample(std::string(name) + "_sum", extra, format_double(s.sum));
    sample(std::string(name) + "_count", extra, std::to_string(s.count));
}

void MetricsText::histogram(std::string_view name, std::string_view help,
                            const sys::Histogram::Snapshot& s) {
    family(name, help, "histogram");
    histogram_series(name, "", s);
}

void MetricsText::info(std::string_view name, std::string_view help,
                       const std::vector<Label>& extra) {
    family(name, help, "gauge");
    sample(name, render_labels(extra), "1");
}

}  // namespace locus::server
