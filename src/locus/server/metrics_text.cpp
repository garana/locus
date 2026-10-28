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

void MetricsText::header(std::string_view name, std::string_view help,
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

void MetricsText::counter(std::string_view name, std::string_view help,
                          std::uint64_t v) {
    header(name, help, "counter");
    sample(name, "", std::to_string(v));
}

void MetricsText::counter(std::string_view name, std::string_view help,
                          double v) {
    header(name, help, "counter");
    sample(name, "", format_double(v));
}

void MetricsText::gauge(std::string_view name, std::string_view help,
                        std::uint64_t v) {
    header(name, help, "gauge");
    sample(name, "", std::to_string(v));
}

void MetricsText::histogram(std::string_view name, std::string_view help,
                            const engine::Histogram::Snapshot& s) {
    header(name, help, "histogram");
    const std::string bucket = std::string(name) + "_bucket";
    for (std::size_t i = 0; i < s.cumulative.size(); ++i) {
        const std::string le =
            i < s.bounds.size() ? format_double(s.bounds[i]) : "+Inf";
        sample(bucket, "le=\"" + le + "\"", std::to_string(s.cumulative[i]));
    }
    sample(std::string(name) + "_sum", "", format_double(s.sum_seconds));
    sample(std::string(name) + "_count", "", std::to_string(s.count));
}

void MetricsText::info(std::string_view name, std::string_view help,
                       const std::vector<Label>& extra) {
    header(name, help, "gauge");
    sample(name, render_labels(extra), "1");
}

}  // namespace locus::server
