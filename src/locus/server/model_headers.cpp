#include "locus/server/model_headers.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>

namespace locus::server {

std::string sf_string(std::string_view s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (c >= 0x20 && c < 0x7f) {
            out += c;
        } else {
            out += '?';
        }
    }
    out += '"';
    return out;
}

std::string model_file_item(std::string_view file, const gguf::GgufFile& g) {
    std::string out = sf_string(file);
    if (const auto name = g.get_string("general.name")) {
        out += ";name=" + sf_string(*name);
    }
    if (const auto v = g.get_string("general.version")) {
        out += ";version=" + sf_string(*v);
    } else if (const auto rev = g.get_string("general.source.revision")) {
        out += ";version=" + sf_string(rev->substr(0, 12));
    }
    out += ";type=";
    out += gguf::tensor_type_name(gguf::main_tensor_type(g));
    return out;
}

ModelHeaders model_headers(std::string model,
                           std::span<const std::string> paths) {
    ModelHeaders h;
    h.model = std::move(model);
    for (const std::string& p : paths) {
        const auto g = gguf::GgufFile::open(p);
        if (!h.files.empty()) h.files += ", ";
        h.files += model_file_item(
            std::filesystem::path(p).filename().string(), g);
    }
    return h;
}

std::vector<std::string> gguf_files_in(const std::string& dir) {
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (e.is_regular_file() && e.path().extension() == ".gguf") {
            out.push_back(e.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::string lora_header(std::span<const image::LoraPick> picks) {
    std::string out;
    for (const image::LoraPick& p : picks) {
        // A structured field decimal: at most 3 decimals, no exponent;
        // trailing zeros cut, so 0.6 reads 0.6 and 1 reads 1.0.
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.3f",
                      static_cast<double>(p.strength));
        std::string scale = buf;
        while (scale.back() == '0' && scale[scale.size() - 2] != '.') {
            scale.pop_back();
        }
        if (scale == "-0.0") scale = "0.0";
        if (!out.empty()) out += ", ";
        out += sf_string(p.name) + ";scale=" + scale;
    }
    return out;
}

}  // namespace locus::server
