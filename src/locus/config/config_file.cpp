#include "locus/config/config_file.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>

namespace locus::config {

namespace {

// Trims leading/trailing ASCII whitespace.
std::string trim(const std::string& s) {
    const auto is_ws = [](unsigned char c) {
        return std::isspace(c) != 0;
    };
    std::size_t b = 0;
    std::size_t e = s.size();
    while (b < e && is_ws(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    while (e > b && is_ws(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

}  // namespace

ConfigFile ConfigFile::parse_string(const std::string& text,
                                    const std::string& source) {
    ConfigFile cfg;
    std::istringstream in(text);
    std::string line;
    std::size_t lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        const std::string t = trim(line);
        if (t.empty() || t.front() == '#') {
            continue;  // blank or comment
        }
        const auto eq = t.find('=');
        if (eq == std::string::npos) {
            throw Error{source + ":" + std::to_string(lineno) +
                        ": expected key = value"};
        }
        const std::string key = trim(t.substr(0, eq));
        if (key.empty()) {
            throw Error{source + ":" + std::to_string(lineno) +
                        ": empty key"};
        }
        const std::string value = trim(t.substr(eq + 1));
        if (std::find(cfg.keys_.begin(), cfg.keys_.end(), key) ==
            cfg.keys_.end()) {
            cfg.keys_.push_back(key);
        }
        cfg.entries_.push_back({key, value});
    }
    return cfg;
}

ConfigFile ConfigFile::parse(const std::string& path) {
    // Reject a non-regular path (missing, or a directory/device) up
    // front: an ifstream opens a directory "successfully" but reads
    // nothing, which would otherwise look like an empty config rather
    // than an error -- and silently applying nothing on a SIGHUP reload
    // (i#19 part 2) would read as a successful no-op.
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        throw Error{"config file does not exist: " + path};
    }
    if (!std::filesystem::is_regular_file(path, ec)) {
        throw Error{"cannot read config file (not a regular file): " +
                    path};
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw Error{"cannot open config file: " + path};
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    if (f.bad()) {
        throw Error{"error reading config file: " + path};
    }
    return parse_string(ss.str(), path);
}

bool ConfigFile::has(const std::string& key) const {
    return std::find(keys_.begin(), keys_.end(), key) != keys_.end();
}

std::optional<std::string> ConfigFile::get(const std::string& key) const {
    std::optional<std::string> out;
    for (const auto& e : entries_) {
        if (e.key == key) {
            out = e.value;  // last occurrence wins
        }
    }
    return out;
}

std::vector<std::string> ConfigFile::values(const std::string& key) const {
    std::vector<std::string> out;
    for (const auto& e : entries_) {
        if (e.key == key) {
            out.push_back(e.value);
        }
    }
    return out;
}

}  // namespace locus::config
