#include "locus/auth/helper_message.hpp"

#include <cstdint>

namespace locus::auth {

namespace {

/** Cap on an un-terminated record, so a runaway peer cannot force
 * unbounded buffering before we reject the connection. */
constexpr std::size_t kMaxTextRecord = 64 * 1024;

/** A field key/value is wire-safe if it has no newline (which would
 * break framing); keys additionally may not contain '='. */
bool value_ok(const std::string& s) {
    return s.find('\n') == std::string::npos;
}
bool key_ok(const std::string& s) {
    return value_ok(s) && s.find('=') == std::string::npos;
}

std::string_view trim_cr(std::string_view s) {
    if (!s.empty() && s.back() == '\r') {
        s.remove_suffix(1);
    }
    return s;
}

}  // namespace

bool encode_text(const HelperMessage& msg, std::string& out) {
    if (!value_ok(msg.type)) {
        return false;
    }
    for (const auto& kv : msg.fields) {
        if (!key_ok(kv.first) || !value_ok(kv.second)) {
            return false;
        }
    }
    out.append("req_id=");
    out.append(std::to_string(msg.req_id));
    out.push_back('\n');
    out.append("type=");
    out.append(msg.type);
    out.push_back('\n');
    for (const auto& kv : msg.fields) {
        out.append(kv.first);
        out.push_back('=');
        out.append(kv.second);
        out.push_back('\n');
    }
    out.push_back('\n');  // blank line terminates the record
    return true;
}

HelperDecode decode_text(std::string& buf, HelperMessage& out,
                         std::string& err) {
    const std::size_t term = buf.find("\n\n");
    if (term == std::string::npos) {
        if (buf.size() > kMaxTextRecord) {
            err = "text codec: record exceeds cap without terminator";
            return HelperDecode::kError;
        }
        return HelperDecode::kIncomplete;
    }
    const std::string_view block(buf.data(), term);

    HelperMessage msg;
    bool have_req_id = false;
    std::size_t pos = 0;
    while (pos <= block.size()) {
        std::size_t eol = block.find('\n', pos);
        if (eol == std::string_view::npos) {
            eol = block.size();
        }
        const std::string_view line =
            trim_cr(block.substr(pos, eol - pos));
        pos = eol + 1;
        if (line.empty()) {
            continue;
        }
        const std::size_t eq = line.find('=');
        if (eq == std::string_view::npos) {
            err = "text codec: line without '='";
            return HelperDecode::kError;
        }
        const std::string_view key = line.substr(0, eq);
        const std::string_view value = line.substr(eq + 1);
        if (key == "req_id") {
            if (value.empty() || value.size() > 20) {
                err = "text codec: bad req_id";
                return HelperDecode::kError;
            }
            std::uint64_t id = 0;
            for (char c : value) {
                if (c < '0' || c > '9') {
                    err = "text codec: bad req_id";
                    return HelperDecode::kError;
                }
                const std::uint64_t d =
                    static_cast<std::uint64_t>(c - '0');
                if (id > (UINT64_MAX - d) / 10) {
                    err = "text codec: req_id overflow";
                    return HelperDecode::kError;
                }
                id = id * 10 + d;
            }
            msg.req_id = id;
            have_req_id = true;
        } else if (key == "type") {
            msg.type = std::string(value);
        } else {
            msg.set(std::string(key), std::string(value));
        }
    }
    if (!have_req_id) {
        err = "text codec: missing req_id";
        return HelperDecode::kError;
    }

    buf.erase(0, term + 2);
    out = std::move(msg);
    return HelperDecode::kComplete;
}

}  // namespace locus::auth
