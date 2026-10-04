#include "locus/http/http.hpp"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace locus::http {

namespace {

char lower(char c) {
    return static_cast<char>(
        std::tolower(static_cast<unsigned char>(c)));
}

bool ci_equal(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (lower(a[i]) != lower(b[i])) {
            return false;
        }
    }
    return true;
}

std::string_view trim_ows(std::string_view s) {
    // RFC 9110 OWS = SP / HTAB, around a header value.
    std::size_t b = 0;
    std::size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) {
        ++b;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t')) {
        --e;
    }
    return s.substr(b, e - b);
}

bool is_tchar(char c) {
    // RFC 9110 token chars (method, header-name, TE/Connection list
    // members). Letters, digits, and a fixed punctuation set.
    if (std::isalnum(static_cast<unsigned char>(c))) {
        return true;
    }
    switch (c) {
        case '!': case '#': case '$': case '%': case '&': case '\'':
        case '*': case '+': case '-': case '.': case '^': case '_':
        case '`': case '|': case '~':
            return true;
        default:
            return false;
    }
}

bool is_token(std::string_view s) {
    if (s.empty()) {
        return false;
    }
    for (char c : s) {
        if (!is_tchar(c)) {
            return false;
        }
    }
    return true;
}

// @param status the status the caller should answer (default 400); pass
// 413 for a size cap, 501 for an unimplemented method, 505 for a version.
ParseResult err(std::string msg, int status = 400) {
    ParseResult r;
    r.state = ParseState::kError;
    r.error = std::move(msg);
    r.suggested_status = status;
    return r;
}

// RFC 9110 field-value: a header value may hold VCHAR (0x21-0x7E), SP,
// HTAB, and obs-text (0x80-0xFF); every other control byte is forbidden.
// @returns false if `s` contains 0x00-0x08, 0x0A-0x1F, or 0x7F. Compared
// through unsigned char: plain char is SIGNED on x86-64, so a signed
// `c < 0x20` test would reject obs-text/UTF-8 (0x80+) on that platform
// only -- the kind of asymmetry two arm64 reviewers would ship green.
bool valid_field_value(std::string_view s) {
    for (char c : s) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if ((uc < 0x20 && uc != '\t') || uc == 0x7F) {
            return false;
        }
    }
    return true;
}

ParseResult need_more() {
    ParseResult r;
    r.state = ParseState::kNeedMore;
    return r;
}

// Parses a decimal Content-Length with no sign, no surrounding space,
// no overflow. @returns false (and leaves `out` unset) on any of those.
bool parse_content_length(std::string_view s, std::uint64_t& out) {
    if (s.empty()) {
        return false;
    }
    std::uint64_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') {
            return false;  // sign, space, or junk
        }
        const std::uint64_t d = static_cast<std::uint64_t>(c - '0');
        if (v > (std::numeric_limits<std::uint64_t>::max() - d) / 10) {
            return false;  // overflow
        }
        v = v * 10 + d;
    }
    out = v;
    return true;
}

// Parses a hex chunk-size (no "0x", no sign, overflow-guarded). The
// caller has already split off any chunk-extension. @returns false on
// empty/junk/overflow.
bool parse_chunk_size(std::string_view s, std::uint64_t& out) {
    if (s.empty()) {
        return false;
    }
    std::uint64_t v = 0;
    for (char c : s) {
        std::uint64_t d;
        if (c >= '0' && c <= '9') {
            d = static_cast<std::uint64_t>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            d = static_cast<std::uint64_t>(c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            d = static_cast<std::uint64_t>(c - 'A' + 10);
        } else {
            return false;  // junk (a chunk-extension ';' lands here)
        }
        if (v > (std::numeric_limits<std::uint64_t>::max() - d) / 16) {
            return false;  // overflow
        }
        v = v * 16 + d;
    }
    out = v;
    return true;
}

// @returns true if `list` (a comma-separated field) contains `token` as
// a case-insensitive member -- e.g. Connection: keep-alive, close.
bool list_has_token(std::string_view list, std::string_view token) {
    std::size_t i = 0;
    while (i < list.size()) {
        std::size_t j = list.find(',', i);
        if (j == std::string_view::npos) {
            j = list.size();
        }
        if (ci_equal(trim_ows(list.substr(i, j - i)), token)) {
            return true;
        }
        i = j + 1;
    }
    return false;
}

}  // namespace

bool Request::has_header(std::string_view name) const {
    for (const auto& [k, v] : headers) {
        if (ci_equal(k, name)) {
            return true;
        }
    }
    return false;
}

std::string Request::get_header_value(std::string_view name) const {
    for (const auto& [k, v] : headers) {
        if (ci_equal(k, name)) {
            return v;
        }
    }
    return {};
}

// Finds the next CRLF at or after `from` in `buf`, requiring CR and LF
// to be paired: a bare LF (LF without a preceding CR) is a hard error,
// reported via `bare_lf`. @returns the offset of the CR, or npos if no
// line terminator is present yet (need more bytes).
static std::size_t find_crlf(std::string_view buf, std::size_t from,
                             bool& bare_lf) {
    bare_lf = false;
    for (std::size_t i = from; i < buf.size(); ++i) {
        if (buf[i] == '\n') {
            bare_lf = true;  // LF not reached via the CR branch below
            return std::string_view::npos;
        }
        if (buf[i] == '\r') {
            if (i + 1 >= buf.size()) {
                return std::string_view::npos;  // need the LF
            }
            if (buf[i + 1] != '\n') {
                bare_lf = true;  // CR not followed by LF
                return std::string_view::npos;
            }
            return i;
        }
    }
    return std::string_view::npos;
}

// Length of the line-so-far when no CRLF has arrived yet: everything in
// [from, buf.size()) except a trailing CR still awaiting its LF. Using
// this in the waiting branch makes the limit agree with the complete
// branch (which measures up to, not including, the CR) at exactly the
// cap -- so whether a line at the cap passes does not depend on how TCP
// split it. (find_crlf guarantees any interior bare CR is already an
// error, so a lone CR here can only be the last byte.)
static std::size_t pending_len(std::string_view buf, std::size_t from) {
    std::size_t end = buf.size();
    if (end > from && buf[end - 1] == '\r') {
        --end;
    }
    return end - from;
}

ParseResult parse_request(std::string_view buf, const Limits& limits,
                          ParseContext&, Request& out) {
    out = Request{};

    // --- Request line: METHOD SP target SP HTTP/1.x CRLF ---
    bool bare_lf = false;
    std::size_t rl_cr = find_crlf(buf, 0, bare_lf);
    if (bare_lf) {
        return err("bare LF in request line (CRLF required)");
    }
    if (rl_cr == std::string_view::npos) {
        if (pending_len(buf, 0) > limits.max_request_line) {
            return err("request line exceeds limit", 413);
        }
        return need_more();
    }
    if (rl_cr > limits.max_request_line) {
        return err("request line exceeds limit", 413);
    }
    std::string_view line = buf.substr(0, rl_cr);
    const std::size_t sp1 = line.find(' ');
    if (sp1 == std::string_view::npos) {
        return err("malformed request line");
    }
    const std::size_t sp2 = line.find(' ', sp1 + 1);
    if (sp2 == std::string_view::npos) {
        return err("malformed request line");
    }
    const std::string_view method = line.substr(0, sp1);
    const std::string_view target = line.substr(sp1 + 1, sp2 - sp1 - 1);
    const std::string_view version = line.substr(sp2 + 1);
    if (!is_token(method)) {
        return err("invalid method");
    }
    // Restrict to the implemented verbs. Methods are case-sensitive
    // (RFC 9110), so "head" is NOT HEAD; accepting only the exact set
    // stops a downstream router from routing "head" to the HEAD path
    // while is_head is false (which would ship a body on a connection
    // the client parses as bodyless and desync keep-alive).
    if (method != "GET" && method != "POST" && method != "HEAD") {
        return err("method not implemented", 501);
    }
    if (target.empty() ||
        target.find(' ') != std::string_view::npos) {
        return err("invalid request target");
    }
    if (version == "HTTP/1.1") {
        out.minor_version = 1;
    } else if (version == "HTTP/1.0") {
        out.minor_version = 0;
    } else {
        return err("unsupported HTTP version", 505);
    }
    out.method.assign(method);
    out.target.assign(target);
    out.is_head = method == "HEAD";
    if (const std::size_t q = target.find('?');
        q != std::string_view::npos) {
        out.path.assign(target.substr(0, q));
        out.query.assign(target.substr(q + 1));
    } else {
        out.path.assign(target);
    }

    // --- Headers: name ":" OWS value OWS CRLF, ending at a blank line.
    std::size_t pos = rl_cr + 2;
    const std::size_t header_start = pos;
    int content_length_seen = 0;
    bool has_te = false;
    std::uint64_t content_length = 0;
    bool have_content_length = false;
    for (;;) {
        std::size_t cr = find_crlf(buf, pos, bare_lf);
        if (bare_lf) {
            return err("bare LF in headers (CRLF required)");
        }
        if (cr == std::string_view::npos) {
            // Bound the wait: headers must terminate within the budget.
            // No complete line yet, so measure the header section so far
            // excluding any trailing CR awaiting its LF -- the same
            // measure the complete branch uses, so the two agree at the
            // cap. (The body cannot be past a line with no CRLF.)
            if (pending_len(buf, header_start) > limits.max_header_bytes) {
                return err("headers exceed limit", 413);
            }
            return need_more();
        }
        // Bound the header SECTION by its own extent (up to this line's
        // CRLF), NOT buf.size(): on a request delivered whole, the body
        // is already in buf, and counting it here would reject a tiny-
        // header request with a large (but legal) body.
        if (cr - header_start > limits.max_header_bytes) {
            return err("headers exceed limit", 413);
        }
        if (cr == pos) {
            pos += 2;  // blank line: end of headers
            break;
        }
        std::string_view hline = buf.substr(pos, cr - pos);
        if (hline.size() > limits.max_header_line) {
            return err("header line exceeds limit", 413);
        }
        if (hline[0] == ' ' || hline[0] == '\t') {
            return err("obs-fold header continuation rejected");
        }
        if (out.headers.size() >= limits.max_header_count) {
            return err("too many headers", 413);
        }
        const std::size_t colon = hline.find(':');
        if (colon == std::string_view::npos || colon == 0) {
            return err("malformed header line");
        }
        const std::string_view name = hline.substr(0, colon);
        if (!is_token(name)) {
            return err("invalid header name");
        }
        const std::string_view value = trim_ows(hline.substr(colon + 1));
        if (!valid_field_value(value)) {
            return err("control byte in header value");
        }
        if (ci_equal(name, "content-length")) {
            ++content_length_seen;
            if (content_length_seen > 1) {
                return err("duplicate Content-Length");
            }
            if (!parse_content_length(value, content_length)) {
                return err("invalid Content-Length");
            }
            have_content_length = true;
        } else if (ci_equal(name, "transfer-encoding")) {
            // A repeated Transfer-Encoding is the list `chunked, chunked`
            // (or worse); reject it for the same reason the single-line
            // list below is sole-chunked-only -- so the two spellings of
            // one message agree and cannot diverge from a front proxy.
            if (has_te) {
                return err("multiple Transfer-Encoding headers");
            }
            has_te = true;
            // Only chunked, and only as the sole/final coding.
            if (!ci_equal(trim_ows(value), "chunked")) {
                return err("unsupported Transfer-Encoding");
            }
        }
        out.headers.emplace_back(std::string(name), std::string(value));
        pos = cr + 2;
    }

    // --- Connection: keep-alive decision (needs the parsed headers). ---
    const std::string conn = out.get_header_value("connection");
    if (out.minor_version == 1) {
        out.keep_alive = !list_has_token(conn, "close");
    } else {
        out.keep_alive = list_has_token(conn, "keep-alive");
    }

    // --- Body framing. ---
    if (has_te && have_content_length) {
        return err("both Content-Length and Transfer-Encoding");
    }

    if (has_te) {
        // Chunked: [hexsize CRLF data CRLF]* 0 CRLF (trailers) CRLF.
        std::string body;
        std::size_t p = pos;
        for (;;) {
            bool blf = false;
            std::size_t cr = find_crlf(buf, p, blf);
            if (blf) {
                return err("bare LF in chunk size (CRLF required)");
            }
            if (cr == std::string_view::npos) {
                if (buf.size() - p > limits.max_header_line) {
                    return err("chunk-size line exceeds limit", 413);
                }
                return need_more();
            }
            std::string_view sizeline = buf.substr(p, cr - p);
            if (sizeline.find(';') != std::string_view::npos) {
                return err("chunk extensions not supported");
            }
            std::uint64_t csize = 0;
            if (!parse_chunk_size(sizeline, csize)) {
                return err("invalid chunk size");
            }
            p = cr + 2;
            if (csize == 0) {
                break;  // last chunk; trailers follow
            }
            if (csize > limits.max_body_bytes ||
                body.size() + csize > limits.max_body_bytes) {
                return err("chunked body exceeds limit", 413);
            }
            // Need csize bytes of data + the trailing CRLF.
            if (buf.size() < p + csize + 2) {
                return need_more();
            }
            body.append(buf.substr(p, csize));
            p += csize;
            if (buf[p] != '\r' || buf[p + 1] != '\n') {
                return err("missing CRLF after chunk data");
            }
            p += 2;
        }
        // Trailers: header lines until a blank line, bounded.
        const std::size_t trailer_start = p;
        for (;;) {
            bool blf = false;
            std::size_t cr = find_crlf(buf, p, blf);
            if (blf) {
                return err("bare LF in trailers (CRLF required)");
            }
            if (cr == std::string_view::npos) {
                if (pending_len(buf, trailer_start) >
                    limits.max_trailer_bytes) {
                    return err("trailers exceed limit", 413);
                }
                return need_more();
            }
            // Bound by the trailer section's own extent, NOT buf.size():
            // on a keep-alive connection the next pipelined request may
            // already be buffered past the trailers, and counting it
            // here would reject a valid request. (Same class as the
            // header-section bound above.)
            if (cr - trailer_start > limits.max_trailer_bytes) {
                return err("trailers exceed limit", 413);
            }
            if (cr == p) {
                p += 2;  // blank line ends trailers
                break;
            }
            p = cr + 2;  // ignore trailer content (bounded above)
        }
        out.body = std::move(body);
        ParseResult r;
        r.state = ParseState::kComplete;
        r.consumed = p;
        return r;
    }

    // Content-Length (or none -> empty body).
    const std::uint64_t clen = have_content_length ? content_length : 0;
    if (clen > limits.max_body_bytes) {
        return err("Content-Length exceeds limit", 413);
    }
    if (buf.size() - pos < clen) {
        return need_more();
    }
    out.body.assign(buf.substr(pos, static_cast<std::size_t>(clen)));
    ParseResult r;
    r.state = ParseState::kComplete;
    r.consumed = pos + static_cast<std::size_t>(clen);
    return r;
}

void Response::set_header(std::string name, std::string value) {
    headers.emplace_back(std::move(name), std::move(value));
}

void Response::set_content(std::string content, std::string content_type) {
    body = std::move(content);
    set_header("Content-Type", std::move(content_type));
}

std::string status_reason(int status) {
    switch (status) {
        case 200: return "OK";
        case 201: return "Created";
        case 202: return "Accepted";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 413: return "Content Too Large";
        case 429: return "Too Many Requests";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        default:  return {};
    }
}

namespace {

// A header name/value must not carry CR, LF, or NUL -- those would split
// the response (header injection). Returns false if `s` is unsafe.
bool header_safe(std::string_view s) {
    return s.find('\r') == std::string_view::npos &&
           s.find('\n') == std::string_view::npos &&
           s.find('\0') == std::string_view::npos;
}

}  // namespace

bool write_response(const Response& res, bool head_request,
                    bool keep_alive, std::string& out, std::string& err) {
    const std::string reason = status_reason(res.status);
    if (reason.empty()) {
        err = "unsupported status code";
        return false;
    }
    for (const auto& [k, v] : res.headers) {
        if (k.empty() || !header_safe(k) || !header_safe(v)) {
            err = "unsafe header (CR/LF/NUL or empty name)";
            return false;
        }
        if (ci_equal(k, "content-length") ||
            ci_equal(k, "transfer-encoding") ||
            ci_equal(k, "connection")) {
            err = "framing header must not be set by the caller: " + k;
            return false;
        }
    }
    out += "HTTP/1.1 ";
    out += std::to_string(res.status);
    out += ' ';
    out += reason;
    out += "\r\n";
    for (const auto& [k, v] : res.headers) {
        out += k;
        out += ": ";
        out += v;
        out += "\r\n";
    }
    out += "Content-Length: ";
    out += std::to_string(res.body.size());
    out += "\r\n";
    out += "Connection: ";
    out += keep_alive ? "keep-alive" : "close";
    out += "\r\n\r\n";
    if (!head_request) {
        out += res.body;
    }
    return true;
}

std::string encode_chunk(std::string_view data) {
    std::string out;
    char hex[17];
    const int n = std::snprintf(hex, sizeof(hex), "%zx", data.size());
    out.append(hex, static_cast<std::size_t>(n));
    out += "\r\n";
    out.append(data);
    out += "\r\n";
    return out;
}

std::string last_chunk() { return "0\r\n\r\n"; }

}  // namespace locus::http
