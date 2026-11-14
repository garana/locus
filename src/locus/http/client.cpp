#include "locus/http/client.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>

#include "locus/pipeline/net.hpp"

namespace locus::http {

namespace {

#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;  // SO_NOSIGPIPE is set by connect_to
#endif

/** write_all's error when the server stopped taking bytes. */
constexpr const char* kWriteTimedOut = "write timed out";
constexpr const char* kTotalTimedOut = "request timed out";

char lower(char c) {
    return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
}

bool iequals(std::string_view a, std::string_view b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(),
                      [](char x, char y) { return lower(x) == lower(y); });
}

std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) {
        s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) {
        s.remove_suffix(1);
    }
    return s;
}

/** RFC 9110 token characters, for header names. */
bool is_tchar(char c) {
    const auto u = static_cast<unsigned char>(c);
    if (u >= '0' && u <= '9') return true;
    if ((u | 0x20) >= 'a' && (u | 0x20) <= 'z') return true;
    return std::strchr("!#$%&'*+-.^_`|~", c) != nullptr && c != '\0';
}

/** field-content: VCHAR, SP, HTAB and obs-text; no control bytes. */
bool is_field_content(std::string_view v) {
    for (const char c : v) {
        const auto u = static_cast<unsigned char>(c);
        if (u == '\t' || u == ' ') continue;
        if (u < 0x21 || u == 0x7F) return false;
    }
    return true;
}

/** Parses a decimal Content-Length: digits only, no overflow. */
bool parse_length(std::string_view s, std::size_t& out) {
    if (s.empty()) return false;
    std::size_t v = 0;
    for (const char c : s) {
        if (c < '0' || c > '9') return false;
        const std::size_t d = static_cast<std::size_t>(c - '0');
        if (v > (std::numeric_limits<std::size_t>::max() - d) / 10) {
            return false;
        }
        v = v * 10 + d;
    }
    out = v;
    return true;
}

ParseResult error(std::string msg, int status = 502) {
    ParseResult r;
    r.state = ParseState::kError;
    r.error = std::move(msg);
    r.suggested_status = status;
    return r;
}

/** True if the comma list `v` has `token` (case-insensitive). */
bool has_token(std::string_view v, std::string_view token) {
    while (!v.empty()) {
        const std::size_t comma = v.find(',');
        if (iequals(trim(v.substr(0, comma)), token)) return true;
        if (comma == std::string_view::npos) break;
        v.remove_prefix(comma + 1);
    }
    return false;
}

}  // namespace

std::string ResponseHead::get_header_value(std::string_view name) const {
    for (const auto& [k, v] : headers) {
        if (iequals(k, name)) return v;
    }
    return "";
}

ParseResult parse_response_head(std::string_view buf, bool head_request,
                                const ResponseLimits& limits,
                                ResponseHead& out) {
    // A bare LF anywhere in the head is an error, so look for the
    // first LF and require a CR before it, line by line.
    ResponseHead h;
    std::size_t pos = 0;
    std::size_t header_bytes = 0;
    bool status_done = false;
    bool have_cl = false, have_te = false;
    for (;;) {
        const std::size_t lf = buf.find('\n', pos);
        if (lf == std::string_view::npos) {
            const std::size_t open = buf.size() - pos;
            if (!status_done ? open > limits.max_status_line
                             : open > limits.max_header_line ||
                                   header_bytes + open >
                                       limits.max_header_bytes) {
                return error("response head too large");
            }
            return {};  // kNeedMore
        }
        if (lf == pos || buf[lf - 1] != '\r') {
            return error("bare LF in the response head");
        }
        const std::string_view line = buf.substr(pos, lf - 1 - pos);
        pos = lf + 1;
        if (!status_done) {
            // HTTP/1.x SP 3DIGIT [SP reason]
            if (line.size() > limits.max_status_line) {
                return error("status line too long");
            }
            if (line.size() < 12 || line.substr(0, 7) != "HTTP/1." ||
                (line[7] != '0' && line[7] != '1') || line[8] != ' ') {
                return error("malformed status line");
            }
            h.minor_version = line[7] - '0';
            int code = 0;
            for (int i = 9; i < 12; ++i) {
                if (line[i] < '0' || line[i] > '9') {
                    return error("malformed status code");
                }
                code = code * 10 + (line[i] - '0');
            }
            if (code < 100 || (line.size() > 12 && line[12] != ' ')) {
                return error("malformed status code");
            }
            h.status = code;
            h.reason = std::string(
                line.size() > 13 ? line.substr(13) : std::string_view{});
            if (!is_field_content(h.reason)) {
                return error("control byte in the reason phrase");
            }
            status_done = true;
            continue;
        }
        if (line.empty()) break;  // end of the head
        if (line.size() > limits.max_header_line) {
            return error("header line too long");
        }
        header_bytes += line.size() + 2;
        if (header_bytes > limits.max_header_bytes ||
            h.headers.size() >= limits.max_header_count) {
            return error("too many header bytes or headers");
        }
        if (line.front() == ' ' || line.front() == '\t') {
            return error("obs-fold in a response header");
        }
        const std::size_t colon = line.find(':');
        if (colon == std::string_view::npos || colon == 0) {
            return error("malformed response header");
        }
        const std::string_view name = line.substr(0, colon);
        if (!std::all_of(name.begin(), name.end(), is_tchar)) {
            return error("malformed header name");
        }
        const std::string_view value = trim(line.substr(colon + 1));
        if (!is_field_content(value)) {
            return error("control byte in a header value");
        }
        if (iequals(name, "Content-Length")) {
            std::size_t n = 0;
            if (have_cl || !parse_length(value, n)) {
                return error("bad or repeated Content-Length");
            }
            have_cl = true;
            h.content_length = n;
        } else if (iequals(name, "Transfer-Encoding")) {
            if (have_te || !iequals(value, "chunked")) {
                return error("unsupported Transfer-Encoding");
            }
            have_te = true;
        }
        h.headers.emplace_back(std::string(name), std::string(value));
    }
    if (have_cl && have_te) {
        return error("both Content-Length and Transfer-Encoding");
    }
    const std::string conn = h.get_header_value("Connection");
    h.keep_alive = h.minor_version == 1 ? !has_token(conn, "close")
                                        : has_token(conn, "keep-alive");
    if (head_request || (h.status >= 100 && h.status < 200) ||
        h.status == 204 || h.status == 304) {
        h.framing = BodyFraming::kNone;
    } else if (have_te) {
        h.framing = BodyFraming::kChunked;
    } else if (have_cl) {
        if (h.content_length > limits.max_body_bytes) {
            return error("response body over the limit");
        }
        h.framing = BodyFraming::kLength;
    } else {
        h.framing = BodyFraming::kClose;
        h.keep_alive = false;
    }
    out = std::move(h);
    ParseResult r;
    r.state = ParseState::kComplete;
    r.consumed = pos;
    return r;
}

ChunkDecoder::State ChunkDecoder::feed(
    std::string_view in, const std::function<void(std::string_view)>& sink,
    std::string& rest) {
    constexpr std::size_t kMaxSizeLine = 1024;
    constexpr std::size_t kMaxTrailers = 4 * 1024;
    if (step_ == Step::kDone) {
        rest.assign(in);
        return State::kDone;
    }
    buf_.append(in);
    std::size_t at = 0;
    const auto fail = [&](const char* msg) {
        error_ = msg;
        return State::kError;
    };
    for (;;) {
        if (step_ == Step::kSize) {
            const std::size_t lf = buf_.find("\r\n", at);
            if (lf == std::string::npos) {
                if (buf_.size() - at > kMaxSizeLine) {
                    return fail("chunk size line too long");
                }
                break;
            }
            // Hex size; a chunk extension after ';' is ignored.
            std::string_view s(buf_.data() + at, lf - at);
            s = trim(s.substr(0, s.find(';')));
            if (s.empty()) return fail("empty chunk size");
            std::size_t n = 0;
            for (const char c : s) {
                int d;
                if (c >= '0' && c <= '9') d = c - '0';
                else if ((c | 0x20) >= 'a' && (c | 0x20) <= 'f')
                    d = (c | 0x20) - 'a' + 10;
                else return fail("bad chunk size");
                if (n > (std::numeric_limits<std::size_t>::max() >> 4)) {
                    return fail("chunk size overflow");
                }
                n = (n << 4) | static_cast<std::size_t>(d);
            }
            at = lf + 2;
            if (n == 0) {
                step_ = Step::kTrailers;
            } else {
                if (n > max_body_ - std::min(total_, max_body_)) {
                    return fail("response body over the limit");
                }
                left_ = n;
                step_ = Step::kData;
            }
        } else if (step_ == Step::kData) {
            const std::size_t take = std::min(left_, buf_.size() - at);
            if (take == 0) break;
            sink(std::string_view(buf_.data() + at, take));
            at += take;
            left_ -= take;
            total_ += take;
            if (left_ > 0) break;
            step_ = Step::kDataEnd;
        } else if (step_ == Step::kDataEnd) {
            if (buf_.size() - at < 2) break;
            if (buf_.compare(at, 2, "\r\n") != 0) {
                return fail("chunk data not followed by CRLF");
            }
            at += 2;
            step_ = Step::kSize;
        } else if (step_ == Step::kTrailers) {
            const std::size_t lf = buf_.find("\r\n", at);
            if (lf == std::string::npos) {
                if (buf_.size() - at > kMaxTrailers) {
                    return fail("trailers too long");
                }
                break;
            }
            const bool blank = lf == at;
            at = lf + 2;
            if (blank) {
                step_ = Step::kDone;
                rest.assign(buf_, at, std::string::npos);
                buf_.clear();
                return State::kDone;
            }
        }
    }
    buf_.erase(0, at);
    return State::kNeedMore;
}

Client::Client(std::string host, int port, ClientOptions opt)
    : host_(std::move(host)), port_(port), opt_(opt) {}

Client::~Client() { close(); }

void Client::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    buf_.clear();
    active_ = false;
}

bool Client::expired() const {
    return opt_.total_timeout_ms > 0 &&
           std::chrono::steady_clock::now() >= deadline_;
}

int Client::wait_ms(int ms) const {
    if (opt_.total_timeout_ms <= 0) return ms > 0 ? ms : -1;
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                          deadline_ - std::chrono::steady_clock::now())
                          .count();
    if (left <= 0) return 0;
    const int l = static_cast<int>(std::min<long long>(left, INT32_MAX));
    return ms > 0 ? std::min(ms, l) : l;
}

bool Client::total_bound(int ms, int t) const {
    return opt_.total_timeout_ms > 0 && (ms <= 0 || t < ms);
}

bool Client::connect(std::string& err) {
    if (expired()) {
        err = kTotalTimedOut;
        return false;
    }
    // connect_to takes 0 for no limit.
    const int t = wait_ms(opt_.connect_timeout_ms);
    fd_ = pipeline::connect_to(host_, port_, t < 0 ? 0 : std::max(t, 1));
    if (fd_ < 0) {
        err = "connect to " + host_ + ":" + std::to_string(port_) +
              " failed";
        return false;
    }
    return true;
}

bool Client::stale() const {
    pollfd p{fd_, POLLIN, 0};
    if (::poll(&p, 1, 0) <= 0) return false;  // nothing pending: alive
    // Readable while idle: the server closed (0) or sent bytes no
    // request asked for; either way the connection is not reusable.
    char c;
    const ssize_t n = ::recv(fd_, &c, 1, MSG_PEEK | MSG_DONTWAIT);
    return n >= 0 || (errno != EAGAIN && errno != EWOULDBLOCK);
}

bool Client::write_all(std::string_view data, std::string& err) {
    // Non-blocking sends with a poll between them, so a server that
    // stops reading a large body trips the write deadline. The socket
    // is non-blocking only for the write (MSG_DONTWAIT is not honoured
    // on a blocking socket everywhere).
    const int flags = ::fcntl(fd_, F_GETFL, 0);
    if (flags >= 0) ::fcntl(fd_, F_SETFL, flags | O_NONBLOCK);
    struct Restore {
        int fd, flags;
        ~Restore() {
            if (flags >= 0) ::fcntl(fd, F_SETFL, flags);
        }
    } restore{fd_, flags};
    std::size_t off = 0;
    while (off < data.size()) {
        const ssize_t n =
            ::send(fd_, data.data() + off, data.size() - off, kSendFlags);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            pollfd p{fd_, POLLOUT, 0};
            const int t = wait_ms(opt_.write_timeout_ms);
            const int r = ::poll(&p, 1, t);
            if (r < 0 && errno == EINTR) continue;
            if (r == 0) {
                err = total_bound(opt_.write_timeout_ms, t) ? kTotalTimedOut
                                                            : kWriteTimedOut;
                return false;
            }
            if (r < 0) {
                err = std::string("poll failed: ") + std::strerror(errno);
                return false;
            }
            continue;
        }
        if (n <= 0) {
            err = std::string("write failed: ") + std::strerror(errno);
            return false;
        }
        off += static_cast<std::size_t>(n);
    }
    return true;
}

long Client::read_some(std::string& err) {
    for (;;) {
        // Checked before the poll too: a server that trickles bytes
        // keeps the poll ready, and must still stop at the total.
        if (expired()) {
            err = kTotalTimedOut;
            return -1;
        }
        pollfd p{fd_, POLLIN, 0};
        const int t = wait_ms(opt_.read_timeout_ms);
        const int r = ::poll(&p, 1, t);
        if (r < 0 && errno == EINTR) continue;
        if (r == 0) {
            // By which limit: poll may wake a little early, so the
            // clock alone cannot tell.
            err = total_bound(opt_.read_timeout_ms, t) ? kTotalTimedOut
                                                       : "read timed out";
            return -1;
        }
        if (r < 0) {
            err = std::string("poll failed: ") + std::strerror(errno);
            return -1;
        }
        char tmp[64 * 1024];
        const ssize_t n = ::read(fd_, tmp, sizeof(tmp));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) {
            err = std::string("read failed: ") + std::strerror(errno);
            return -1;
        }
        buf_.append(tmp, static_cast<std::size_t>(n));
        return static_cast<long>(n);
    }
}

ClientResponse Client::send(const ClientRequest& req) {
    ClientResponse res = begin(req);
    if (!res.ok) return res;
    const std::size_t cap = opt_.limits.max_body_bytes;
    bool over = false;
    res.ok = read_body(
        [&](std::string_view d) {
            if (res.body.size() + d.size() > cap) {
                over = true;
                return false;
            }
            res.body.append(d);
            return true;
        },
        res.error);
    if (over) res.error = "response body over the limit";
    return res;
}

ClientResponse Client::stream(
    const ClientRequest& req,
    const std::function<bool(const ResponseHead&)>& on_head,
    const std::function<bool(std::string_view)>& on_body) {
    ClientResponse res = begin(req);
    if (!res.ok) return res;
    if (on_head && !on_head(res.head)) {
        close();
        res.ok = false;
        res.error = "stopped by the caller";
        return res;
    }
    res.ok = read_body(on_body ? on_body
                               : [](std::string_view) { return true; },
                       res.error);
    return res;
}

ClientResponse Client::begin(const ClientRequest& req) {
    if (active_) close();  // the last body was not read: drop it
    ClientResponse res;
    // The request bytes; header names and values must not carry CR, LF
    // or NUL (the same injection guard as the response writer).
    const auto unsafe = [](std::string_view s) {
        return s.find_first_of(std::string_view("\r\n\0", 3)) !=
               std::string_view::npos;
    };
    if (unsafe(req.method) || unsafe(req.target) ||
        req.target.find(' ') != std::string::npos) {
        res.error = "bad method or target";
        return res;
    }
    std::string out = req.method + " " + req.target + " HTTP/1.1\r\n";
    // An IPv6 literal goes in brackets (RFC 9110 section 7.2).
    const bool v6 =
        host_.find(':') != std::string::npos && host_.front() != '[';
    out += "Host: " + (v6 ? "[" + host_ + "]" : host_) + ":" +
           std::to_string(port_) + "\r\n";
    for (const auto& [k, v] : req.headers) {
        if (k.empty() || unsafe(k) || unsafe(v) || iequals(k, "Host") ||
            iequals(k, "Content-Length") ||
            iequals(k, "Transfer-Encoding") || iequals(k, "Connection")) {
            res.error = "bad or reserved request header: " + k;
            return res;
        }
        out += k + ": " + v + "\r\n";
    }
    if (!req.body.empty() || req.method == "POST" || req.method == "PUT") {
        out += "Content-Length: " + std::to_string(req.body.size()) + "\r\n";
    }
    out += "\r\n";
    out += req.body;
    const bool head_req = req.method == "HEAD";
    // Safe to send twice: the server's state does not change (RFC 9110
    // section 9.2.2).
    const bool idempotent = req.method == "GET" || req.method == "HEAD" ||
                            req.method == "PUT" || req.method == "DELETE" ||
                            req.method == "OPTIONS";
    // Interim 1xx responses before the real head, at most.
    constexpr int kMaxInterim = 8;

    // A kept-alive connection the server closed while it sat idle is
    // dropped before the request goes on it. If the close still races
    // the send (the write fails, or the read sees the end with nothing
    // read), the request is sent once more on a new connection, unless
    // it is not idempotent and the write went through.
    if (fd_ >= 0 && stale()) close();
    if (opt_.total_timeout_ms > 0) {
        deadline_ = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(opt_.total_timeout_ms);
    }
    for (int attempt = 0;; ++attempt) {
        const bool reused = fd_ >= 0;
        if (!reused) {
            buf_.clear();
            if (!connect(res.error)) return res;
        }
        res.connected = true;
        std::string err;
        if (!write_all(out, err)) {
            close();
            // A closed connection never had the whole request, so it is
            // sent again on a new one. A timeout means the server is
            // slow, not gone: sending again would give it the request
            // twice and make the caller wait two timeouts.
            if (reused && attempt == 0 && err != kWriteTimedOut) continue;
            res.error = err;
            return res;
        }
        // The head, skipping interim 1xx responses (but not 101).
        ResponseHead head;
        bool got_any = !buf_.empty();
        int interim = 0;
        for (;;) {
            const ParseResult pr =
                parse_response_head(buf_, head_req, opt_.limits, head);
            if (pr.state == ParseState::kError) {
                close();
                res.error = "bad response: " + pr.error;
                return res;
            }
            if (pr.state == ParseState::kComplete) {
                buf_.erase(0, pr.consumed);
                if (head.status >= 100 && head.status < 200 &&
                    head.status != 101) {
                    if (++interim > kMaxInterim) {
                        close();
                        res.error = "too many interim responses";
                        return res;
                    }
                    continue;  // interim; the real head follows
                }
                break;
            }
            const long n = read_some(err);
            if (n <= 0) {
                close();
                if (reused && attempt == 0 && !got_any && idempotent) break;
                res.error = n == 0 ? "connection closed before a response"
                                   : err;
                return res;
            }
            got_any = true;
        }
        if (fd_ < 0) continue;  // the stale-connection retry
        res.head = head;
        pending_ = head;
        active_ = true;
        res.ok = true;
        return res;
    }
}

bool Client::read_body(const std::function<bool(std::string_view)>& on_body,
                       std::string& err) {
    if (!active_) {
        err = "no response in progress";
        return false;
    }
    active_ = false;
    bool stopped = false;
    const auto deliver = [&](std::string_view d) {
        if (stopped || d.empty()) return;
        if (!on_body(d)) stopped = true;
    };
    const auto fail = [&](long n, std::string why) {
        close();
        err = n == 0 ? std::move(why) : err;
        return false;
    };
    switch (pending_.framing) {
        case BodyFraming::kNone:
            break;
        case BodyFraming::kLength: {
            std::size_t left = pending_.content_length;
            for (;;) {
                const std::size_t take = std::min(left, buf_.size());
                deliver(std::string_view(buf_).substr(0, take));
                buf_.erase(0, take);
                left -= take;
                if (left == 0 || stopped) break;
                const long n = read_some(err);
                if (n <= 0) return fail(n, "connection closed mid-body");
            }
            break;
        }
        case BodyFraming::kChunked: {
            ChunkDecoder dec(opt_.limits.max_body_bytes);
            std::string pending = std::move(buf_);
            buf_.clear();
            for (;;) {
                std::string rest;
                const auto st = dec.feed(pending, deliver, rest);
                if (st == ChunkDecoder::State::kError) {
                    close();
                    err = "bad chunked body: " + dec.error();
                    return false;
                }
                if (st == ChunkDecoder::State::kDone) {
                    buf_ = std::move(rest);
                    break;
                }
                if (stopped) break;
                buf_.clear();
                const long n = read_some(err);
                if (n <= 0) return fail(n, "connection closed mid-body");
                pending = std::move(buf_);
                buf_.clear();
            }
            break;
        }
        case BodyFraming::kClose: {
            for (;;) {
                deliver(buf_);
                buf_.clear();
                if (stopped) break;
                const long n = read_some(err);
                if (n == 0) break;  // the end of the body
                if (n < 0) return fail(n, "");
            }
            break;
        }
    }
    if (stopped) {
        close();
        if (err.empty()) err = "stopped by the caller";
        return false;
    }
    if (!pending_.keep_alive) close();
    return true;
}

}  // namespace locus::http
