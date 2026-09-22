#include "locus/pipeline/net.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace locus::pipeline {

namespace {

// TCP_NODELAY so per-token activations are not delayed by Nagle;
// SO_NOSIGPIPE (where available) so a write to a closed peer returns an
// error rather than raising SIGPIPE.
void set_conn_opts(int fd) {
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
}

int port_of(int fd) {
    sockaddr_storage ss;
    socklen_t sl = sizeof(ss);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&ss), &sl) != 0) {
        return -1;
    }
    if (ss.ss_family == AF_INET) {
        return ntohs(reinterpret_cast<sockaddr_in*>(&ss)->sin_port);
    }
    if (ss.ss_family == AF_INET6) {
        return ntohs(reinterpret_cast<sockaddr_in6*>(&ss)->sin6_port);
    }
    return -1;
}

// Parses a numeric address into `out` (big-endian, v4 uses first 4)
// and its family. @returns false if `s` is not a valid v4/v6 literal.
bool parse_addr(const std::string& s, AddrFamily& fam,
                std::array<std::uint8_t, 16>& out) {
    out.fill(0);
    if (s.find(':') != std::string::npos) {
        in6_addr a6;
        if (::inet_pton(AF_INET6, s.c_str(), &a6) != 1) {
            return false;
        }
        std::memcpy(out.data(), &a6, 16);
        fam = AddrFamily::kV6;
        return true;
    }
    in_addr a4;
    if (::inet_pton(AF_INET, s.c_str(), &a4) != 1) {
        return false;
    }
    std::memcpy(out.data(), &a4, 4);
    fam = AddrFamily::kV4;
    return true;
}

// Zeroes the address bits beyond `bits` in the first `len` bytes.
void mask_bytes(std::uint8_t* p, int bits, int len) {
    for (int i = 0; i < len; ++i) {
        if (bits >= (i + 1) * 8) {
            continue;  // whole byte kept
        }
        if (bits <= i * 8) {
            p[i] = 0;  // whole byte dropped
            continue;
        }
        const int rem = bits - i * 8;  // 1..7 kept bits
        p[i] &= static_cast<std::uint8_t>(0xFF << (8 - rem));
    }
}

// @returns true if the first `bits` bits of a and b are equal.
bool prefix_match(const std::uint8_t* a, const std::uint8_t* b,
                  int bits) {
    const int full = bits / 8;
    const int rem = bits % 8;
    if (full > 0 && std::memcmp(a, b, full) != 0) {
        return false;
    }
    if (rem != 0) {
        const std::uint8_t m = static_cast<std::uint8_t>(0xFF << (8 - rem));
        if ((a[full] & m) != (b[full] & m)) {
            return false;
        }
    }
    return true;
}

}  // namespace

std::optional<Cidr> Cidr::parse(const std::string& s) {
    std::string addr = s;
    // Keep the prefix as a long until it is range-checked against the
    // family's max: narrowing to int first would let a value that
    // wraps into range (e.g. 2^32 + 8) be accepted as a smaller prefix.
    long bits = -1;
    const auto slash = s.find('/');
    if (slash != std::string::npos) {
        addr = s.substr(0, slash);
        const std::string nb = s.substr(slash + 1);
        if (nb.empty()) {
            return std::nullopt;
        }
        char* end = nullptr;
        const long v = std::strtol(nb.c_str(), &end, 10);
        if (*end != '\0' || v < 0) {
            return std::nullopt;
        }
        bits = v;
    }
    Cidr c;
    if (!parse_addr(addr, c.family, c.net)) {
        return std::nullopt;
    }
    const long max_bits = c.family == AddrFamily::kV4 ? 32 : 128;
    if (bits < 0) {
        bits = max_bits;  // bare address == full-length prefix
    }
    if (bits > max_bits) {  // checked on the long, before narrowing
        return std::nullopt;
    }
    c.bits = static_cast<std::uint8_t>(bits);
    mask_bytes(c.net.data(), static_cast<int>(bits),
               c.family == AddrFamily::kV4 ? 4 : 16);
    return c;
}

bool Cidr::contains(const std::string& ip) const {
    AddrFamily fam;
    std::array<std::uint8_t, 16> b;
    if (!parse_addr(ip, fam, b)) {
        return false;
    }
    if (fam != family) {
        return false;
    }
    return prefix_match(net.data(), b.data(), bits);
}

bool ip_allowed(const std::string& ip, const std::vector<Cidr>& allow) {
    if (allow.empty()) {
        return true;  // empty allowlist = allow all
    }
    for (const auto& c : allow) {
        if (c.contains(ip)) {
            return true;
        }
    }
    return false;
}

int listen_on(const std::string& host, int port, int* out_port) {
    addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    const std::string port_str = std::to_string(port);
    addrinfo* res = nullptr;
    if (::getaddrinfo(host.empty() ? nullptr : host.c_str(),
                      port_str.c_str(), &hints, &res) != 0) {
        return -1;
    }
    int fd = -1;
    // Prefer an IPv6 (dual-stack) socket so the wildcard bind behaves
    // the same regardless of the host's getaddrinfo ordering. Pass 0
    // tries AF_INET6 candidates; pass 1 the rest.
    for (int pass = 0; pass < 2 && fd < 0; ++pass) {
        for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
            const bool is6 = p->ai_family == AF_INET6;
            if ((pass == 0) != is6) {
                continue;
            }
            fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
            if (fd < 0) {
                continue;
            }
            int one = 1;
            ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one,
                         sizeof(one));
            if (is6) {
                int v6only = 0;  // accept IPv4-mapped too (dual-stack)
                ::setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &v6only,
                             sizeof(v6only));
            }
            if (::bind(fd, p->ai_addr, p->ai_addrlen) == 0 &&
                ::listen(fd, 16) == 0) {
                break;
            }
            ::close(fd);
            fd = -1;
        }
    }
    ::freeaddrinfo(res);
    if (fd < 0) {
        return -1;
    }
    if (out_port != nullptr) {
        *out_port = port_of(fd);
    }
    return fd;
}

int accept_one(int listen_fd, std::string* peer_ip) {
    for (;;) {
        sockaddr_storage ss;
        socklen_t sl = sizeof(ss);
        const int fd =
            ::accept(listen_fd, reinterpret_cast<sockaddr*>(&ss), &sl);
        if (fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (peer_ip != nullptr) {
            char buf[INET6_ADDRSTRLEN] = {0};
            if (ss.ss_family == AF_INET) {
                ::inet_ntop(
                    AF_INET,
                    &reinterpret_cast<sockaddr_in*>(&ss)->sin_addr, buf,
                    sizeof(buf));
            } else if (ss.ss_family == AF_INET6) {
                auto* s6 = reinterpret_cast<sockaddr_in6*>(&ss);
                if (IN6_IS_ADDR_V4MAPPED(&s6->sin6_addr)) {
                    // ::ffff:a.b.c.d -> a.b.c.d, so an IPv4 rule matches
                    // an IPv4 client on a dual-stack socket.
                    ::inet_ntop(AF_INET,
                                &s6->sin6_addr.s6_addr[12], buf,
                                sizeof(buf));
                } else {
                    ::inet_ntop(AF_INET6, &s6->sin6_addr, buf,
                                sizeof(buf));
                }
            }
            *peer_ip = buf;
        }
        set_conn_opts(fd);
        return fd;
    }
}

namespace {

using Clock = std::chrono::steady_clock;

// Connects fd to (addr,len), giving up at `deadline` (time_point::max()
// means block with the OS default). Retries poll on EINTR. Returns true
// on success; leaves fd in blocking mode either way.
bool connect_within(int fd, const sockaddr* addr, socklen_t len,
                    Clock::time_point deadline) {
    if (deadline == Clock::time_point::max()) {
        return ::connect(fd, addr, len) == 0;  // blocking
    }
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        return false;
    }
    bool ok = false;
    if (::connect(fd, addr, len) == 0) {
        ok = true;  // connected immediately
    } else if (errno == EINPROGRESS) {
        for (;;) {
            const auto now = Clock::now();
            long rem = std::chrono::duration_cast<
                           std::chrono::milliseconds>(deadline - now)
                           .count();
            if (rem < 0) {
                rem = 0;
            }
            pollfd pfd{fd, POLLOUT, 0};
            const int pr = ::poll(&pfd, 1, static_cast<int>(rem));
            if (pr < 0) {
                if (errno == EINTR) {
                    continue;  // resume with the remaining budget
                }
                break;  // poll error
            }
            if (pr == 0) {
                break;  // timed out
            }
            int soerr = 0;
            socklen_t sl = sizeof(soerr);
            if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &sl) ==
                    0 &&
                soerr == 0) {
                ok = true;  // connection completed cleanly
            }
            break;
        }
    }
    ::fcntl(fd, F_SETFL, flags);  // restore blocking
    return ok;
}

}  // namespace

int connect_to(const std::string& host, int port, int timeout_ms) {
    addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    const std::string port_str = std::to_string(port);
    addrinfo* res = nullptr;
    if (::getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) !=
        0) {
        return -1;
    }
    // One deadline shared across all getaddrinfo candidates, so
    // timeout_ms bounds the whole call rather than each address (a
    // dual-stack host with a blackholed address otherwise costs up to
    // 2x). max() == block with the OS default.
    const Clock::time_point deadline =
        timeout_ms > 0
            ? Clock::now() + std::chrono::milliseconds(timeout_ms)
            : Clock::time_point::max();
    int fd = -1;
    for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
        if (timeout_ms > 0 && Clock::now() >= deadline) {
            break;  // budget spent
        }
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (connect_within(fd, p->ai_addr, p->ai_addrlen, deadline)) {
            set_conn_opts(fd);
            break;
        }
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);
    return fd;
}

void set_recv_timeout(int fd, int ms) {
    // {0,0} clears the timeout (block indefinitely).
    timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}

void set_keepalive(int fd, int idle_s, int intvl_s, int count) {
    const int on = idle_s > 0 ? 1 : 0;
    ::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &on, sizeof(on));
    if (on == 0) {
        return;  // keepalive disabled
    }
#if defined(TCP_KEEPIDLE)
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle_s,
                 sizeof(idle_s));  // Linux: idle seconds
#elif defined(TCP_KEEPALIVE)
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPALIVE, &idle_s,
                 sizeof(idle_s));  // macOS/BSD: idle seconds
#endif
#if defined(TCP_KEEPINTVL)
    if (intvl_s > 0) {
        ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl_s,
                     sizeof(intvl_s));
    }
#endif
#if defined(TCP_KEEPCNT)
    if (count > 0) {
        ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &count,
                     sizeof(count));
    }
#endif
}

}  // namespace locus::pipeline
