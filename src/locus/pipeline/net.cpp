#include "locus/pipeline/net.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
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

}  // namespace

std::optional<CidrV4> CidrV4::parse(const std::string& s) {
    std::string ip = s;
    int bits = 32;
    const auto slash = s.find('/');
    if (slash != std::string::npos) {
        ip = s.substr(0, slash);
        const std::string nb = s.substr(slash + 1);
        if (nb.empty()) {
            return std::nullopt;
        }
        char* end = nullptr;
        const long v = std::strtol(nb.c_str(), &end, 10);
        if (*end != '\0' || v < 0 || v > 32) {
            return std::nullopt;
        }
        bits = static_cast<int>(v);
    }
    in_addr a;
    if (::inet_pton(AF_INET, ip.c_str(), &a) != 1) {
        return std::nullopt;
    }
    CidrV4 c;
    c.mask = bits == 0 ? 0u : (0xFFFFFFFFu << (32 - bits));
    c.network = ntohl(a.s_addr) & c.mask;
    return c;
}

bool CidrV4::contains(const std::string& ipv4) const {
    in_addr a;
    if (::inet_pton(AF_INET, ipv4.c_str(), &a) != 1) {
        return false;
    }
    return (ntohl(a.s_addr) & mask) == network;
}

bool ip_allowed(const std::string& ipv4,
                const std::vector<CidrV4>& allow) {
    if (allow.empty()) {
        return true;  // empty allowlist = allow all
    }
    for (const auto& c : allow) {
        if (c.contains(ipv4)) {
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
    for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) {
            continue;
        }
        int one = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (::bind(fd, p->ai_addr, p->ai_addrlen) == 0 &&
            ::listen(fd, 16) == 0) {
            break;  // bound + listening
        }
        ::close(fd);
        fd = -1;
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
                ::inet_ntop(
                    AF_INET6,
                    &reinterpret_cast<sockaddr_in6*>(&ss)->sin6_addr,
                    buf, sizeof(buf));
            }
            *peer_ip = buf;
        }
        set_conn_opts(fd);
        return fd;
    }
}

int connect_to(const std::string& host, int port) {
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
    int fd = -1;
    for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) {
            set_conn_opts(fd);
            break;  // connected
        }
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);
    return fd;
}

}  // namespace locus::pipeline
