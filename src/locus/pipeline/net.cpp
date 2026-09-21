#include "locus/pipeline/net.hpp"

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

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

int accept_one(int listen_fd) {
    for (;;) {
        const int fd = ::accept(listen_fd, nullptr, nullptr);
        if (fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
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
