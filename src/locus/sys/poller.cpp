#include "locus/sys/poller.hpp"

#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(__linux__)
#include <sys/epoll.h>
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || \
    defined(__NetBSD__)
#define LOCUS_POLLER_KQUEUE 1
#include <sys/event.h>
#include <sys/time.h>
#include <sys/types.h>
#else
#error "Poller: no epoll or kqueue backend for this platform"
#endif

namespace locus::sys {

namespace {
[[noreturn]] void fail(const char* what) {
    throw std::runtime_error(std::string("Poller: ") + what + ": " +
                             std::strerror(errno));
}
}  // namespace

#if defined(LOCUS_POLLER_KQUEUE)

Poller::Poller() {
    poll_fd_ = ::kqueue();
    if (poll_fd_ < 0) {
        fail("kqueue");
    }
}

Poller::~Poller() {
    if (poll_fd_ >= 0) {
        ::close(poll_fd_);
    }
}

void Poller::add_read(int fd) {
    struct kevent ev;
    EV_SET(&ev, fd, EVFILT_READ, EV_ADD, 0, 0, nullptr);
    if (::kevent(poll_fd_, &ev, 1, nullptr, 0, nullptr) < 0) {
        fail("kevent EV_ADD");
    }
}

void Poller::remove(int fd) {
    struct kevent ev;
    EV_SET(&ev, fd, EVFILT_READ, EV_DELETE, 0, 0, nullptr);
    // Ignore errors: the fd may already be closed (kqueue drops a
    // closed fd's registration) or never have been added.
    ::kevent(poll_fd_, &ev, 1, nullptr, 0, nullptr);
}

int Poller::wait(std::vector<Event>& events, int timeout_ms) {
    events.clear();
    struct kevent evs[64];
    // Track a deadline so an EINTR retry resumes with the REMAINING
    // budget rather than restarting timeout_ms (honoring "waits up to
    // timeout_ms"). -1 stays infinite.
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(
                              timeout_ms < 0 ? 0 : timeout_ms);
    for (;;) {
        timespec ts;
        timespec* tsp = nullptr;
        if (timeout_ms >= 0) {
            auto rem = std::chrono::duration_cast<std::chrono::milliseconds>(
                           deadline - std::chrono::steady_clock::now())
                           .count();
            if (rem < 0) {
                rem = 0;
            }
            ts.tv_sec = static_cast<time_t>(rem / 1000);
            ts.tv_nsec = static_cast<long>(rem % 1000) * 1000000L;
            tsp = &ts;
        }
        const int n = ::kevent(poll_fd_, nullptr, 0, evs, 64, tsp);
        if (n < 0) {
            if (errno == EINTR) {
                continue;  // retry; a self-pipe is how callers wake us
            }
            return -1;
        }
        for (int i = 0; i < n; ++i) {
            Event e;
            e.fd = static_cast<int>(evs[i].ident);
            e.error = (evs[i].flags & EV_ERROR) != 0;
            e.hangup = (evs[i].flags & EV_EOF) != 0;
            // On EV_ERROR the data field is an errno, not a byte count.
            e.readable = e.error ? false : (evs[i].data > 0 || e.hangup);
            events.push_back(e);
        }
        return n;
    }
}

#else  // epoll (Linux)

Poller::Poller() {
    poll_fd_ = ::epoll_create1(EPOLL_CLOEXEC);
    if (poll_fd_ < 0) {
        fail("epoll_create1");
    }
}

Poller::~Poller() {
    if (poll_fd_ >= 0) {
        ::close(poll_fd_);
    }
}

void Poller::add_read(int fd) {
    epoll_event ev;
    std::memset(&ev, 0, sizeof(ev));
    ev.events = EPOLLIN;
    ev.data.fd = fd;
    if (::epoll_ctl(poll_fd_, EPOLL_CTL_ADD, fd, &ev) < 0) {
        if (errno == EEXIST) {
            return;  // already registered: treat add as idempotent
        }
        fail("epoll_ctl ADD");
    }
}

void Poller::remove(int fd) {
    // Ignore errors: the fd may already be closed (epoll drops a closed
    // fd automatically) or never have been added.
    ::epoll_ctl(poll_fd_, EPOLL_CTL_DEL, fd, nullptr);
}

int Poller::wait(std::vector<Event>& events, int timeout_ms) {
    events.clear();
    epoll_event evs[64];
    // Track a deadline so an EINTR retry resumes with the REMAINING
    // budget rather than restarting timeout_ms (honoring "waits up to
    // timeout_ms"). -1 stays infinite.
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(
                              timeout_ms < 0 ? 0 : timeout_ms);
    for (;;) {
        int to = timeout_ms;
        if (timeout_ms >= 0) {
            auto rem = std::chrono::duration_cast<std::chrono::milliseconds>(
                           deadline - std::chrono::steady_clock::now())
                           .count();
            if (rem < 0) {
                rem = 0;
            }
            to = static_cast<int>(rem);
        }
        const int n = ::epoll_wait(poll_fd_, evs, 64, to);
        if (n < 0) {
            if (errno == EINTR) {
                continue;  // retry; a self-pipe is how callers wake us
            }
            return -1;
        }
        for (int i = 0; i < n; ++i) {
            Event e;
            e.fd = evs[i].data.fd;
            const auto bits = evs[i].events;
            e.error = (bits & EPOLLERR) != 0;
            e.hangup = (bits & EPOLLHUP) != 0;
            // EPOLLIN or a hangup both mean a read won't block (it may
            // return 0 at EOF); on an error we leave readable false.
            e.readable =
                e.error ? false : (bits & (EPOLLIN | EPOLLHUP)) != 0;
            events.push_back(e);
        }
        return n;
    }
}

#endif

}  // namespace locus::sys
