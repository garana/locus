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

// Dir is a bit mask (kReadWrite == kRead | kWrite); these pull the two
// directions back out.
bool wants_read(Poller::Dir d) {
    return (static_cast<unsigned>(d) & static_cast<unsigned>(
                Poller::Dir::kRead)) != 0;
}
bool wants_write(Poller::Dir d) {
    return (static_cast<unsigned>(d) & static_cast<unsigned>(
                Poller::Dir::kWrite)) != 0;
}
}  // namespace

#if defined(LOCUS_POLLER_KQUEUE)

namespace {
// Applies one kqueue filter change. An EV_DELETE of a filter that is not
// set yields ENOENT, which is tolerated so modify()/remove() are safe to
// call regardless of which filters were present. Any other failure (and
// any EV_ADD failure) throws.
void kq_change(int kq, int fd, int16_t filter, uint16_t flags) {
    struct kevent ev;
    EV_SET(&ev, fd, filter, flags, 0, 0, nullptr);
    if (::kevent(kq, &ev, 1, nullptr, 0, nullptr) < 0) {
        if ((flags & EV_DELETE) != 0 && errno == ENOENT) {
            return;  // deleting an absent filter is a no-op
        }
        fail("kevent");
    }
}
}  // namespace

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

void Poller::add(int fd, Dir dir) {
    if (dir_.find(fd) != dir_.end()) {
        throw std::runtime_error(
            "Poller: add() on an already-registered fd (use modify)");
    }
    if (wants_read(dir)) {
        kq_change(poll_fd_, fd, EVFILT_READ, EV_ADD);
    }
    if (wants_write(dir)) {
        kq_change(poll_fd_, fd, EVFILT_WRITE, EV_ADD);
    }
    dir_[fd] = dir;
}

void Poller::modify(int fd, Dir dir) {
    const auto it = dir_.find(fd);
    if (it == dir_.end()) {
        throw std::runtime_error("Poller: modify() on an unregistered fd");
    }
    const Dir old = it->second;
    // Touch ONLY the filters that change. kqueue has no "set the mask"
    // call, and re-issuing EV_ADD on a filter that is already registered
    // resets its pending state -- which would silently drop an
    // already-readable fd when narrowing kReadWrite -> kRead. So add a
    // newly-wanted filter, delete a no-longer-wanted one, and leave an
    // unchanged filter untouched.
    if (wants_read(dir) && !wants_read(old)) {
        kq_change(poll_fd_, fd, EVFILT_READ, EV_ADD);
    } else if (!wants_read(dir) && wants_read(old)) {
        kq_change(poll_fd_, fd, EVFILT_READ, EV_DELETE);
    }
    if (wants_write(dir) && !wants_write(old)) {
        kq_change(poll_fd_, fd, EVFILT_WRITE, EV_ADD);
    } else if (!wants_write(dir) && wants_write(old)) {
        kq_change(poll_fd_, fd, EVFILT_WRITE, EV_DELETE);
    }
    it->second = dir;
}

void Poller::remove(int fd) {
    dir_.erase(fd);
    // Best-effort delete of BOTH filters, each independently so one
    // missing filter does not stop the other from being removed, and all
    // errors ignored: the fd may already be closed (kqueue drops a closed
    // fd's registration) or never have carried that filter.
    struct kevent ev;
    EV_SET(&ev, fd, EVFILT_READ, EV_DELETE, 0, 0, nullptr);
    ::kevent(poll_fd_, &ev, 1, nullptr, 0, nullptr);
    EV_SET(&ev, fd, EVFILT_WRITE, EV_DELETE, 0, 0, nullptr);
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
        // A fd watched kReadWrite can return two kevents (one per filter)
        // in a single wait; coalesce them into one Event per fd so the
        // caller sees each ready fd once with both flags set.
        for (int i = 0; i < n; ++i) {
            const int fd = static_cast<int>(evs[i].ident);
            Event* e = nullptr;
            for (auto& x : events) {
                if (x.fd == fd) {
                    e = &x;
                    break;
                }
            }
            if (e == nullptr) {
                events.push_back(Event{});
                e = &events.back();
                e->fd = fd;
            }
            const bool eof = (evs[i].flags & EV_EOF) != 0;
            const bool err = (evs[i].flags & EV_ERROR) != 0;
            if (err) {
                e->error = true;
            }
            if (eof) {
                e->hangup = true;
            }
            if (evs[i].filter == EVFILT_READ) {
                // EOF makes a read return 0 without blocking, so it still
                // counts as readable; on EV_ERROR data is an errno, not a
                // byte count, so leave readable false.
                e->readable = err ? false : (evs[i].data > 0 || eof);
            } else if (evs[i].filter == EVFILT_WRITE) {
                // data is the free space in the send buffer. EV_EOF on the
                // write side is a failed/closed connect, so NOT writable:
                // the caller checks getsockopt(SO_ERROR) or learns from
                // the write. (Asymmetry with readable is deliberate -- you
                // read EOF to learn the peer is gone; you do not write it.)
                e->writable = err ? false : (!eof && evs[i].data > 0);
            }
        }
        return static_cast<int>(events.size());
    }
}

#else  // epoll (Linux)

namespace {
// epoll event mask for a direction.
uint32_t epoll_mask(Poller::Dir dir) {
    uint32_t m = 0;
    if (wants_read(dir)) {
        m |= EPOLLIN;
    }
    if (wants_write(dir)) {
        m |= EPOLLOUT;
    }
    return m;
}
}  // namespace

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

void Poller::add(int fd, Dir dir) {
    if (dir_.find(fd) != dir_.end()) {
        throw std::runtime_error(
            "Poller: add() on an already-registered fd (use modify)");
    }
    epoll_event ev;
    std::memset(&ev, 0, sizeof(ev));
    ev.events = epoll_mask(dir);
    ev.data.fd = fd;
    if (::epoll_ctl(poll_fd_, EPOLL_CTL_ADD, fd, &ev) < 0) {
        fail("epoll_ctl ADD");
    }
    dir_[fd] = dir;
}

void Poller::modify(int fd, Dir dir) {
    const auto it = dir_.find(fd);
    if (it == dir_.end()) {
        throw std::runtime_error("Poller: modify() on an unregistered fd");
    }
    epoll_event ev;
    std::memset(&ev, 0, sizeof(ev));
    ev.events = epoll_mask(dir);
    ev.data.fd = fd;
    // One MOD replaces the whole mask (and preserves level-triggered
    // readiness), so narrowing kReadWrite -> kRead is a single call with
    // no filter-reset hazard -- unlike kqueue, which is why that backend
    // diffs the filters instead.
    if (::epoll_ctl(poll_fd_, EPOLL_CTL_MOD, fd, &ev) < 0) {
        fail("epoll_ctl MOD");
    }
    it->second = dir;
}

void Poller::remove(int fd) {
    dir_.erase(fd);
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
        // epoll reports one event per fd with a combined mask, so no
        // coalescing is needed (unlike kqueue's per-filter events).
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
            // EPOLLOUT means a write won't block. A failed non-blocking
            // connect raises EPOLLERR alongside EPOLLOUT, and error
            // suppresses writable, so it lands as writable=false /
            // error=true -- pushing the caller to getsockopt(SO_ERROR)
            // rather than writing into a dead socket.
            e.writable = e.error ? false : (bits & EPOLLOUT) != 0;
            events.push_back(e);
        }
        return n;
    }
}

#endif

}  // namespace locus::sys
