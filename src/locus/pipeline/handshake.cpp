#include "locus/pipeline/handshake.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <random>
#include <stdexcept>

#include "locus/sys/sha256.hpp"

namespace locus::pipeline {

namespace {

constexpr std::string_view kMagic = "LOCUSHS1";
constexpr std::size_t kNonceBytes = 16;
constexpr std::size_t kMaxKeyBytes = 4096;

#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;
#endif

std::string random_nonce() {
    std::random_device rd;  // the OS's CSPRNG on Linux and macOS
    std::string n(kNonceBytes, '\0');
    for (std::size_t i = 0; i < kNonceBytes; i += 4) {
        const unsigned v = rd();
        for (std::size_t j = 0; j < 4 && i + j < kNonceBytes; ++j) {
            n[i + j] = static_cast<char>(v >> (8 * j));
        }
    }
    return n;
}

std::string tag(std::string_view key, std::string_view who,
                std::string_view a, std::string_view b) {
    std::string msg(who);
    msg.append(a);
    msg.append(b);
    const sys::Sha256Digest d = sys::hmac_sha256(key, msg);
    return std::string(d.begin(), d.end());
}

bool same_tag(std::string_view a, const std::string& b) {
    if (a.size() != b.size() || b.size() != 32) return false;
    sys::Sha256Digest x{}, y{};
    for (std::size_t i = 0; i < 32; ++i) {
        x[i] = static_cast<std::uint8_t>(a[i]);
        y[i] = static_cast<std::uint8_t>(b[i]);
    }
    return sys::digest_equal(x, y);
}

using Clock = std::chrono::steady_clock;

int left_ms(Clock::time_point deadline) {
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        deadline - Clock::now())
                        .count();
    return ms <= 0 ? 0 : static_cast<int>(ms);
}

/** Reads exactly n bytes before the deadline. */
bool read_exact(int fd, std::size_t n, std::string& out,
                Clock::time_point deadline) {
    out.clear();
    char buf[64];
    while (out.size() < n) {
        pollfd p{fd, POLLIN, 0};
        const int r = ::poll(&p, 1, left_ms(deadline));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        const ssize_t got = ::recv(fd, buf, std::min(sizeof buf, n - out.size()), 0);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        out.append(buf, static_cast<std::size_t>(got));
    }
    return true;
}

/** Writes all of s before the deadline. */
bool write_all(int fd, std::string_view s, Clock::time_point deadline) {
    std::size_t off = 0;
    while (off < s.size()) {
        pollfd p{fd, POLLOUT, 0};
        const int r = ::poll(&p, 1, left_ms(deadline));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        const ssize_t put =
            ::send(fd, s.data() + off, s.size() - off, kSendFlags);
        if (put < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        if (put <= 0) return false;
        off += static_cast<std::size_t>(put);
    }
    return true;
}

}  // namespace

std::string read_key_file(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        throw std::runtime_error("cannot open key file " + path);
    }
    struct stat st {};
    std::string key;
    std::string why;
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        why = "not a regular file";
    } else if ((st.st_mode & 077) != 0) {
        why = "group or others can read or write it (chmod 600 it)";
    } else if (st.st_size < static_cast<off_t>(kMinKeyBytes) ||
               st.st_size > static_cast<off_t>(kMaxKeyBytes)) {
        why = "it must hold 16 to 4096 bytes";
    } else {
        key.resize(static_cast<std::size_t>(st.st_size));
        std::size_t off = 0;
        while (off < key.size()) {
            const ssize_t n = ::read(fd, key.data() + off, key.size() - off);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            off += static_cast<std::size_t>(n);
        }
        if (off != key.size()) why = "short read";
    }
    ::close(fd);
    if (!why.empty()) {
        throw std::runtime_error("key file " + path + ": " + why);
    }
    return key;
}

HandshakeServer::HandshakeServer(std::string_view key)
    : key_(key), nonce_(random_nonce()) {
    hello_.assign(kMagic);
    hello_.append(nonce_);
}

std::optional<std::string> HandshakeServer::on_answer(
    std::string_view answer) const {
    if (answer.size() != kAnswerBytes || answer.substr(0, 8) != kMagic) {
        return std::nullopt;
    }
    const std::string_view peer = answer.substr(8, kNonceBytes);
    const std::string_view mac = answer.substr(8 + kNonceBytes);
    if (!same_tag(mac, tag(key_, "client", nonce_, peer))) {
        return std::nullopt;
    }
    return tag(key_, "server", peer, nonce_);
}

HandshakeClient::HandshakeClient(std::string_view key)
    : key_(key), nonce_(random_nonce()) {}

std::optional<std::string> HandshakeClient::on_hello(std::string_view hello) {
    if (hello.size() != kHelloBytes || hello.substr(0, 8) != kMagic) {
        return std::nullopt;
    }
    peer_nonce_.assign(hello.substr(8));
    std::string answer(kMagic);
    answer.append(nonce_);
    answer.append(tag(key_, "client", peer_nonce_, nonce_));
    return answer;
}

bool HandshakeClient::on_proof(std::string_view proof) const {
    return !peer_nonce_.empty() &&
           same_tag(proof, tag(key_, "server", nonce_, peer_nonce_));
}

bool handshake_accept(int fd, std::string_view key, int timeout_ms) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    const HandshakeServer hs(key);
    std::string answer;
    if (!write_all(fd, hs.hello(), deadline) ||
        !read_exact(fd, kAnswerBytes, answer, deadline)) {
        return false;
    }
    const auto proof = hs.on_answer(answer);
    return proof && write_all(fd, *proof, deadline);
}

bool handshake_connect(int fd, std::string_view key, int timeout_ms) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    HandshakeClient hc(key);
    std::string hello;
    std::string proof;
    if (!read_exact(fd, kHelloBytes, hello, deadline)) return false;
    const auto answer = hc.on_hello(hello);
    return answer && write_all(fd, *answer, deadline) &&
           read_exact(fd, kProofBytes, proof, deadline) && hc.on_proof(proof);
}

}  // namespace locus::pipeline
