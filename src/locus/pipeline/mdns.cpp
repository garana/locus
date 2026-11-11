#include "locus/pipeline/mdns.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <optional>
#include <stdexcept>

namespace locus::pipeline {

namespace {

constexpr std::uint16_t kTypeA = 1, kTypePtr = 12, kTypeSrv = 33,
                        kTypeAaaa = 28;
constexpr std::size_t kMaxMessage = 9000, kMaxRecords = 256,
                      kMaxJumps = 16, kMaxName = 255;

std::string lower(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

/** Reads DNS messages with bounds on every access. */
class Reader {
  public:
    explicit Reader(std::span<const std::uint8_t> m) : m_(m) {}

    bool ok() const { return ok_; }
    std::size_t pos() const { return pos_; }

    std::uint16_t u16() {
        if (!need(2)) return 0;
        const std::uint16_t v =
            static_cast<std::uint16_t>((m_[pos_] << 8) | m_[pos_ + 1]);
        pos_ += 2;
        return v;
    }
    std::uint32_t u32() {
        const std::uint32_t hi = u16();
        return (hi << 16) | u16();
    }
    void skip(std::size_t n) {
        if (need(n)) pos_ += n;
    }
    std::span<const std::uint8_t> bytes(std::size_t n) {
        if (!need(n)) return {};
        const auto s = m_.subspan(pos_, n);
        pos_ += n;
        return s;
    }

    /** A (possibly compressed) name at the current position, dotted,
     *  lowercase, without the trailing dot. */
    std::string name() {
        std::string out;
        std::size_t p = pos_, jumps = 0;
        bool jumped = false;
        for (;;) {
            if (p >= m_.size()) return fail();
            const std::uint8_t len = m_[p];
            if ((len & 0xC0) == 0xC0) {
                if (p + 1 >= m_.size()) return fail();
                const std::size_t to = (static_cast<std::size_t>(len & 0x3F)
                                        << 8) |
                                       m_[p + 1];
                // A pointer must point back, and not too often.
                if (to >= p || ++jumps > kMaxJumps) return fail();
                if (!jumped) pos_ = p + 2;
                jumped = true;
                p = to;
                continue;
            }
            if ((len & 0xC0) != 0) return fail();  // reserved label types
            if (len == 0) {
                if (!jumped) pos_ = p + 1;
                return lower(out);
            }
            if (p + 1 + len > m_.size()) return fail();
            if (!out.empty()) out += '.';
            out.append(reinterpret_cast<const char*>(&m_[p + 1]), len);
            if (out.size() > kMaxName) return fail();
            p += 1 + len;
        }
    }

  private:
    bool need(std::size_t n) {
        if (!ok_ || pos_ + n > m_.size()) {
            ok_ = false;
            return false;
        }
        return true;
    }
    std::string fail() {
        ok_ = false;
        return {};
    }

    std::span<const std::uint8_t> m_;
    std::size_t pos_ = 0;
    bool ok_ = true;
};

void put16(std::vector<std::uint8_t>& out, std::uint16_t v) {
    out.push_back(static_cast<std::uint8_t>(v >> 8));
    out.push_back(static_cast<std::uint8_t>(v & 0xFF));
}

}  // namespace

std::vector<std::uint8_t> mdns_query(const std::string& service) {
    const std::string name = service + ".local";
    if (name.size() > kMaxName) {
        throw std::invalid_argument("mdns: name over 255 bytes");
    }
    std::vector<std::uint8_t> out;
    put16(out, 0);  // id: 0 for mDNS
    put16(out, 0);  // flags: a standard query
    put16(out, 1);  // one question
    put16(out, 0);
    put16(out, 0);
    put16(out, 0);
    std::size_t start = 0;
    while (start <= name.size()) {
        const std::size_t dot = std::min(name.find('.', start), name.size());
        const std::size_t len = dot - start;
        if (len == 0 || len > 63) {
            throw std::invalid_argument("mdns: bad label in " + name);
        }
        out.push_back(static_cast<std::uint8_t>(len));
        out.insert(out.end(), name.begin() + static_cast<long>(start),
                   name.begin() + static_cast<long>(dot));
        start = dot + 1;
    }
    out.push_back(0);
    put16(out, kTypePtr);
    put16(out, 1);  // class IN, multicast response wanted (QM)
    return out;
}

std::vector<SrvTarget> mdns_parse(std::span<const std::uint8_t> msg,
                                  const std::string& service) {
    if (msg.size() > kMaxMessage || msg.size() < 12) return {};
    Reader r(msg);
    r.u16();  // id
    const std::uint16_t flags = r.u16();
    if ((flags & 0x8000) == 0) return {};  // a query, not a response
    const std::size_t qd = r.u16();
    const std::size_t records =
        std::size_t{r.u16()} + std::size_t{r.u16()} + std::size_t{r.u16()};
    if (qd > kMaxRecords || records > kMaxRecords) return {};
    for (std::size_t i = 0; i < qd && r.ok(); ++i) {
        r.name();
        r.skip(4);
    }
    const std::string want = lower(service) + ".local";
    std::vector<std::string> instances;
    struct Srv {
        std::string target;
        int port, priority, weight;
    };
    std::map<std::string, Srv> srv;
    std::map<std::string, std::string> v4, v6;
    for (std::size_t i = 0; i < records && r.ok(); ++i) {
        const std::string owner = r.name();
        const std::uint16_t type = r.u16();
        r.u16();  // class (and the cache-flush bit)
        const std::uint32_t ttl = r.u32();
        const std::uint16_t rdlen = r.u16();
        if (!r.ok()) break;
        const std::size_t end = r.pos() + rdlen;
        if (end > msg.size()) break;
        if (ttl != 0) {
            if (type == kTypePtr && owner == want) {
                const std::string inst = r.name();
                if (r.ok() &&
                    std::find(instances.begin(), instances.end(), inst) ==
                        instances.end()) {
                    instances.push_back(inst);
                }
            } else if (type == kTypeSrv && rdlen >= 7) {
                Srv s;
                s.priority = r.u16();
                s.weight = r.u16();
                s.port = r.u16();
                s.target = r.name();
                if (r.ok()) srv[owner] = s;
            } else if (type == kTypeA && rdlen == 4) {
                char buf[INET_ADDRSTRLEN];
                const auto b = r.bytes(4);
                if (inet_ntop(AF_INET, b.data(), buf, sizeof(buf)) != nullptr) {
                    v4.emplace(owner, buf);
                }
            } else if (type == kTypeAaaa && rdlen == 16) {
                char buf[INET6_ADDRSTRLEN];
                const auto b = r.bytes(16);
                if (inet_ntop(AF_INET6, b.data(), buf, sizeof(buf)) !=
                    nullptr) {
                    v6.emplace(owner, buf);
                }
            }
        }
        if (!r.ok() || r.pos() > end) break;
        // Continue at the record's end whatever its rdata held.
        r = Reader(msg);
        r.skip(end);
    }
    std::vector<SrvTarget> out;
    for (const auto& inst : instances) {
        const auto it = srv.find(inst);
        if (it == srv.end() || it->second.port <= 0) continue;
        const auto& s = it->second;
        std::string host = s.target;
        if (const auto a = v4.find(s.target); a != v4.end()) {
            host = a->second;
        } else if (const auto a6 = v6.find(s.target); a6 != v6.end()) {
            host = a6->second;
        }
        out.push_back({host, s.port, s.priority, s.weight});
    }
    return out;
}

std::vector<SrvTarget> mdns_browse(const std::string& service,
                                   const MdnsBrowse& opt) {
    const auto query = mdns_query(service);
    in_addr group{};
    if (inet_pton(AF_INET, opt.group.c_str(), &group) != 1) return {};
    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return {};
    // An mDNS query from a port other than 5353 is a one-shot query:
    // responders answer it by unicast to that port.
    const unsigned char ttl = 255, loop = 1;
    ::setsockopt(fd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl));
    ::setsockopt(fd, IPPROTO_IP, IP_MULTICAST_LOOP, &loop, sizeof(loop));
    sockaddr_in to{};
    to.sin_family = AF_INET;
    to.sin_port = htons(static_cast<std::uint16_t>(opt.port));
    to.sin_addr = group;
    std::vector<SrvTarget> out;
    if (::sendto(fd, query.data(), query.size(), 0,
                 reinterpret_cast<const sockaddr*>(&to), sizeof(to)) ==
        static_cast<ssize_t>(query.size())) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(opt.timeout_ms);
        std::vector<std::uint8_t> buf(kMaxMessage);
        for (;;) {
            const auto left =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - std::chrono::steady_clock::now())
                    .count();
            if (left <= 0) break;
            pollfd p{fd, POLLIN, 0};
            if (::poll(&p, 1, static_cast<int>(left)) <= 0) break;
            const ssize_t n = ::recv(fd, buf.data(), buf.size(), 0);
            if (n <= 0) continue;
            for (auto& t : mdns_parse(
                     std::span<const std::uint8_t>(buf.data(),
                                                   static_cast<std::size_t>(n)),
                     service)) {
                const bool seen =
                    std::any_of(out.begin(), out.end(), [&](const auto& o) {
                        return o.host == t.host && o.port == t.port;
                    });
                if (!seen) out.push_back(std::move(t));
            }
        }
    }
    ::close(fd);
    return out;
}

}  // namespace locus::pipeline
