#include "locus/pipeline/message.hpp"

#include <unistd.h>

#include <sys/socket.h>

#include <cerrno>
#include <cstring>
#include <utility>

namespace locus::pipeline {

namespace {

// Header bytes after the frame_len field: type(2) + flags(2) +
// request_id(8) + position(4) + token(4) + n_floats(4).
constexpr std::uint32_t kHeaderBytes = 2 + 2 + 8 + 4 + 4 + 4;
// Cap a single frame so a desynced/hostile length prefix cannot make us
// allocate unbounded memory. 256 MiB covers any realistic n_vocab.
constexpr std::uint32_t kMaxFrameBytes = 256u * 1024 * 1024;

bool valid_type(std::uint16_t t) {
    return t == static_cast<std::uint16_t>(MsgType::kToken) ||
           t == static_cast<std::uint16_t>(MsgType::kActivation) ||
           t == static_cast<std::uint16_t>(MsgType::kLogits);
}

// Little-endian integer append helpers (portable regardless of host
// endianness; the wire is defined little-endian).
void put_u16(std::string& out, std::uint16_t v) {
    out.push_back(static_cast<char>(v & 0xff));
    out.push_back(static_cast<char>((v >> 8) & 0xff));
}
void put_u32(std::string& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
    }
}
void put_u64(std::string& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
    }
}
std::uint16_t get_u16(const char* p) {
    return static_cast<std::uint16_t>(
        (static_cast<std::uint8_t>(p[0])) |
        (static_cast<std::uint8_t>(p[1]) << 8));
}
std::uint32_t get_u32(const char* p) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        v |= static_cast<std::uint32_t>(
                 static_cast<std::uint8_t>(p[i]))
             << (8 * i);
    }
    return v;
}
std::uint64_t get_u64(const char* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(
                 static_cast<std::uint8_t>(p[i]))
             << (8 * i);
    }
    return v;
}

// Parses the frame payload (everything after frame_len) into `out`.
// Returns false on any inconsistency (bad type, size mismatch).
bool parse_payload(const char* p, std::uint32_t len, Message& out) {
    if (len < kHeaderBytes) {
        return false;
    }
    const std::uint16_t type = get_u16(p);
    if (!valid_type(type)) {
        return false;
    }
    // p[2..3] flags, currently ignored.
    const std::uint64_t req = get_u64(p + 4);
    const std::uint32_t pos = get_u32(p + 12);
    const std::int32_t token =
        static_cast<std::int32_t>(get_u32(p + 16));
    const std::uint32_t nf = get_u32(p + 20);
    if (static_cast<std::uint64_t>(nf) * 4u + kHeaderBytes != len) {
        return false;
    }
    out.type = static_cast<MsgType>(type);
    out.request_id = req;
    out.position = pos;
    out.token = token;
    out.data.resize(nf);
    if (nf > 0) {
        std::memcpy(out.data.data(), p + kHeaderBytes,
                    static_cast<std::size_t>(nf) * 4u);
    }
    return true;
}

// Reads up to n bytes into p, retrying EINTR. Returns the count read
// (n on success, less on EOF, -1 on error).
ssize_t read_some(int fd, char* p, std::size_t n) {
    std::size_t got = 0;
    while (got < n) {
        const ssize_t r = ::read(fd, p + got, n - got);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (r == 0) {
            break;  // EOF
        }
        got += static_cast<std::size_t>(r);
    }
    return static_cast<ssize_t>(got);
}

}  // namespace

Message make_token(std::uint64_t request_id, std::uint32_t position,
                   tok::TokenId token) {
    Message m;
    m.type = MsgType::kToken;
    m.request_id = request_id;
    m.position = position;
    m.token = token;
    return m;
}

Message make_activation(std::uint64_t request_id,
                        std::uint32_t position,
                        std::vector<float> hidden) {
    Message m;
    m.type = MsgType::kActivation;
    m.request_id = request_id;
    m.position = position;
    m.data = std::move(hidden);
    return m;
}

Message make_logits(std::uint64_t request_id, std::uint32_t position,
                    std::vector<float> logits) {
    Message m;
    m.type = MsgType::kLogits;
    m.request_id = request_id;
    m.position = position;
    m.data = std::move(logits);
    return m;
}

void encode(const Message& m, std::string& out) {
    const std::uint32_t nf = static_cast<std::uint32_t>(m.data.size());
    const std::uint32_t frame_len = kHeaderBytes + nf * 4u;
    put_u32(out, frame_len);
    put_u16(out, static_cast<std::uint16_t>(m.type));
    put_u16(out, 0);  // flags
    put_u64(out, m.request_id);
    put_u32(out, m.position);
    put_u32(out, static_cast<std::uint32_t>(m.token));
    put_u32(out, nf);
    if (nf > 0) {
        out.append(reinterpret_cast<const char*>(m.data.data()),
                   static_cast<std::size_t>(nf) * 4u);
    }
}

Decode decode(std::string& buf, Message& out, std::string& err) {
    if (buf.size() < 4) {
        return Decode::kIncomplete;
    }
    const std::uint32_t frame_len = get_u32(buf.data());
    if (frame_len < kHeaderBytes || frame_len > kMaxFrameBytes) {
        err = "pipeline: bad frame length";
        return Decode::kError;
    }
    if (buf.size() < 4u + frame_len) {
        return Decode::kIncomplete;
    }
    if (!parse_payload(buf.data() + 4, frame_len, out)) {
        err = "pipeline: malformed frame";
        return Decode::kError;
    }
    buf.erase(0, 4u + frame_len);
    return Decode::kComplete;
}

std::uint32_t max_frame_bytes() { return kMaxFrameBytes; }

bool write_all(int fd, std::span<const char> bytes) {
    std::size_t off = 0;
    while (off < bytes.size()) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags = MSG_NOSIGNAL;
#endif
        const ssize_t w =
            ::send(fd, bytes.data() + off, bytes.size() - off, flags);
        if (w < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (w == 0) {
            return false;
        }
        off += static_cast<std::size_t>(w);
    }
    return true;
}

bool write_message(int fd, const Message& m) {
    std::string frame;
    encode(m, frame);
    return write_all(fd, std::span<const char>(frame.data(),
                                               frame.size()));
}

ReadResult read_message(int fd, Message& out) {
    char lenb[4];
    const ssize_t g = read_some(fd, lenb, 4);
    if (g == 0) {
        return ReadResult::kEof;  // clean close at a frame boundary
    }
    if (g < 0) {
        // A recv timeout (SO_RCVTIMEO) surfaces as EAGAIN/EWOULDBLOCK;
        // report it distinctly (kTimeout) so a caller can tell a stalled
        // peer from a broken one. Both are terminal for the connection:
        // a partial frame may have been consumed, so the caller must
        // reconnect, not re-read.
        return (errno == EAGAIN || errno == EWOULDBLOCK)
                   ? ReadResult::kTimeout
                   : ReadResult::kError;
    }
    if (g != 4) {
        return ReadResult::kError;  // truncated length prefix
    }
    const std::uint32_t frame_len = get_u32(lenb);
    if (frame_len < kHeaderBytes || frame_len > kMaxFrameBytes) {
        return ReadResult::kError;
    }
    std::string payload(frame_len, '\0');
    const ssize_t p = read_some(fd, payload.data(), frame_len);
    if (p != static_cast<ssize_t>(frame_len)) {
        if (p < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return ReadResult::kTimeout;
        }
        return ReadResult::kError;  // close mid-frame is an error
    }
    if (!parse_payload(payload.data(), frame_len, out)) {
        return ReadResult::kError;
    }
    return ReadResult::kOk;
}

}  // namespace locus::pipeline
