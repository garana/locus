#include "locus/image/inflate.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace locus::image {

namespace {

[[noreturn]] void bad(const char* what) {
    throw std::invalid_argument(std::string("inflate: ") + what);
}

/** LSB-first bit reader over a bounded buffer. */
class Bits {
  public:
    explicit Bits(std::span<const std::uint8_t> in) : in_(in) {}

    /** @returns The next n bits (n <= 16), least significant first. */
    std::uint32_t get(int n) {
        while (cnt_ < n) {
            if (pos_ >= in_.size()) bad("truncated stream");
            buf_ |= std::uint32_t{in_[pos_++]} << cnt_;
            cnt_ += 8;
        }
        const std::uint32_t v = buf_ & ((1u << n) - 1u);
        buf_ >>= n;
        cnt_ -= n;
        return v;
    }
    /** Drops the bits left in the current byte. */
    void align() {
        buf_ = 0;
        cnt_ = 0;
    }
    std::size_t pos() const { return pos_; }
    std::uint8_t byte() {
        if (pos_ >= in_.size()) bad("truncated stream");
        return in_[pos_++];
    }

  private:
    std::span<const std::uint8_t> in_;
    std::size_t pos_ = 0;
    std::uint32_t buf_ = 0;
    int cnt_ = 0;
};

constexpr int kMaxBits = 15;

/** A canonical Huffman code: symbols per length, symbols in order. */
struct Huffman {
    std::array<std::uint16_t, kMaxBits + 1> count{};
    std::array<std::uint16_t, 288> symbol{};
};

/**
 * Builds h from code lengths (0 = unused). An over-subscribed set is
 * an error; an incomplete one is allowed (a code it lacks fails at
 * decode), as zlib allows a lone distance code.
 */
void build(Huffman& h, std::span<const std::uint8_t> lengths) {
    h.count.fill(0);
    for (const std::uint8_t l : lengths) h.count[l]++;
    if (h.count[0] == lengths.size()) return;  // no codes at all
    int left = 1;
    for (int len = 1; len <= kMaxBits; ++len) {
        left <<= 1;
        left -= h.count[len];
        if (left < 0) bad("over-subscribed Huffman code");
    }
    std::array<std::uint16_t, kMaxBits + 1> offs{};
    for (int len = 1; len < kMaxBits; ++len) {
        offs[len + 1] = static_cast<std::uint16_t>(offs[len] + h.count[len]);
    }
    for (std::size_t s = 0; s < lengths.size(); ++s) {
        if (lengths[s] != 0) {
            h.symbol[offs[lengths[s]]++] = static_cast<std::uint16_t>(s);
        }
    }
}

/** Decodes one symbol (puff's canonical walk, one bit at a time). */
int decode(Bits& b, const Huffman& h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= kMaxBits; ++len) {
        code |= static_cast<int>(b.get(1));
        const int count = h.count[len];
        if (code - count < first) return h.symbol[index + (code - first)];
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    bad("invalid Huffman code");
}

constexpr std::uint16_t kLenBase[29] = {3,  4,  5,  6,   7,   8,   9,   10,
                                        11, 13, 15, 17,  19,  23,  27,  31,
                                        35, 43, 51, 59,  67,  83,  99,  115,
                                        131, 163, 195, 227, 258};
constexpr std::uint8_t kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                        1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                        4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr std::uint16_t kDistBase[30] = {
    1,   2,   3,   4,   5,   7,    9,    13,   17,   25,
    33,  49,  65,  97,  129, 193,  257,  385,  513,  769,
    1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr std::uint8_t kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2,  2,  3,  3,
                                         4, 4, 5, 5, 6, 6, 7,  7,  8,  8,
                                         9, 9, 10, 10, 11, 11, 12, 12, 13,
                                         13};

void put(std::vector<std::uint8_t>& out, std::uint8_t v, std::size_t max) {
    if (out.size() >= max) bad("output over the limit");
    out.push_back(v);
}

/** Inflates one block's symbols with the given codes. */
void codes(Bits& b, const Huffman& lit, const Huffman& dist,
           std::vector<std::uint8_t>& out, std::size_t max) {
    for (;;) {
        const int sym = decode(b, lit);
        if (sym < 256) {
            put(out, static_cast<std::uint8_t>(sym), max);
        } else if (sym == 256) {
            return;
        } else {
            const int li = sym - 257;
            if (li >= 29) bad("invalid length symbol");
            const std::size_t len =
                kLenBase[li] + b.get(kLenExtra[li]);
            const int di = decode(b, dist);
            if (di >= 30) bad("invalid distance symbol");
            const std::size_t d = kDistBase[di] + b.get(kDistExtra[di]);
            if (d > out.size()) bad("distance before the start");
            if (len > max - out.size()) bad("output over the limit");
            const std::size_t from = out.size() - d;
            for (std::size_t i = 0; i < len; ++i) {
                out.push_back(out[from + i]);  // may overlap: byte by byte
            }
        }
    }
}

void stored(Bits& b, std::vector<std::uint8_t>& out, std::size_t max) {
    b.align();
    const std::uint32_t len = b.byte() | (std::uint32_t{b.byte()} << 8);
    const std::uint32_t nlen = b.byte() | (std::uint32_t{b.byte()} << 8);
    if ((len ^ 0xFFFFu) != nlen) bad("stored block length check");
    if (len > max - out.size()) bad("output over the limit");
    for (std::uint32_t i = 0; i < len; ++i) out.push_back(b.byte());
}

void fixed(Bits& b, std::vector<std::uint8_t>& out, std::size_t max) {
    static const auto tables = [] {
        std::array<std::uint8_t, 288> l{};
        for (int i = 0; i < 144; ++i) l[i] = 8;
        for (int i = 144; i < 256; ++i) l[i] = 9;
        for (int i = 256; i < 280; ++i) l[i] = 7;
        for (int i = 280; i < 288; ++i) l[i] = 8;
        std::array<std::uint8_t, 30> d{};
        d.fill(5);
        std::pair<Huffman, Huffman> t;
        build(t.first, l);
        build(t.second, d);
        return t;
    }();
    codes(b, tables.first, tables.second, out, max);
}

void dynamic(Bits& b, std::vector<std::uint8_t>& out, std::size_t max) {
    const int nlen = static_cast<int>(b.get(5)) + 257;
    const int ndist = static_cast<int>(b.get(5)) + 1;
    const int ncode = static_cast<int>(b.get(4)) + 4;
    if (nlen > 286 || ndist > 30) bad("bad code counts");
    static constexpr std::uint8_t kOrder[19] = {16, 17, 18, 0, 8,  7, 9,
                                                6,  10, 5,  11, 4, 12, 3,
                                                13, 2,  14, 1,  15};
    std::array<std::uint8_t, 19> cl{};
    for (int i = 0; i < ncode; ++i) {
        cl[kOrder[i]] = static_cast<std::uint8_t>(b.get(3));
    }
    Huffman lencode;
    build(lencode, cl);
    std::array<std::uint8_t, 286 + 30> lengths{};
    int n = 0;
    while (n < nlen + ndist) {
        int sym = decode(b, lencode);
        if (sym < 16) {
            lengths[n++] = static_cast<std::uint8_t>(sym);
            continue;
        }
        std::uint8_t len = 0;
        int rep = 0;
        if (sym == 16) {
            if (n == 0) bad("repeat with no previous length");
            len = lengths[n - 1];
            rep = 3 + static_cast<int>(b.get(2));
        } else if (sym == 17) {
            rep = 3 + static_cast<int>(b.get(3));
        } else {
            rep = 11 + static_cast<int>(b.get(7));
        }
        if (n + rep > nlen + ndist) bad("too many code lengths");
        while (rep-- > 0) lengths[n++] = len;
    }
    if (lengths[256] == 0) bad("no end-of-block code");
    Huffman lit, dist;
    build(lit, std::span<const std::uint8_t>(lengths.data(), nlen));
    build(dist, std::span<const std::uint8_t>(lengths.data() + nlen, ndist));
    codes(b, lit, dist, out, max);
}

}  // namespace

std::uint32_t adler32(std::span<const std::uint8_t> d) {
    std::uint32_t a = 1, s = 0;
    std::size_t i = 0;
    while (i < d.size()) {
        // 5552 keeps the sums below 2^32 before the modulo (zlib's NMAX).
        const std::size_t end = std::min(d.size(), i + 5552);
        for (; i < end; ++i) {
            a += d[i];
            s += a;
        }
        a %= 65521u;
        s %= 65521u;
    }
    return (s << 16) | a;
}

std::vector<std::uint8_t> zlib_inflate(std::span<const std::uint8_t> in,
                                       std::size_t max_out) {
    if (in.size() < 2) bad("truncated stream");
    const std::uint32_t cmf = in[0], flg = in[1];
    if ((cmf & 0x0Fu) != 8 || (cmf >> 4) > 7) bad("not deflate");
    if (((cmf << 8) | flg) % 31u != 0) bad("bad zlib header check");
    if (flg & 0x20u) bad("preset dictionary");
    Bits b(in.subspan(2));
    std::vector<std::uint8_t> out;
    bool last = false;
    while (!last) {
        last = b.get(1) != 0;
        switch (b.get(2)) {
            case 0: stored(b, out, max_out); break;
            case 1: fixed(b, out, max_out); break;
            case 2: dynamic(b, out, max_out); break;
            default: bad("invalid block type");
        }
    }
    b.align();
    std::uint32_t want = 0;
    for (int i = 0; i < 4; ++i) want = (want << 8) | b.byte();
    if (want != adler32(out)) bad("Adler-32 mismatch");
    return out;
}

}  // namespace locus::image
