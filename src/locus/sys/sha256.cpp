#include "locus/sys/sha256.hpp"

#include <cstdio>

namespace locus::sys {

namespace {

constexpr std::array<std::uint32_t, 64> kK = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

std::uint32_t rotr(std::uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

/** One 64-byte block into the state h. */
void compress(std::array<std::uint32_t, 8>& h, const unsigned char* p) {
    std::array<std::uint32_t, 64> w{};
    for (int i = 0; i < 16; ++i) {
        w[i] = (std::uint32_t{p[4 * i]} << 24) |
               (std::uint32_t{p[4 * i + 1]} << 16) |
               (std::uint32_t{p[4 * i + 2]} << 8) | std::uint32_t{p[4 * i + 3]};
    }
    for (int i = 16; i < 64; ++i) {
        const std::uint32_t s0 =
            rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 =
            rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto v = h;
    for (int i = 0; i < 64; ++i) {
        const std::uint32_t s1 = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
        const std::uint32_t ch = (v[4] & v[5]) ^ (~v[4] & v[6]);
        const std::uint32_t t1 = v[7] + s1 + ch + kK[i] + w[i];
        const std::uint32_t s0 = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
        const std::uint32_t maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
        v = {t1 + s0 + maj, v[0], v[1], v[2], v[3] + t1, v[4], v[5], v[6]};
    }
    for (int i = 0; i < 8; ++i) h[i] += v[i];
}

}  // namespace

Sha256Digest sha256(std::string_view data) {
    std::array<std::uint32_t, 8> h = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                      0xa54ff53a, 0x510e527f, 0x9b05688c,
                                      0x1f83d9ab, 0x5be0cd19};
    const auto* p = reinterpret_cast<const unsigned char*>(data.data());
    std::size_t n = data.size();
    std::size_t off = 0;
    for (; n - off >= 64; off += 64) compress(h, p + off);
    // The tail, a 1 bit, zeros to 56 mod 64 bytes, the bit length.
    unsigned char tail[128] = {};
    const std::size_t rest = n - off;
    for (std::size_t i = 0; i < rest; ++i) tail[i] = p[off + i];
    tail[rest] = 0x80;
    const std::size_t len = rest + 1 + 8 <= 64 ? 64 : 128;
    const std::uint64_t bits = static_cast<std::uint64_t>(n) * 8;
    for (int i = 0; i < 8; ++i) {
        tail[len - 1 - i] = static_cast<unsigned char>(bits >> (8 * i));
    }
    compress(h, tail);
    if (len == 128) compress(h, tail + 64);
    Sha256Digest out{};
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[4 * i + j] = static_cast<std::uint8_t>(h[i] >> (24 - 8 * j));
        }
    }
    return out;
}

std::string to_hex(const Sha256Digest& d) {
    std::string out;
    char buf[3];
    for (const std::uint8_t b : d) {
        std::snprintf(buf, sizeof buf, "%02x", b);
        out += buf;
    }
    return out;
}

Sha256Digest hmac_sha256(std::string_view key, std::string_view msg) {
    std::string k(key);
    if (k.size() > 64) {
        const Sha256Digest kd = sha256(k);
        k.assign(kd.begin(), kd.end());
    }
    k.resize(64, '\0');
    std::string inner(64, '\0');
    std::string outer(64, '\0');
    for (int i = 0; i < 64; ++i) {
        inner[i] = static_cast<char>(k[i] ^ 0x36);
        outer[i] = static_cast<char>(k[i] ^ 0x5c);
    }
    inner.append(msg);
    const Sha256Digest ih = sha256(inner);
    outer.append(ih.begin(), ih.end());
    return sha256(outer);
}

bool digest_equal(const Sha256Digest& a, const Sha256Digest& b) {
    std::uint8_t diff = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff = static_cast<std::uint8_t>(diff | (a[i] ^ b[i]));
    }
    return diff == 0;
}

}  // namespace locus::sys
