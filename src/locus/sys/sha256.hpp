#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace locus::sys {

/** A SHA-256 digest. */
using Sha256Digest = std::array<std::uint8_t, 32>;

/** SHA-256 (FIPS 180-4) fed in pieces, for data too large to copy. */
class Sha256 {
  public:
    Sha256();
    /** Adds `n` bytes at `p`. */
    void update(const void* p, std::size_t n);
    void update(std::string_view s) { update(s.data(), s.size()); }
    /** @returns The digest of everything added (call once). */
    Sha256Digest finish();

  private:
    std::array<std::uint32_t, 8> h_;
    std::array<unsigned char, 64> buf_{};
    std::size_t used_ = 0;     ///< bytes in buf_
    std::uint64_t total_ = 0;  ///< bytes added
};

/**
 * SHA-256 (FIPS 180-4).
 *
 * @param data The bytes to hash.
 * @returns The 32-byte digest.
 */
Sha256Digest sha256(std::string_view data);

/** HMAC-SHA-256 (RFC 2104) fed in pieces. */
class HmacSha256 {
  public:
    /** @param key Any length; a key over 64 bytes is hashed first. */
    explicit HmacSha256(std::string_view key);
    void update(const void* p, std::size_t n) { inner_.update(p, n); }
    void update(std::string_view s) { inner_.update(s); }
    /** @returns The tag of everything added (call once). */
    Sha256Digest finish();

  private:
    std::string outer_pad_;
    Sha256 inner_;
};

/** @returns The digest as 64 lowercase hex digits. */
std::string to_hex(const Sha256Digest& d);

/**
 * HMAC-SHA-256 (RFC 2104).
 *
 * @param key The key (any length; a key over 64 bytes is hashed first).
 * @param msg The message.
 * @returns The 32-byte tag.
 */
Sha256Digest hmac_sha256(std::string_view key, std::string_view msg);

/**
 * Compares two digests in time that does not depend on where they
 * differ, so a tag check leaks nothing through its timing.
 *
 * @returns Whether a and b are equal.
 */
bool digest_equal(const Sha256Digest& a, const Sha256Digest& b);

}  // namespace locus::sys
