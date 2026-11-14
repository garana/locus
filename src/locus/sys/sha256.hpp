#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace locus::sys {

/** A SHA-256 digest. */
using Sha256Digest = std::array<std::uint8_t, 32>;

/**
 * SHA-256 (FIPS 180-4).
 *
 * @param data The bytes to hash.
 * @returns The 32-byte digest.
 */
Sha256Digest sha256(std::string_view data);

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
