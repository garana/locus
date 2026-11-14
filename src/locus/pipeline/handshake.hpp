#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace locus::pipeline {

/**
 * The shared-key handshake of the stage protocol (i#847): before any
 * frame, both ends of a connection prove they hold the same key, by a
 * mutual HMAC-SHA-256 challenge-response.
 *
 *   accepting side -> "LOCUSHS1" Ns                       (24 bytes)
 *   connecting side -> "LOCUSHS1" Nc HMAC(key, "client" Ns Nc)  (56)
 *   accepting side -> HMAC(key, "server" Nc Ns)           (32 bytes)
 *
 * Ns and Nc are 16 random bytes each. It authenticates the peer when
 * the connection opens, against an attacker who is not on the network
 * path: the frames after it are neither signed nor encrypted, so one on
 * the path (who can also relay a key holder's answer) is not stopped.
 */

/** Bytes of the accepting side's greeting. */
inline constexpr std::size_t kHelloBytes = 24;
/** Bytes of the connecting side's answer. */
inline constexpr std::size_t kAnswerBytes = 56;
/** Bytes of the accepting side's proof. */
inline constexpr std::size_t kProofBytes = 32;
/** The shortest key read_key_file takes. */
inline constexpr std::size_t kMinKeyBytes = 16;

/**
 * Reads a key file: its whole content is the key.
 *
 * @param path The file (`--key-file`); make one with
 *     `umask 077; head -c 32 /dev/urandom > stage.key`.
 * @returns The key.
 * @throws std::runtime_error when the file cannot be read, is not a
 *     regular file, group or others can read or write it, or it holds
 *     fewer than kMinKeyBytes or more than 4096 bytes.
 */
std::string read_key_file(const std::string& path);

/** The accepting side's steps, for a caller that owns the I/O (an
 *  event loop). */
class HandshakeServer {
  public:
    /** @param key The shared key (not copied: it must outlive this). */
    explicit HandshakeServer(std::string_view key);

    /** @returns The greeting to send first (kHelloBytes). */
    const std::string& hello() const { return hello_; }

    /**
     * @param answer The peer's kAnswerBytes.
     * @returns The proof to send back (kProofBytes), or nullopt when
     *     the answer is not from a holder of the key: close then.
     */
    std::optional<std::string> on_answer(std::string_view answer) const;

  private:
    std::string_view key_;
    std::string nonce_;
    std::string hello_;
};

/** The connecting side's steps. */
class HandshakeClient {
  public:
    /** @param key The shared key (not copied: it must outlive this). */
    explicit HandshakeClient(std::string_view key);

    /**
     * @param hello The peer's kHelloBytes.
     * @returns The answer to send (kAnswerBytes), or nullopt for a
     *     greeting that is not the protocol's.
     */
    std::optional<std::string> on_hello(std::string_view hello);

    /**
     * @param proof The peer's kProofBytes.
     * @returns Whether the peer holds the key.
     */
    bool on_proof(std::string_view proof) const;

  private:
    std::string_view key_;
    std::string nonce_;
    std::string peer_nonce_;
};

/**
 * Runs the accepting side on a connected socket, blocking.
 *
 * @param timeout_ms The longest the whole handshake may take (> 0).
 * @returns Whether the peer proved it holds `key`.
 */
bool handshake_accept(int fd, std::string_view key, int timeout_ms);

/** As handshake_accept, for the connecting side. */
bool handshake_connect(int fd, std::string_view key, int timeout_ms);

}  // namespace locus::pipeline
