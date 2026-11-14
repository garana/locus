#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "locus/kv/kv_quant.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::engine {

/**
 * Everything besides the tokens that decides whether cached KV can be
 * reused (i#416): KV computed by another model, backend, KV type, pool
 * geometry or RoPE / context override differs, so any field that
 * differs makes a stored prefix a miss.
 */
struct DiskKey {
    /** Hash of the GGUF header and tensor table (the caller's). */
    std::uint64_t model_hash = 0;
    /** Model file size in bytes. */
    std::uint64_t model_size = 0;
    /** The backend that computed the KV ("cpu", "vulkan", ...);
     *  at most 32 bytes. */
    std::string backend;
    kv::KvType kv_type = kv::KvType::kF32;
    std::uint32_t block_tokens = 0;
    std::uint32_t kv_dim = 0;
    std::uint32_t n_layers = 0;
    std::uint32_t layer_base = 0;
    /** A digest of any RoPE or context override that changes K
     *  (0 when none). */
    std::uint64_t overrides = 0;
    /** Attention ran on the device over a KV mirror (CUDA, i#466): its
     *  sums differ from the host's, so later layers' K/V do too. Only
     *  the engine's own cache knows (the mirror attaches when it fits),
     *  so the engine sets it (DiskPrefixStore::set_device_attention). */
    bool device_attention = false;
    /** The routed experts ran on the GPU, not the host (Vulkan,
     *  i#654): the two sum in other orders. The route depends on the
     *  free device memory once the KV pool exists, so the engine sets
     *  it (DiskPrefixStore::set_experts_on_gpu). */
    bool experts_on_gpu = false;
};

/**
 * Creates `dir` (mode 0700) when it is missing, and checks that only
 * this user can change what is in it: its files hold clients' prompt
 * tokens, and a read trusts the KV in them.
 *
 * @param dir The cache directory.
 * @throws std::runtime_error when `dir` is not a directory, another
 *     user owns it, or its group or others can write to it.
 */
void ensure_private_dir(const std::filesystem::path& dir);

/**
 * Prefix-cache KV blocks on persistent storage (i#416), so a prompt
 * prefix seen before is read back instead of computed again, also
 * after a server restart.
 *
 * One file per cached prefix, named by a 64-bit digest of the key and
 * the prefix tokens. The file holds the full key, the tokens and one
 * checksum per block, and a read compares all of them: a digest
 * collision, a file from another key, a torn write or a flipped bit
 * reads as a miss (and the file is removed), never as wrong KV. Files
 * are written under a temporary name and renamed, and need no fsync,
 * since a torn file fails its checksum.
 *
 * The directory is capped in bytes; a write evicts the least recently
 * used files (by unlink) until it fits. Reads and writes refresh a
 * file's place; across restarts the order comes from file mtimes.
 * Files of other keys in the same directory share the cap.
 *
 * Thread-safe: the index is under one mutex, and the file I/O of
 * read() and write() runs outside it, so find() does not wait behind
 * a long read. One store per directory: a second store or process on
 * the same directory does not see the other's new files (the cap can
 * be exceeded, and a removed file is only a miss later).
 */
class DiskPrefixStore {
  public:
    static constexpr std::uint32_t kFormatVersion = 2;

    /**
     * Opens (creating it if needed) the cache directory `dir`, indexes
     * the files in it and removes temporary files a crash left behind.
     *
     * @param block_bytes Bytes of one KV block (all layers of
     *     block_tokens positions).
     * @param cap_bytes Disk budget for all files in `dir`.
     * @throws std::invalid_argument on a zero block_tokens or
     *     block_bytes, or a backend name over 32 bytes.
     * @throws std::filesystem::filesystem_error when `dir` cannot be
     *     created or listed.
     */
    DiskPrefixStore(std::filesystem::path dir, DiskKey key,
                    std::size_t block_bytes, std::uint64_t cap_bytes);

    /**
     * @returns the blocks of the longest prefix of `prompt` with a
     *     file on disk (0 if none). The file is only checked by read().
     */
    virtual std::uint32_t find(std::span<const tok::TokenId> prompt) const;

    /**
     * Reads the stored blocks of `prompt`'s first `n_blocks` blocks,
     * block i into dst[i] (block_bytes each). On a mismatch (key,
     * tokens, size or a checksum) the file is removed; on an I/O error
     * it is kept.
     *
     * @returns true when every block was read and verified; on false
     *     the contents of dst are unspecified.
     */
    virtual bool read(std::span<const tok::TokenId> prompt,
                      std::uint32_t n_blocks,
              std::span<std::uint8_t* const> dst);

    /**
     * Stores `prompt`'s first `n_blocks` blocks, block i read from
     * src[i] (block_bytes each), evicting least recently used files to
     * stay under the cap. A prefix already stored is only refreshed.
     *
     * @returns false when the file alone exceeds the cap or on an I/O
     *     error (nothing is left behind).
     */
    virtual bool write(std::span<const tok::TokenId> prompt,
                       std::uint32_t n_blocks,
               std::span<const std::uint8_t* const> src);

    /** @returns the file that holds `prompt`'s first `n_blocks`
     *  blocks (whether or not it exists); empty when the prompt is
     *  shorter than that or n_blocks is 0. */
    std::filesystem::path path_for(std::span<const tok::TokenId> prompt,
                                   std::uint32_t n_blocks) const;

    /** @returns bytes of the indexed files in the directory. */
    std::uint64_t used_bytes() const;

    const DiskKey& key() const { return key_; }

    /** Sets DiskKey::device_attention. The engine calls it once, at
     *  construction, before any read or write. */
    void set_device_attention(bool on);
    /** Sets DiskKey::experts_on_gpu, as set_device_attention does. */
    void set_experts_on_gpu(bool on);
    std::size_t block_bytes() const { return block_bytes_; }
    std::uint64_t cap_bytes() const { return cap_bytes_; }

    /** Virtual so a test can wrap the I/O (a slow or failing read). */
    virtual ~DiskPrefixStore() = default;

  protected:
    /**
     * For a store that keeps its bytes elsewhere (HelperPrefixStore,
     * i#655): the key, digests and file format of this class, no
     * directory. The derived class overrides find, read and write.
     */
    DiskPrefixStore(DiskKey key, std::size_t block_bytes,
                    std::uint64_t cap_bytes, std::nullptr_t no_dir);

    /** @returns the digest naming `prompt`'s first `n_blocks` blocks. */
    std::uint64_t digest_of(std::span<const tok::TokenId> prompt,
                            std::uint32_t n_blocks) const;
    /** @returns the digest of every block count 1..prompt blocks
     *  (index 0 unused). */
    std::vector<std::uint64_t> digests(
        std::span<const tok::TokenId> prompt) const;
    /** The serialized header for `n_blocks` blocks of `prompt`,
     *  without its trailing checksum. */
    std::vector<std::uint8_t> header(std::span<const tok::TokenId> prompt,
                                     std::uint32_t n_blocks) const;
    /** @returns the bytes of a file whose header has `header_bytes`. */
    std::uint64_t file_bytes(std::size_t header_bytes,
                             std::uint32_t n_blocks) const;
    /**
     * @returns the start of a file for `prompt`'s first `n_blocks`
     *     blocks (the header, the block checksums of src[i] and the
     *     header checksum); the blocks follow it in the file.
     */
    std::vector<std::uint8_t> meta(std::span<const tok::TokenId> prompt,
                                   std::uint32_t n_blocks,
                                   std::span<const std::uint8_t* const> src)
        const;
    /**
     * Checks `blob` (one file's bytes) against this key, `prompt`'s
     * first `n_blocks` blocks and every checksum, as read() checks a
     * file, and copies block i into dst[i].
     *
     * @returns true when it matches; on false dst is unspecified.
     */
    bool decode(std::span<const std::uint8_t> blob,
                std::span<const tok::TokenId> prompt, std::uint32_t n_blocks,
                std::span<std::uint8_t* const> dst) const;

  private:
    struct File {
        std::uint64_t bytes = 0;
        /** Larger is more recent. */
        std::uint64_t tick = 0;
        /** Set once when indexed, so a read that finds a mismatch
         *  removes only the file it read, not a newer one under the
         *  same name. */
        std::uint64_t gen = 0;
    };

    std::filesystem::path path_of(std::uint64_t digest) const;
    /** Drops a file from the index when its gen is `gen` (0: any).
     *  @returns its path, for the caller to unlink after releasing the
     *  lock, or an empty path when nothing was dropped. */
    std::filesystem::path forget(std::uint64_t digest, std::uint64_t gen);
    /** @returns true when `digest` is indexed and its file exists;
     *  forgets an indexed file that no longer exists. */
    bool stored(std::uint64_t digest);
    /** Marks a file most recently used in the index (the caller sets
     *  its mtime after releasing the lock). */
    void touch(std::uint64_t digest);

    std::filesystem::path dir_;
    DiskKey key_;
    std::size_t block_bytes_;
    std::uint64_t cap_bytes_;
    /** Digest seed: a hash of the serialized key, so the same tokens
     *  under another key name another file. */
    std::uint64_t key_seed_ = 0;

    mutable std::mutex mu_;
    std::unordered_map<std::uint64_t, File> files_;
    std::uint64_t used_ = 0;
    std::uint64_t clock_ = 0;
    /** Names the temporary files of writes in flight. */
    std::uint64_t tmp_seq_ = 0;
};

}  // namespace locus::engine
