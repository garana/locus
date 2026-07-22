#pragma once

#include <cstddef>
#include <span>
#include <string>

namespace locus::sys {

/**
 * Read-only memory-mapped file (RAII, move-only).
 *
 * Model files are mapped PROT_READ so hostile content can never be
 * executed or written back; parsing works on a bounds-checked span.
 */
class MappedFile {
  public:
    /** Creates an empty (unmapped) instance. */
    MappedFile() = default;

    /**
     * Maps a file read-only.
     *
     * @param path Filesystem path to an existing regular file.
     * @throws std::system_error on open/stat/map failure.
     */
    explicit MappedFile(const std::string& path);

    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile();

    /** @returns The mapped bytes; empty span for an empty file. */
    std::span<const std::byte> bytes() const {
        return {static_cast<const std::byte*>(data_), size_};
    }

  private:
    void* data_ = nullptr;
    std::size_t size_ = 0;
};

/**
 * Asks the OS to start faulting in [p, p+n) asynchronously
 * (madvise WILLNEED, page-aligned). Best-effort: errors are
 * ignored. Used by the R8 expert-readahead policy to overlap the
 * SSD reads of the experts a token just routed to.
 */
void advise_willneed(const void* p, std::size_t n);

/**
 * Wires [p, p+n) into RAM (mlock, page-aligned). Best-effort:
 * on failure (RLIMIT_MEMLOCK, memory pressure) the range simply
 * stays demand-paged. Used by the R9 LOCUS_PIN_STATIC policy to
 * keep a streamed model's non-expert weights resident.
 *
 * @returns true when the whole range was locked.
 */
bool lock_resident(const void* p, std::size_t n);

}  // namespace locus::sys
