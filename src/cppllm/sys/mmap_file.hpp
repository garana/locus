#pragma once

#include <cstddef>
#include <span>
#include <string>

namespace cppllm::sys {

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

}  // namespace cppllm::sys
