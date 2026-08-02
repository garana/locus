#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "locus/sys/mmap_file.hpp"

namespace locus::gguf {

/** Thrown for any malformed or unsupported GGUF input. */
class Error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/** GGUF metadata value type ids, as in the spec. */
enum class ValueType : std::uint32_t {
    kUint8 = 0,
    kInt8 = 1,
    kUint16 = 2,
    kInt16 = 3,
    kUint32 = 4,
    kInt32 = 5,
    kFloat32 = 6,
    kBool = 7,
    kString = 8,
    kArray = 9,
    kUint64 = 10,
    kInt64 = 11,
    kFloat64 = 12,
};

/** ggml tensor data type ids (subset locus can size-check). */
enum class TensorType : std::uint32_t {
    kF32 = 0,
    kF16 = 1,
    kQ4_0 = 2,
    kQ4_1 = 3,
    kQ5_0 = 6,
    kQ5_1 = 7,
    kQ8_0 = 8,
    kQ8_1 = 9,
    kQ2_K = 10,
    kQ3_K = 11,
    kQ4_K = 12,
    kQ5_K = 13,
    kQ6_K = 14,
    kQ8_K = 15,
    kIQ2_XXS = 16,
    kIQ2_XS = 17,
    kIQ3_XXS = 18,
    kIQ1_S = 19,
    kIQ4_XS = 23,
    kTQ1_0 = 34,
    kTQ2_0 = 35,
};

/**
 * One parsed metadata value. Scalars live in the matching field;
 * arrays hold their elements recursively in `arr`.
 */
struct Value {
    ValueType type = ValueType::kUint8;
    std::uint64_t u = 0;
    std::int64_t i = 0;
    double f = 0.0;
    bool b = false;
    std::string s;
    /** Element type when type == kArray. */
    ValueType elem = ValueType::kUint8;
    std::vector<Value> arr;
};

/** Geometry and location of one tensor inside the data section. */
struct TensorInfo {
    std::string name;
    std::uint32_t n_dims = 0;
    /** Element counts per dimension; unused dims are 1. */
    std::uint64_t ne[4] = {1, 1, 1, 1};
    TensorType type = TensorType::kF32;
    /** Byte offset relative to the start of the data section. */
    std::uint64_t offset = 0;
    /** Validated total byte size of the tensor payload. */
    std::uint64_t nbytes = 0;

    /** @returns Total number of elements across all dims. */
    std::uint64_t nelements() const {
        return ne[0] * ne[1] * ne[2] * ne[3];
    }
};

/**
 * A parsed GGUF model file (versions 2 and 3, little-endian).
 *
 * Parsing is fully bounds-checked against the input span: every
 * count, string length, offset, and size is validated before use,
 * no allocation is sized by unvalidated input, and arithmetic that
 * could overflow uses checked helpers. Hostile files throw Error;
 * they must never crash the process (see tests/test_gguf.cpp).
 */
class GgufFile {
  public:
    /**
     * Maps and parses a model file. When the file is the first
     * shard of a split model (split.no == 0, split.count > 1),
     * sibling shards ("-NNNNN-of-MMMMM.gguf") are mapped too and
     * their tensors become visible through find_tensor and
     * tensor_data; tensors() lists only this file's own tensors.
     *
     * @param path Path to a .gguf file.
     * @throws Error on malformed content or a missing sibling
     *     shard, std::system_error on IO.
     */
    static GgufFile open(const std::string& path);

    /**
     * Parses an in-memory image (used by tests and fuzzing).
     *
     * @param buf The full file image; must outlive the GgufFile.
     */
    static GgufFile parse(std::span<const std::byte> buf);

    /** @returns GGUF format version (2 or 3). */
    std::uint32_t version() const { return version_; }

    /** @returns Data-section alignment (power of two). */
    std::uint32_t alignment() const { return alignment_; }

    /** @returns All metadata, keyed by name. */
    const std::map<std::string, Value, std::less<>>& metadata() const {
        return metadata_;
    }

    /** @returns The value for key, or nullptr if absent. */
    const Value* find(std::string_view key) const;

    /** @returns String value for key, if present and a string. */
    std::optional<std::string_view> get_string(std::string_view key) const;

    /** @returns Any unsigned/positive integer value for key. */
    std::optional<std::uint64_t> get_uint(std::string_view key) const;

    /** @returns Bool value for key, if present and a bool. */
    std::optional<bool> get_bool(std::string_view key) const;

    /** @returns Array elements for key, or nullptr. */
    const std::vector<Value>* get_array(std::string_view key) const;

    /**
     * @returns true when the image is a read-only file mapping
     *     (open()); false for in-memory images (parse()), whose
     *     anonymous pages must never receive madvise DONTNEED --
     *     that would discard their contents.
     */
    bool file_backed() const { return !file_.bytes().empty(); }

    /** @returns Tensor descriptors in file order. */
    const std::vector<TensorInfo>& tensors() const { return tensors_; }

    /** @returns The tensor named `name` (any shard), or nullptr. */
    const TensorInfo* find_tensor(std::string_view name) const;

    /** @returns Tensor count across this file and all shards. */
    std::size_t total_tensor_count() const;

    /**
     * @param info A descriptor obtained from this file.
     * @returns The tensor's bytes inside the mapped data section.
     */
    std::span<const std::byte> tensor_data(const TensorInfo& info) const;

  private:
    GgufFile() = default;
    void parse_buffer();
    void open_shards(const std::string& path);
    /** @returns true when info belongs to this file's tensors_. */
    bool owns(const TensorInfo& info) const;

    sys::MappedFile file_;
    std::span<const std::byte> buf_;
    std::span<const std::byte> data_;
    std::uint32_t version_ = 0;
    std::uint32_t alignment_ = 32;
    std::map<std::string, Value, std::less<>> metadata_;
    std::vector<TensorInfo> tensors_;
    /** Sibling shards of a split model (first shard only). */
    std::vector<std::unique_ptr<GgufFile>> shards_;
};

}  // namespace locus::gguf
