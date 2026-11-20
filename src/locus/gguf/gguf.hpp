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

/**
 * ggml tensor data type ids (subset locus can size-check).
 *
 * A quantized type stores the weights in blocks. Each block holds a
 * few scales (and sometimes a minimum) plus one small integer code per
 * weight; a weight is rebuilt as scale * code, or scale * code - min.
 * "bpw" below is the bits per weight, scales included: block bytes * 8
 * / block weights. d is the block scale, m the block minimum.
 *
 * The name says the family:
 * - Qn_0: blocks of 32 weights, an f16 d, n-bit codes centered on
 *   zero (weight = d * code).
 * - Qn_1: as Qn_0, plus an f16 m (weight = d * code + m).
 * - Qn_K ("k-quants"): super-blocks of 256 weights cut into sub-blocks
 *   of 16 or 32; each sub-block has its own small scale (and min),
 *   themselves scaled by an f16 d (and dmin) of the super-block.
 * - IQn ("i-quants"): super-blocks of 256 (IQ4_NL: blocks of 32);
 *   groups of 4 or 8 weights share one index into a fixed table of
 *   value patterns, plus sign bits; 1.5625 to 4.5 bpw. _XXS, _XS, _S
 *   and _M name variants of one width, from fewest to most bits.
 * - TQn_0: ternary weights (each -1, 0 or +1, times d), for BitNet
 *   b1.58 models.
 *
 * A file named after a recipe (Q4_K_M, Q5_K_S) mixes several of these
 * types, chosen per tensor.
 */
enum class TensorType : std::uint32_t {
    /** IEEE 754 32-bit float; 32 bpw, no loss. */
    kF32 = 0,
    /** IEEE 754 16-bit float; 16 bpw. */
    kF16 = 1,
    /** 4-bit codes, weight = d * (code - 8); 4.5 bpw. */
    kQ4_0 = 2,
    /** 4-bit codes, weight = d * code + m; 5 bpw. locus does not
     *  compute with it. */
    kQ4_1 = 3,
    /** 5-bit codes, weight = d * (code - 16); 5.5 bpw. */
    kQ5_0 = 6,
    /** 5-bit codes, weight = d * code + m; 6 bpw. locus does not
     *  compute with it. */
    kQ5_1 = 7,
    /** 8-bit signed codes, weight = d * code; 8.5 bpw. */
    kQ8_0 = 8,
    /** Q8_0 plus the block's sum; 9 bpw. ggml uses it for
     *  activations; locus does not compute with it. */
    kQ8_1 = 9,
    /** 2-bit codes; 16 sub-blocks of 16 weights, each with a 4-bit
     *  scale and a 4-bit min; 2.625 bpw. */
    kQ2_K = 10,
    /** 3-bit codes; 16 sub-blocks of 16 weights with 6-bit scales;
     *  3.4375 bpw. */
    kQ3_K = 11,
    /** 4-bit codes; 8 sub-blocks of 32 weights, each with a 6-bit
     *  scale and a 6-bit min; 4.5 bpw. */
    kQ4_K = 12,
    /** 5-bit codes, sub-blocks as Q4_K; 5.5 bpw. */
    kQ5_K = 13,
    /** 6-bit codes; 16 sub-blocks of 16 weights with 8-bit scales;
     *  6.5625 bpw. */
    kQ6_K = 14,
    /** 8-bit codes, an f32 d and the sum of each 16 codes; 9.125 bpw.
     *  locus quantizes activations to it for k-quant dot products. */
    kQ8_K = 15,
    /** 8 weights per index into a 256-entry table, a 4-bit scale per
     *  32 weights; 2.0625 bpw. */
    kIQ2_XXS = 16,
    /** 8 weights per index into a 512-entry table, a 4-bit scale per
     *  16 weights; 2.3125 bpw. */
    kIQ2_XS = 17,
    /** 4 weights per index into a 256-entry table, a 4-bit scale per
     *  32 weights; 3.0625 bpw. */
    kIQ3_XXS = 18,
    /** 8 weights (each -1, 0 or +1, plus a shift) per index into a
     *  2048-entry table, a 3-bit scale per 32 weights; 1.5625 bpw. */
    kIQ1_S = 19,
    /** Blocks of 32: 4-bit codes mapped through a fixed table of 16
     *  unevenly spaced values ("non-linear"), times an f16 d; 4.5 bpw. */
    kIQ4_NL = 20,
    /** 4 weights per index into a 512-entry table, a 4-bit scale per
     *  32 weights; 3.4375 bpw. */
    kIQ3_S = 21,
    /** 8 weights per index into a 1024-entry table, a 4-bit scale per
     *  16 weights; 2.5625 bpw. */
    kIQ2_S = 22,
    /** IQ4_NL's 16-value table in super-blocks of 256, a 6-bit scale
     *  per 32 weights; 4.25 bpw. */
    kIQ4_XS = 23,
    /** IQ1_S's table, a 3-bit scale per 16 weights; 1.75 bpw. */
    kIQ1_M = 29,
    /** bfloat16: the top 16 bits of an f32 (same range, fewer digits);
     *  16 bpw. */
    kBF16 = 30,
    /** Ternary, 5 weights packed per byte in base 3; 1.6875 bpw. */
    kTQ1_0 = 34,
    /** Ternary, 2 bits per weight; 2.0625 bpw. */
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

    /** Calls `fn(file)` for this file, then each sibling shard. */
    template <typename F>
    void for_each_file(F&& fn) const {
        fn(*this);
        for (const auto& s : shards_) {
            fn(*s);
        }
    }

    /** @returns This file's size in bytes. */
    std::size_t file_bytes() const { return buf_.size(); }

    /** @returns This file's bytes before its data section: the header,
     *  the metadata and the tensor table (the disk prefix cache hashes
     *  them to name the model, i#416). */
    std::span<const std::byte> header_bytes() const {
        if (data_.data() == nullptr) {
            return buf_;
        }
        return buf_.first(
            static_cast<std::size_t>(data_.data() - buf_.data()));
    }

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
