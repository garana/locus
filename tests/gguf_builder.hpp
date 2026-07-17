#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

/**
 * Byte-level GGUF image builder for tests: assembles arbitrary
 * (including deliberately malformed) file images in memory.
 */
class GgufBuilder {
  public:
    GgufBuilder& u8(std::uint8_t v) { return raw(&v, 1); }
    GgufBuilder& u32(std::uint32_t v) { return raw(&v, 4); }
    GgufBuilder& i32(std::int32_t v) { return raw(&v, 4); }
    GgufBuilder& u64(std::uint64_t v) { return raw(&v, 8); }
    GgufBuilder& f32(float v) { return raw(&v, 4); }

    GgufBuilder& str(std::string_view s) {
        u64(s.size());
        return raw(s.data(), s.size());
    }

    /** Standard header: magic, version 3, counts. */
    GgufBuilder& header(std::uint64_t n_tensors, std::uint64_t n_kv) {
        u32(0x46554747);
        u32(3);
        u64(n_tensors);
        u64(n_kv);
        return *this;
    }

    GgufBuilder& kv_string(std::string_view key, std::string_view val) {
        str(key);
        u32(8);  // ValueType::kString
        return str(val);
    }

    GgufBuilder& kv_u32(std::string_view key, std::uint32_t val) {
        str(key);
        u32(4);  // ValueType::kUint32
        return u32(val);
    }

    GgufBuilder& kv_bool(std::string_view key, bool val) {
        str(key);
        u32(7);  // ValueType::kBool
        return u8(val ? 1 : 0);
    }

    /** Array-of-f32 metadata entry. */
    GgufBuilder& kv_f32_array(std::string_view key,
                              std::span<const float> vals) {
        str(key);
        u32(9);  // ValueType::kArray
        u32(6);  // elem: kFloat32
        u64(vals.size());
        for (float v : vals) {
            f32(v);
        }
        return *this;
    }

    /** Array-of-string metadata entry. */
    GgufBuilder& kv_str_array(std::string_view key,
                              std::span<const std::string_view> vals) {
        str(key);
        u32(9);  // ValueType::kArray
        u32(8);  // elem: kString
        u64(vals.size());
        for (auto v : vals) {
            str(v);
        }
        return *this;
    }

    /** Array-of-i32 metadata entry. */
    GgufBuilder& kv_i32_array(std::string_view key,
                              std::span<const std::int32_t> vals) {
        str(key);
        u32(9);  // ValueType::kArray
        u32(5);  // elem: kInt32
        u64(vals.size());
        for (auto v : vals) {
            i32(v);
        }
        return *this;
    }

    /** Tensor descriptor entry. */
    GgufBuilder& tensor(std::string_view name,
                        std::span<const std::uint64_t> dims,
                        std::uint32_t type, std::uint64_t offset) {
        str(name);
        u32(static_cast<std::uint32_t>(dims.size()));
        for (auto d : dims) {
            u64(d);
        }
        u32(type);
        return u64(offset);
    }

    /** Pads with zeros to the given alignment (default 32). */
    GgufBuilder& pad(std::size_t alignment = 32) {
        while (out_.size() % alignment != 0) {
            out_.push_back(std::byte{0});
        }
        return *this;
    }

    /** Appends `n` zero bytes of tensor payload. */
    GgufBuilder& zeros(std::size_t n) {
        out_.resize(out_.size() + n, std::byte{0});
        return *this;
    }

    std::span<const std::byte> bytes() const { return out_; }

  private:
    GgufBuilder& raw(const void* p, std::size_t n) {
        auto* b = static_cast<const std::byte*>(p);
        out_.insert(out_.end(), b, b + n);
        return *this;
    }

    std::vector<std::byte> out_;
};
