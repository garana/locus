#include "locus/gguf/gguf.hpp"

#include <cstring>
#include <unordered_set>

namespace locus::gguf {

namespace {

constexpr std::uint32_t kMagic = 0x46554747;  // "GGUF" little-endian
constexpr int kMaxArrayDepth = 8;

/** Per-tensor-type block geometry; {0, 0} marks unsupported ids. */
struct TypeTraits {
    std::uint32_t block_elems;
    std::uint32_t block_bytes;
};

TypeTraits traits(TensorType t) {
    switch (t) {
        case TensorType::kF32: return {1, 4};
        case TensorType::kF16: return {1, 2};
        case TensorType::kQ4_0: return {32, 18};
        case TensorType::kQ4_1: return {32, 20};
        case TensorType::kQ5_0: return {32, 22};
        case TensorType::kQ5_1: return {32, 24};
        case TensorType::kQ8_0: return {32, 34};
        case TensorType::kQ8_1: return {32, 36};
        case TensorType::kQ2_K: return {256, 84};
        case TensorType::kQ3_K: return {256, 110};
        case TensorType::kQ4_K: return {256, 144};
        case TensorType::kQ5_K: return {256, 176};
        case TensorType::kQ6_K: return {256, 210};
        case TensorType::kQ8_K: return {256, 292};
    }
    return {0, 0};
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    std::uint64_t r = 0;
    if (__builtin_mul_overflow(a, b, &r)) {
        throw Error("integer overflow in size computation");
    }
    return r;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    std::uint64_t r = 0;
    if (__builtin_add_overflow(a, b, &r)) {
        throw Error("integer overflow in offset computation");
    }
    return r;
}

/**
 * Bounds-checked sequential reader over the file image. Every read
 * validates the requested length against the bytes remaining, so
 * truncated or hostile inputs surface as Error, never as OOB reads.
 */
class Reader {
  public:
    explicit Reader(std::span<const std::byte> buf) : buf_(buf) {}

    std::size_t offset() const { return off_; }
    std::size_t remaining() const { return buf_.size() - off_; }

    template <typename T>
    T scalar() {
        static_assert(std::is_trivially_copyable_v<T>);
        need(sizeof(T));
        T v;
        std::memcpy(&v, buf_.data() + off_, sizeof(T));
        off_ += sizeof(T);
        return v;
    }

    std::string str() {
        std::uint64_t len = scalar<std::uint64_t>();
        if (len > remaining()) {
            throw Error("string length exceeds file size");
        }
        std::string s(reinterpret_cast<const char*>(buf_.data() + off_),
                      static_cast<std::size_t>(len));
        off_ += static_cast<std::size_t>(len);
        return s;
    }

  private:
    void need(std::size_t n) {
        if (n > remaining()) {
            throw Error("unexpected end of file");
        }
    }

    std::span<const std::byte> buf_;
    std::size_t off_ = 0;
};

Value parse_value(Reader& r, ValueType type, int depth) {
    Value v;
    v.type = type;
    switch (type) {
        case ValueType::kUint8: v.u = r.scalar<std::uint8_t>(); break;
        case ValueType::kUint16: v.u = r.scalar<std::uint16_t>(); break;
        case ValueType::kUint32: v.u = r.scalar<std::uint32_t>(); break;
        case ValueType::kUint64: v.u = r.scalar<std::uint64_t>(); break;
        case ValueType::kInt8: v.i = r.scalar<std::int8_t>(); break;
        case ValueType::kInt16: v.i = r.scalar<std::int16_t>(); break;
        case ValueType::kInt32: v.i = r.scalar<std::int32_t>(); break;
        case ValueType::kInt64: v.i = r.scalar<std::int64_t>(); break;
        case ValueType::kFloat32: v.f = r.scalar<float>(); break;
        case ValueType::kFloat64: v.f = r.scalar<double>(); break;
        case ValueType::kBool: v.b = r.scalar<std::uint8_t>() != 0; break;
        case ValueType::kString: v.s = r.str(); break;
        case ValueType::kArray: {
            if (depth >= kMaxArrayDepth) {
                throw Error("array nesting too deep");
            }
            v.elem = static_cast<ValueType>(r.scalar<std::uint32_t>());
            std::uint64_t count = r.scalar<std::uint64_t>();
            // No reserve(count): growth is bounded by actual bytes
            // parsed, so a huge declared count cannot balloon
            // memory -- it just hits end-of-file.
            for (std::uint64_t k = 0; k < count; ++k) {
                v.arr.push_back(parse_value(r, v.elem, depth + 1));
            }
            break;
        }
        default:
            throw Error("unknown metadata value type");
    }
    return v;
}

}  // namespace

GgufFile GgufFile::open(const std::string& path) {
    GgufFile g;
    g.file_ = sys::MappedFile(path);
    g.buf_ = g.file_.bytes();
    g.parse_buffer();
    return g;
}

GgufFile GgufFile::parse(std::span<const std::byte> buf) {
    GgufFile g;
    g.buf_ = buf;
    g.parse_buffer();
    return g;
}

void GgufFile::parse_buffer() {
    Reader r(buf_);
    if (r.scalar<std::uint32_t>() != kMagic) {
        throw Error("not a GGUF file (bad magic)");
    }
    version_ = r.scalar<std::uint32_t>();
    if (version_ != 2 && version_ != 3) {
        throw Error("unsupported GGUF version");
    }
    std::uint64_t n_tensors = r.scalar<std::uint64_t>();
    std::uint64_t n_kv = r.scalar<std::uint64_t>();

    for (std::uint64_t k = 0; k < n_kv; ++k) {
        std::string key = r.str();
        auto type = static_cast<ValueType>(r.scalar<std::uint32_t>());
        Value v = parse_value(r, type, 0);
        if (!metadata_.emplace(std::move(key), std::move(v)).second) {
            throw Error("duplicate metadata key");
        }
    }

    if (auto a = get_uint("general.alignment")) {
        if (*a == 0 || (*a & (*a - 1)) != 0 || *a > (1u << 20)) {
            throw Error("alignment is not a power of two");
        }
        alignment_ = static_cast<std::uint32_t>(*a);
    }

    std::unordered_set<std::string_view> names;
    for (std::uint64_t k = 0; k < n_tensors; ++k) {
        TensorInfo t;
        t.name = r.str();
        t.n_dims = r.scalar<std::uint32_t>();
        if (t.n_dims < 1 || t.n_dims > 4) {
            throw Error("tensor rank out of range");
        }
        std::uint64_t nelems = 1;
        for (std::uint32_t d = 0; d < t.n_dims; ++d) {
            t.ne[d] = r.scalar<std::uint64_t>();
            if (t.ne[d] == 0) {
                throw Error("tensor dimension is zero");
            }
            nelems = checked_mul(nelems, t.ne[d]);
        }
        t.type = static_cast<TensorType>(r.scalar<std::uint32_t>());
        t.offset = r.scalar<std::uint64_t>();

        auto tt = traits(t.type);
        if (tt.block_elems == 0) {
            throw Error("unsupported tensor data type");
        }
        if (t.ne[0] % tt.block_elems != 0) {
            throw Error("row size not a multiple of the block size");
        }
        t.nbytes = checked_mul(nelems / t.ne[0],
                               checked_mul(t.ne[0] / tt.block_elems,
                                           tt.block_bytes));
        if (t.offset % alignment_ != 0) {
            throw Error("tensor offset not aligned");
        }
        tensors_.push_back(std::move(t));
    }
    for (const TensorInfo& t : tensors_) {
        if (!names.insert(t.name).second) {
            throw Error("duplicate tensor name");
        }
    }

    std::uint64_t data_start =
        checked_add(r.offset(), alignment_ - 1) / alignment_ * alignment_;
    if (data_start > buf_.size()) {
        if (!tensors_.empty()) {
            throw Error("data section past end of file");
        }
        data_start = buf_.size();
    }
    data_ = buf_.subspan(static_cast<std::size_t>(data_start));

    for (const TensorInfo& t : tensors_) {
        if (checked_add(t.offset, t.nbytes) > data_.size()) {
            throw Error("tensor extends past end of file");
        }
    }
}

const Value* GgufFile::find(std::string_view key) const {
    auto it = metadata_.find(key);
    return it == metadata_.end() ? nullptr : &it->second;
}

std::optional<std::string_view> GgufFile::get_string(
    std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr || v->type != ValueType::kString) {
        return std::nullopt;
    }
    return std::string_view(v->s);
}

std::optional<std::uint64_t> GgufFile::get_uint(std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr) {
        return std::nullopt;
    }
    switch (v->type) {
        case ValueType::kUint8:
        case ValueType::kUint16:
        case ValueType::kUint32:
        case ValueType::kUint64:
            return v->u;
        case ValueType::kInt8:
        case ValueType::kInt16:
        case ValueType::kInt32:
        case ValueType::kInt64:
            if (v->i < 0) {
                return std::nullopt;
            }
            return static_cast<std::uint64_t>(v->i);
        default:
            return std::nullopt;
    }
}

std::optional<bool> GgufFile::get_bool(std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr || v->type != ValueType::kBool) {
        return std::nullopt;
    }
    return v->b;
}

const std::vector<Value>* GgufFile::get_array(std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr || v->type != ValueType::kArray) {
        return nullptr;
    }
    return &v->arr;
}

const TensorInfo* GgufFile::find_tensor(std::string_view name) const {
    for (const TensorInfo& t : tensors_) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

std::span<const std::byte> GgufFile::tensor_data(
    const TensorInfo& info) const {
    // Re-validated here so a corrupted TensorInfo from elsewhere
    // cannot produce an out-of-bounds span.
    if (checked_add(info.offset, info.nbytes) > data_.size()) {
        throw Error("tensor extends past end of file");
    }
    return data_.subspan(static_cast<std::size_t>(info.offset),
                         static_cast<std::size_t>(info.nbytes));
}

}  // namespace locus::gguf
