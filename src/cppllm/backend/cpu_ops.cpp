#include "cppllm/backend/cpu_ops.hpp"

#include <cassert>
#include <cmath>
#include <cstring>

namespace cppllm::backend {

namespace {

/** Q8_0 block: 32 elems, f16 scale + 32 int8 quants (34 bytes). */
constexpr std::size_t kQ8_0BlockBytes = 34;
/** Q4_0 block: 32 elems, f16 scale + 16 nibble bytes (18 bytes). */
constexpr std::size_t kQ4_0BlockBytes = 18;

std::uint16_t load_u16(const std::byte* p) {
    std::uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

float dot_f32(const std::byte* row, std::span<const float> x) {
    float acc = 0.0f;
    const float* w = reinterpret_cast<const float*>(row);
    for (std::size_t i = 0; i < x.size(); ++i) {
        acc += w[i] * x[i];
    }
    return acc;
}

float dot_f16(const std::byte* row, std::span<const float> x) {
    float acc = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        acc += f16_to_f32(load_u16(row + 2 * i)) * x[i];
    }
    return acc;
}

float dot_q8_0(const std::byte* row, std::span<const float> x) {
    float acc = 0.0f;
    const std::size_t nb = x.size() / 32;
    for (std::size_t b = 0; b < nb; ++b) {
        const std::byte* blk = row + b * kQ8_0BlockBytes;
        const float d = f16_to_f32(load_u16(blk));
        const auto* q = reinterpret_cast<const std::int8_t*>(blk + 2);
        float s = 0.0f;
        for (std::size_t i = 0; i < 32; ++i) {
            s += static_cast<float>(q[i]) * x[b * 32 + i];
        }
        acc += d * s;
    }
    return acc;
}

float dot_q4_0(const std::byte* row, std::span<const float> x) {
    float acc = 0.0f;
    const std::size_t nb = x.size() / 32;
    for (std::size_t b = 0; b < nb; ++b) {
        const std::byte* blk = row + b * kQ4_0BlockBytes;
        const float d = f16_to_f32(load_u16(blk));
        const auto* q = reinterpret_cast<const std::uint8_t*>(blk + 2);
        float s = 0.0f;
        for (std::size_t i = 0; i < 16; ++i) {
            const float lo = static_cast<float>(q[i] & 0x0f) - 8.0f;
            const float hi = static_cast<float>(q[i] >> 4) - 8.0f;
            s += lo * x[b * 32 + i] + hi * x[b * 32 + i + 16];
        }
        acc += d * s;
    }
    return acc;
}

/** @returns Bytes per row for a supported matvec weight type. */
std::size_t row_bytes(const Mat& w) {
    switch (w.type) {
        case gguf::TensorType::kF32: return w.cols * 4ull;
        case gguf::TensorType::kF16: return w.cols * 2ull;
        case gguf::TensorType::kQ8_0:
            return w.cols / 32ull * kQ8_0BlockBytes;
        case gguf::TensorType::kQ4_0:
            return w.cols / 32ull * kQ4_0BlockBytes;
        default:
            throw gguf::Error("no CPU kernel for this weight type");
    }
}

}  // namespace

float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u)
                               << 16;
    std::uint32_t exp = (h >> 10) & 0x1f;
    std::uint32_t mant = h & 0x3ffu;
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // +-0
        } else {
            // Subnormal: renormalize into the f32 exponent range.
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);  // inf/nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

std::uint16_t f32_to_f16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const std::uint16_t sign =
        static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const std::int32_t exp =
        static_cast<std::int32_t>((bits >> 23) & 0xff) - 127 + 15;
    std::uint32_t mant = bits & 0x7fffffu;
    if (exp >= 31) {
        // Overflow or inf/nan.
        const bool nan = ((bits >> 23) & 0xff) == 0xff && mant != 0;
        return sign | 0x7c00 | (nan ? 0x200 : 0);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return sign;  // underflows to zero
        }
        mant |= 0x800000u;  // implicit bit
        const std::uint32_t shift =
            static_cast<std::uint32_t>(14 - exp);
        const std::uint16_t sub =
            static_cast<std::uint16_t>(mant >> shift);
        const std::uint32_t rem = mant & ((1u << shift) - 1);
        return sign |
               static_cast<std::uint16_t>(
                   sub + (rem > (1u << (shift - 1)) ? 1 : 0));
    }
    std::uint16_t out = static_cast<std::uint16_t>(
        sign | (exp << 10) | (mant >> 13));
    // Round to nearest even on the dropped 13 bits.
    const std::uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (out & 1))) {
        ++out;
    }
    return out;
}

void rmsnorm(std::span<const float> x, std::span<const float> w,
             float eps, std::span<float> out) {
    assert(x.size() == w.size() && x.size() == out.size());
    float ss = 0.0f;
    for (float v : x) {
        ss += v * v;
    }
    const float scale =
        1.0f / std::sqrt(ss / static_cast<float>(x.size()) + eps);
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] * scale * w[i];
    }
}

void softmax_inplace(std::span<float> x) {
    float mx = x[0];
    for (float v : x) {
        mx = std::max(mx, v);
    }
    float sum = 0.0f;
    for (float& v : x) {
        v = std::exp(v - mx);
        sum += v;
    }
    for (float& v : x) {
        v /= sum;
    }
}

void silu_mul(std::span<const float> gate, std::span<const float> up,
              std::span<float> out) {
    assert(gate.size() == up.size() && gate.size() == out.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const float g = gate[i];
        out[i] = g / (1.0f + std::exp(-g)) * up[i];
    }
}

void rope_norm(std::span<float> x, std::uint32_t n_heads,
               std::uint32_t head_dim, std::uint32_t pos,
               float freq_base, std::span<const float> factors) {
    assert(x.size() == static_cast<std::size_t>(n_heads) * head_dim);
    assert(factors.empty() || factors.size() >= head_dim / 2);
    for (std::uint32_t h = 0; h < n_heads; ++h) {
        float* v = x.data() + static_cast<std::size_t>(h) * head_dim;
        for (std::uint32_t i = 0; i + 1 < head_dim; i += 2) {
            const float theta =
                static_cast<float>(pos) *
                std::pow(freq_base,
                         -static_cast<float>(i) /
                             static_cast<float>(head_dim)) /
                (factors.empty() ? 1.0f : factors[i / 2]);
            const float c = std::cos(theta);
            const float s = std::sin(theta);
            const float x0 = v[i];
            const float x1 = v[i + 1];
            v[i] = x0 * c - x1 * s;
            v[i + 1] = x0 * s + x1 * c;
        }
    }
}

void matvec(const Mat& w, std::span<const float> x,
            std::span<float> out) {
    assert(x.size() == w.cols && out.size() == w.rows);
    const std::size_t stride = row_bytes(w);
    for (std::uint32_t r = 0; r < w.rows; ++r) {
        const std::byte* row = w.data + r * stride;
        switch (w.type) {
            case gguf::TensorType::kF32:
                out[r] = dot_f32(row, x);
                break;
            case gguf::TensorType::kF16:
                out[r] = dot_f16(row, x);
                break;
            case gguf::TensorType::kQ8_0:
                out[r] = dot_q8_0(row, x);
                break;
            default:
                out[r] = dot_q4_0(row, x);
                break;
        }
    }
}

void dequant_row(const Mat& w, std::uint32_t row,
                 std::span<float> out) {
    assert(out.size() == w.cols && row < w.rows);
    const std::byte* p = w.data + row * row_bytes(w);
    switch (w.type) {
        case gguf::TensorType::kF32:
            std::memcpy(out.data(), p, w.cols * sizeof(float));
            break;
        case gguf::TensorType::kF16:
            for (std::uint32_t i = 0; i < w.cols; ++i) {
                out[i] = f16_to_f32(load_u16(p + 2 * i));
            }
            break;
        case gguf::TensorType::kQ8_0:
            for (std::uint32_t b = 0; b < w.cols / 32; ++b) {
                const std::byte* blk = p + b * kQ8_0BlockBytes;
                const float d = f16_to_f32(load_u16(blk));
                const auto* q =
                    reinterpret_cast<const std::int8_t*>(blk + 2);
                for (std::uint32_t i = 0; i < 32; ++i) {
                    out[b * 32 + i] = d * static_cast<float>(q[i]);
                }
            }
            break;
        default: {
            for (std::uint32_t b = 0; b < w.cols / 32; ++b) {
                const std::byte* blk = p + b * kQ4_0BlockBytes;
                const float d = f16_to_f32(load_u16(blk));
                const auto* q =
                    reinterpret_cast<const std::uint8_t*>(blk + 2);
                for (std::uint32_t i = 0; i < 16; ++i) {
                    out[b * 32 + i] =
                        d * (static_cast<float>(q[i] & 0x0f) - 8.0f);
                    out[b * 32 + i + 16] =
                        d * (static_cast<float>(q[i] >> 4) - 8.0f);
                }
            }
            break;
        }
    }
}

}  // namespace cppllm::backend
