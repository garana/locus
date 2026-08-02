#include "locus/kv/kv_quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "locus/backend/cpu_ops.hpp"  // f16_to_f32, f32_to_f16

namespace locus::kv {

namespace {

using backend::f16_to_f32;
using backend::f32_to_f16;

/** Q8 block: f16 scale + 32 int8. */
constexpr std::size_t kQ8Block = 2 + kKvBlock;       // 34
/** Q4 block: f16 scale + 32 packed nibbles (16 bytes). */
constexpr std::size_t kQ4Block = 2 + kKvBlock / 2;   // 18

void store_u16(void* p, std::uint16_t v) {
    std::memcpy(p, &v, sizeof v);
}
std::uint16_t load_u16(const void* p) {
    std::uint16_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

}  // namespace

std::size_t kv_row_bytes(std::size_t n, KvType type) {
    switch (type) {
        case KvType::kF32:
            return n * sizeof(float);
        case KvType::kQ8:
            return (n / kKvBlock) * kQ8Block;
        case KvType::kQ4:
            return (n / kKvBlock) * kQ4Block;
    }
    return 0;
}

void kv_quantize_row(const float* src, std::size_t n, void* dst,
                     KvType type) {
    if (type == KvType::kF32) {
        std::memcpy(dst, src, n * sizeof(float));
        return;
    }
    auto* out = static_cast<std::uint8_t*>(dst);
    const std::size_t nb = n / kKvBlock;
    for (std::size_t b = 0; b < nb; ++b) {
        const float* x = src + b * kKvBlock;
        if (type == KvType::kQ8) {
            // Symmetric: d = amax/127, q = round(x/d) in [-127,127].
            float amax = 0.0f;
            for (std::size_t i = 0; i < kKvBlock; ++i) {
                amax = std::max(amax, std::fabs(x[i]));
            }
            const float d = amax / 127.0f;
            const float inv = d != 0.0f ? 1.0f / d : 0.0f;
            std::uint8_t* blk = out + b * kQ8Block;
            store_u16(blk, f32_to_f16(d));
            auto* q = reinterpret_cast<std::int8_t*>(blk + 2);
            for (std::size_t i = 0; i < kKvBlock; ++i) {
                const float v = std::round(x[i] * inv);
                q[i] = static_cast<std::int8_t>(
                    std::min(127.0f, std::max(-127.0f, v)));
            }
        } else {  // kQ4, ggml Q4_0-style asymmetric-around-8
            // d from the largest-magnitude signed element / -8.
            float max = 0.0f;
            for (std::size_t i = 0; i < kKvBlock; ++i) {
                if (std::fabs(x[i]) > std::fabs(max)) {
                    max = x[i];
                }
            }
            const float d = max / -8.0f;
            const float inv = d != 0.0f ? 1.0f / d : 0.0f;
            std::uint8_t* blk = out + b * kQ4Block;
            store_u16(blk, f32_to_f16(d));
            std::uint8_t* q = blk + 2;
            for (std::size_t i = 0; i < kKvBlock / 2; ++i) {
                const int lo = std::min(
                    15, static_cast<int>(x[i] * inv + 8.5f));
                const int hi = std::min(
                    15, static_cast<int>(
                            x[i + kKvBlock / 2] * inv + 8.5f));
                q[i] = static_cast<std::uint8_t>(
                    (lo & 0x0F) | ((hi & 0x0F) << 4));
            }
        }
    }
}

void kv_dequantize_row(const void* src, std::size_t n, float* dst,
                       KvType type) {
    if (type == KvType::kF32) {
        std::memcpy(dst, src, n * sizeof(float));
        return;
    }
    const auto* in = static_cast<const std::uint8_t*>(src);
    const std::size_t nb = n / kKvBlock;
    for (std::size_t b = 0; b < nb; ++b) {
        float* y = dst + b * kKvBlock;
        if (type == KvType::kQ8) {
            const std::uint8_t* blk = in + b * kQ8Block;
            const float d = f16_to_f32(load_u16(blk));
            const auto* q =
                reinterpret_cast<const std::int8_t*>(blk + 2);
            for (std::size_t i = 0; i < kKvBlock; ++i) {
                y[i] = q[i] * d;
            }
        } else {  // kQ4
            const std::uint8_t* blk = in + b * kQ4Block;
            const float d = f16_to_f32(load_u16(blk));
            const std::uint8_t* q = blk + 2;
            for (std::size_t i = 0; i < kKvBlock / 2; ++i) {
                const int lo = q[i] & 0x0F;
                const int hi = q[i] >> 4;
                y[i] = (lo - 8) * d;
                y[i + kKvBlock / 2] = (hi - 8) * d;
            }
        }
    }
}

}  // namespace locus::kv
