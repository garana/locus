#include "locus/backend/cpu_ops.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

namespace locus::backend {

namespace {

/** Q8_0 block: 32 elems, f16 scale + 32 int8 quants (34 bytes). */
constexpr std::size_t kQ8_0BlockBytes = 34;
/** Q4_0 block: 32 elems, f16 scale + 16 nibble bytes (18 bytes). */
constexpr std::size_t kQ4_0BlockBytes = 18;
/** Q5_0 block: 32 elems, f16 scale + 4 high-bit bytes + 16
 * nibble bytes (22 bytes). */
constexpr std::size_t kQ5_0BlockBytes = 22;
constexpr std::size_t kIQ4_NLBlockBytes = 18;

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

/** Dequantizes one Q5_0 block (32 elems) into y. */
void dequant_block_q5_0(const std::byte* blk, float* y) {
    const float d = f16_to_f32(load_u16(blk));
    std::uint32_t qh;
    std::memcpy(&qh, blk + 2, 4);
    const auto* qs =
        reinterpret_cast<const std::uint8_t*>(blk + 6);
    for (int j = 0; j < 16; ++j) {
        const std::uint8_t xh0 =
            static_cast<std::uint8_t>(((qh >> j) << 4) & 0x10);
        const std::uint8_t xh1 =
            static_cast<std::uint8_t>((qh >> (j + 12)) & 0x10);
        y[j] = d * (static_cast<float>((qs[j] & 0x0f) | xh0) -
                    16.0f);
        y[j + 16] =
            d * (static_cast<float>((qs[j] >> 4) | xh1) - 16.0f);
    }
}

float dot_q5_0(const std::byte* row, std::span<const float> x) {
    float acc = 0.0f;
    float tmp[32];
    for (std::size_t b = 0; b < x.size() / 32; ++b) {
        dequant_block_q5_0(row + b * kQ5_0BlockBytes, tmp);
        for (int i = 0; i < 32; ++i) {
            acc += tmp[i] * x[b * 32 + i];
        }
    }
    return acc;
}

/**
 * Unpacks the 6-bit scale/min pair of K-quant sub-block j from
 * the packed 12-byte scales array (ggml's get_scale_min_k4).
 */
void scale_min_k4(int j, const std::uint8_t* q,
                  std::uint8_t& d, std::uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = static_cast<std::uint8_t>((q[j + 4] & 0x0f) |
                                      ((q[j - 4] >> 6) << 4));
        m = static_cast<std::uint8_t>((q[j + 4] >> 4) |
                                      ((q[j] >> 6) << 4));
    }
}

/** Dequantizes one 256-element Q4_K super-block (144 bytes). */
void dequant_block_q4_k(const std::byte* blk, float* y) {
    const float d = f16_to_f32(load_u16(blk));
    const float dmin = f16_to_f32(load_u16(blk + 2));
    const auto* scales =
        reinterpret_cast<const std::uint8_t*>(blk + 4);
    const auto* q =
        reinterpret_cast<const std::uint8_t*>(blk + 16);
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        std::uint8_t sc, mn;
        scale_min_k4(is + 0, scales, sc, mn);
        const float d1 = d * sc, m1 = dmin * mn;
        scale_min_k4(is + 1, scales, sc, mn);
        const float d2 = d * sc, m2 = dmin * mn;
        for (int l = 0; l < 32; ++l) {
            *y++ = d1 * static_cast<float>(q[l] & 0x0f) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            *y++ = d2 * static_cast<float>(q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

/** Dequantizes one 256-element Q5_K super-block (176 bytes). */
void dequant_block_q5_k(const std::byte* blk, float* y) {
    const float d = f16_to_f32(load_u16(blk));
    const float dmin = f16_to_f32(load_u16(blk + 2));
    const auto* scales =
        reinterpret_cast<const std::uint8_t*>(blk + 4);
    const auto* qh =
        reinterpret_cast<const std::uint8_t*>(blk + 16);
    const auto* ql =
        reinterpret_cast<const std::uint8_t*>(blk + 48);
    int is = 0;
    std::uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        std::uint8_t sc, mn;
        scale_min_k4(is + 0, scales, sc, mn);
        const float d1 = d * sc, m1 = dmin * mn;
        scale_min_k4(is + 1, scales, sc, mn);
        const float d2 = d * sc, m2 = dmin * mn;
        for (int l = 0; l < 32; ++l) {
            *y++ = d1 * static_cast<float>((ql[l] & 0x0f) +
                                           ((qh[l] & u1) ? 16
                                                         : 0)) -
                   m1;
        }
        for (int l = 0; l < 32; ++l) {
            *y++ = d2 * static_cast<float>((ql[l] >> 4) +
                                           ((qh[l] & u2) ? 16
                                                         : 0)) -
                   m2;
        }
        ql += 32;
        is += 2;
        u1 = static_cast<std::uint8_t>(u1 << 2);
        u2 = static_cast<std::uint8_t>(u2 << 2);
    }
}

/** Dequantizes one 256-element Q6_K super-block (210 bytes). */
void dequant_block_q6_k(const std::byte* blk, float* y) {
    const auto* ql =
        reinterpret_cast<const std::uint8_t*>(blk);
    const auto* qh =
        reinterpret_cast<const std::uint8_t*>(blk + 128);
    const auto* sc =
        reinterpret_cast<const std::int8_t*>(blk + 192);
    const float d = f16_to_f32(load_u16(blk + 208));
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 =
                static_cast<int>((ql[l] & 0x0f) |
                                 (((qh[l] >> 0) & 3) << 4)) -
                32;
            const int q2 =
                static_cast<int>((ql[l + 32] & 0x0f) |
                                 (((qh[l] >> 2) & 3) << 4)) -
                32;
            const int q3 =
                static_cast<int>((ql[l] >> 4) |
                                 (((qh[l] >> 4) & 3) << 4)) -
                32;
            const int q4 =
                static_cast<int>((ql[l + 32] >> 4) |
                                 (((qh[l] >> 6) & 3) << 4)) -
                32;
            y[l] = d * sc[is] * static_cast<float>(q1);
            y[l + 32] = d * sc[is + 2] * static_cast<float>(q2);
            y[l + 64] = d * sc[is + 4] * static_cast<float>(q3);
            y[l + 96] = d * sc[is + 6] * static_cast<float>(q4);
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

// ----------------------------------------------------------------
// Q2_K/Q3_K and IQ-family super-block dequantizers, ported from
// ggml's dequantize_row_* reference implementations (llama.cpp,
// MIT); the IQ lookup grids are vendored in third_party/ggml.
// ----------------------------------------------------------------
#include "iq_grids.h"

constexpr std::uint8_t kmask_iq2xs[8] = {1,  2,  4,  8,
                                         16, 32, 64, 128};

/** Q2_K (84 bytes): scales[16] | qs[64] | d | dmin. */
void dequant_block_q2_k(const std::byte* blk, float* y) {
    const auto* scales =
        reinterpret_cast<const std::uint8_t*>(blk);
    const auto* q =
        reinterpret_cast<const std::uint8_t*>(blk + 16);
    const float d = f16_to_f32(load_u16(blk + 80));
    const float min = f16_to_f32(load_u16(blk + 82));
    int is = 0;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            std::uint8_t sc = scales[is++];
            float dl = d * (sc & 0x0f), ml = min * (sc >> 4);
            for (int l = 0; l < 16; ++l) {
                *y++ = dl * ((q[l] >> shift) & 3) - ml;
            }
            sc = scales[is++];
            dl = d * (sc & 0x0f);
            ml = min * (sc >> 4);
            for (int l = 0; l < 16; ++l) {
                *y++ = dl * ((q[l + 16] >> shift) & 3) - ml;
            }
            shift += 2;
        }
        q += 32;
    }
}

/** Q3_K (110 bytes): hmask[32] | qs[64] | scales[12] | d. */
void dequant_block_q3_k(const std::byte* blk, float* y) {
    const auto* hm = reinterpret_cast<const std::uint8_t*>(blk);
    const auto* q =
        reinterpret_cast<const std::uint8_t*>(blk + 32);
    const float d_all = f16_to_f32(load_u16(blk + 108));

    constexpr std::uint32_t kmask1 = 0x03030303;
    constexpr std::uint32_t kmask2 = 0x0f0f0f0f;
    std::uint32_t aux[4];
    std::memcpy(aux, blk + 96, 12);
    const std::uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) |
             (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) |
             (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const auto* scales =
        reinterpret_cast<const std::int8_t*>(aux);

    int is = 0;
    std::uint8_t m = 1;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) {
                *y++ = dl * (static_cast<int>((q[l] >> shift) &
                                              3) -
                             ((hm[l] & m) ? 0 : 4));
            }
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) {
                *y++ = dl *
                       (static_cast<int>((q[l + 16] >> shift) &
                                         3) -
                        ((hm[l + 16] & m) ? 0 : 4));
            }
            shift += 2;
            m = static_cast<std::uint8_t>(m << 1);
        }
        q += 32;
    }
}

/** IQ2_XXS (66 bytes): d | qs as 32 u16. */
void dequant_block_iq2_xxs(const std::byte* blk, float* y) {
    const float d = f16_to_f32(load_u16(blk));
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        std::uint32_t aux32[2];
        std::memcpy(aux32, blk + 2 + 8 * ib32, 8);
        const auto* aux8 =
            reinterpret_cast<const std::uint8_t*>(aux32);
        const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const auto* grid =
                reinterpret_cast<const std::uint8_t*>(
                    iq2xxs_grid + aux8[l]);
            const std::uint8_t signs =
                ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
            for (int j = 0; j < 8; ++j) {
                y[j] = db * grid[j] *
                       ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
            }
            y += 8;
        }
    }
}

/** IQ3_XXS (98 bytes): d | qs[64] | scales_and_signs[32]. */
void dequant_block_iq3_xxs(const std::byte* blk, float* y) {
    const float d = f16_to_f32(load_u16(blk));
    const auto* qs =
        reinterpret_cast<const std::uint8_t*>(blk + 2);
    const std::byte* sas = blk + 2 + 64;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        std::uint32_t aux32;
        std::memcpy(&aux32, sas + 4 * ib32, 4);
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const std::uint8_t signs =
                ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            const auto* grid1 =
                reinterpret_cast<const std::uint8_t*>(
                    iq3xxs_grid + qs[2 * l]);
            const auto* grid2 =
                reinterpret_cast<const std::uint8_t*>(
                    iq3xxs_grid + qs[2 * l + 1]);
            for (int j = 0; j < 4; ++j) {
                y[j] = db * grid1[j] *
                       ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                y[j + 4] = db * grid2[j] *
                           ((signs & kmask_iq2xs[j + 4])
                                ? -1.0f
                                : 1.0f);
            }
            y += 8;
        }
        qs += 8;
    }
}

/** IQ1_S (50 bytes): d | qs[32] | qh as 8 u16. */
void dequant_block_iq1_s(const std::byte* blk, float* y) {
    constexpr float kIq1sDelta = 0.125f;
    const float d = f16_to_f32(load_u16(blk));
    const auto* qs =
        reinterpret_cast<const std::uint8_t*>(blk + 2);
    for (int ib = 0; ib < 8; ++ib) {
        std::uint16_t qh;
        std::memcpy(&qh, blk + 34 + 2 * ib, 2);
        const float dl = d * (2 * ((qh >> 12) & 7) + 1);
        const float delta =
            (qh & 0x8000) ? -kIq1sDelta : kIq1sDelta;
        for (int l = 0; l < 4; ++l) {
            const auto* grid =
                reinterpret_cast<const std::int8_t*>(
                    iq1s_grid +
                    (qs[l] | (((qh >> (3 * l)) & 7) << 8)));
            for (int j = 0; j < 8; ++j) {
                y[j] = dl * (grid[j] + delta);
            }
            y += 8;
        }
        qs += 4;
    }
}

/** IQ4_XS (136 bytes): d | scales_h u16 | scales_l[4] | qs[128]. */
void dequant_block_iq4_xs(const std::byte* blk, float* y) {
    const float d = f16_to_f32(load_u16(blk));
    std::uint16_t scales_h;
    std::memcpy(&scales_h, blk + 2, 2);
    const auto* scales_l =
        reinterpret_cast<const std::uint8_t*>(blk + 4);
    const auto* qs =
        reinterpret_cast<const std::uint8_t*>(blk + 8);
    for (int ib = 0; ib < 8; ++ib) {
        const int ls =
            ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f) |
            (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (ls - 32);
        for (int j = 0; j < 16; ++j) {
            y[j] = dl * kvalues_iq4nl[qs[j] & 0x0f];
            y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
        }
        y += 32;
        qs += 16;
    }
}

/** IQ1_M (56 bytes): qs[32] | qh[16] | scales[8]. No d field: the f16
 *  scale is assembled from the top nibble of each of the 4 scale u16s;
 *  each 32-group has two 3-bit sub-scales. Grid index = qs byte + 3
 *  high bits from qh; value dl*(grid[j] + +/-0.125). Reuses iq1s_grid.
 *  Ports ggml dequantize_row_iq1_m. */
void dequant_block_iq1_m(const std::byte* blk, float* y) {
    constexpr float kDelta = 0.125f;  // IQ1S_DELTA
    const auto* qs = reinterpret_cast<const std::uint8_t*>(blk);
    const auto* qh = reinterpret_cast<const std::uint8_t*>(blk + 32);
    std::uint16_t sc[4];
    for (int k = 0; k < 4; ++k) {
        sc[k] = load_u16(blk + 48 + 2 * k);
    }
    const std::uint16_t su16 =
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
        ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = f16_to_f32(su16);
    const std::uint8_t* qsp = qs;
    const std::uint8_t* qhp = qh;
    for (int ib = 0; ib < 8; ++ib) {
        const float dl1 =
            d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
        const float dl2 =
            d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
        std::uint16_t idx[4];
        idx[0] = qsp[0] | ((qhp[0] << 8) & 0x700);
        idx[1] = qsp[1] | ((qhp[0] << 4) & 0x700);
        idx[2] = qsp[2] | ((qhp[1] << 8) & 0x700);
        idx[3] = qsp[3] | ((qhp[1] << 4) & 0x700);
        float delta[4];
        delta[0] = (qhp[0] & 0x08) ? -kDelta : kDelta;
        delta[1] = (qhp[0] & 0x80) ? -kDelta : kDelta;
        delta[2] = (qhp[1] & 0x08) ? -kDelta : kDelta;
        delta[3] = (qhp[1] & 0x80) ? -kDelta : kDelta;
        for (int l = 0; l < 2; ++l) {
            const auto* grid = reinterpret_cast<const std::int8_t*>(
                iq1s_grid + idx[l]);
            for (int j = 0; j < 8; ++j) {
                *y++ = dl1 * (grid[j] + delta[l]);
            }
        }
        for (int l = 2; l < 4; ++l) {
            const auto* grid = reinterpret_cast<const std::int8_t*>(
                iq1s_grid + idx[l]);
            for (int j = 0; j < 8; ++j) {
                *y++ = dl2 * (grid[j] + delta[l]);
            }
        }
        qsp += 4;
        qhp += 2;
    }
}

/** IQ2_XS (74 bytes): d | qs[32] u16 | scales[8]. Each 32-group uses
 *  scales[ib32] (two 4-bit sub-scales) and 4 grid entries: qs low 9
 *  bits index iq2xs_grid (8 u8 each), high 7 bits index ksigns.
 *  value = db[l/2] * grid[j] * sign. Ports ggml dequantize_row_iq2_xs. */
void dequant_block_iq2_xs(const std::byte* blk, float* y) {
    const float d = f16_to_f32(load_u16(blk));
    const auto* scales =
        reinterpret_cast<const std::uint8_t*>(blk + 66);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float db0 =
            d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const std::uint16_t q =
                load_u16(blk + 2 + 2 * (4 * ib32 + l));
            const auto* grid = reinterpret_cast<const std::uint8_t*>(
                iq2xs_grid + (q & 511));
            const std::uint8_t signs = ksigns_iq2xs[q >> 9];
            const float db = (l / 2 == 0) ? db0 : db1;
            for (int j = 0; j < 8; ++j) {
                y[j] = db * grid[j] *
                       ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
            }
            y += 8;
        }
    }
}

/** IQ2_S (82 bytes): d | qs[64] | qh[8] | scales[8]. The 64 qs bytes
 *  split into 32 grid-index low bytes then 32 sign bytes; qh supplies
 *  2 high index bits per element (10-bit iq2s_grid index). Signs are
 *  the raw sign bytes (bit j), not a ksigns lookup. Ports ggml
 *  dequantize_row_iq2_s. */
void dequant_block_iq2_s(const std::byte* blk, float* y) {
    const float d = f16_to_f32(load_u16(blk));
    const auto* qs = reinterpret_cast<const std::uint8_t*>(blk + 2);
    const auto* qh = reinterpret_cast<const std::uint8_t*>(blk + 66);
    const auto* scales =
        reinterpret_cast<const std::uint8_t*>(blk + 74);
    const std::uint8_t* qsp = qs;
    const std::uint8_t* signs = qs + 32;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float db0 =
            d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const float dl = (l / 2 == 0) ? db0 : db1;
            const auto* grid = reinterpret_cast<const std::uint8_t*>(
                iq2s_grid +
                (qsp[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
            for (int j = 0; j < 8; ++j) {
                y[j] = dl * grid[j] *
                       ((signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f);
            }
            y += 8;
        }
        qsp += 4;
        signs += 4;
    }
}

/** IQ3_S (110 bytes): d | qs[64] | qh[8] | signs[32] | scales[4].
 *  Processed two 32-groups at a time: each element's 9-bit iq3s_grid
 *  index is qs byte + one high bit from qh; grid entry is 4 u8; signs
 *  are raw bytes; scale = d*(1+2*sub). Ports dequantize_row_iq3_s. */
void dequant_block_iq3_s(const std::byte* blk, float* y) {
    const float d = f16_to_f32(load_u16(blk));
    const auto* qs = reinterpret_cast<const std::uint8_t*>(blk + 2);
    const auto* qh = reinterpret_cast<const std::uint8_t*>(blk + 66);
    const auto* signs =
        reinterpret_cast<const std::uint8_t*>(blk + 74);
    const auto* scales =
        reinterpret_cast<const std::uint8_t*>(blk + 106);
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        const float db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf));
        const float db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4));
        for (int l = 0; l < 4; ++l) {
            const auto* g1 = reinterpret_cast<const std::uint8_t*>(
                iq3s_grid + (qs[2 * l] | ((qh[0] << (8 - 2 * l)) &
                                          256)));
            const auto* g2 = reinterpret_cast<const std::uint8_t*>(
                iq3s_grid + (qs[2 * l + 1] |
                             ((qh[0] << (7 - 2 * l)) & 256)));
            for (int j = 0; j < 4; ++j) {
                y[j] = db1 * g1[j] *
                       ((signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                y[j + 4] = db1 * g2[j] *
                           ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f
                                                            : 1.0f);
            }
            y += 8;
        }
        qs += 8;
        signs += 4;
        for (int l = 0; l < 4; ++l) {
            const auto* g1 = reinterpret_cast<const std::uint8_t*>(
                iq3s_grid + (qs[2 * l] | ((qh[1] << (8 - 2 * l)) &
                                          256)));
            const auto* g2 = reinterpret_cast<const std::uint8_t*>(
                iq3s_grid + (qs[2 * l + 1] |
                             ((qh[1] << (7 - 2 * l)) & 256)));
            for (int j = 0; j < 4; ++j) {
                y[j] = db2 * g1[j] *
                       ((signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                y[j + 4] = db2 * g2[j] *
                           ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f
                                                            : 1.0f);
            }
            y += 8;
        }
        qh += 2;
        qs += 8;
        signs += 4;
    }
}

/** TQ1_0 (54 bytes): qs[48] | qh[4] | d. Ternary (BitNet b1.58): each
 *  byte packs 5 base-3 digits (qs) or 4 (qh); digit n is recovered as
 *  ((byte*pow3[n]*3)>>8) in {0,1,2}, value (digit-1)*d in {-1,0,1}*d.
 *  Ports ggml dequantize_row_tq1_0 exactly (uint8 wrap intended). */
void dequant_block_tq1_0(const std::byte* blk, float* y) {
    static constexpr std::uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    const auto* qs = reinterpret_cast<const std::uint8_t*>(blk);
    const auto* qh = reinterpret_cast<const std::uint8_t*>(blk + 48);
    const float d = f16_to_f32(load_u16(blk + 52));
    for (std::size_t j = 0; j < 32; j += 32) {  // first 32 qs bytes
        for (std::size_t n = 0; n < 5; ++n) {
            for (std::size_t m = 0; m < 32; ++m) {
                std::uint8_t q =
                    static_cast<std::uint8_t>(qs[j + m] * pow3[n]);
                std::int16_t xi =
                    (static_cast<std::uint16_t>(q) * 3) >> 8;
                *y++ = static_cast<float>(xi - 1) * d;
            }
        }
    }
    for (std::size_t j = 32; j < 48; j += 16) {  // trailing 16 bytes
        for (std::size_t n = 0; n < 5; ++n) {
            for (std::size_t m = 0; m < 16; ++m) {
                std::uint8_t q =
                    static_cast<std::uint8_t>(qs[j + m] * pow3[n]);
                std::int16_t xi =
                    (static_cast<std::uint16_t>(q) * 3) >> 8;
                *y++ = static_cast<float>(xi - 1) * d;
            }
        }
    }
    for (std::size_t n = 0; n < 4; ++n) {  // qh: 4 digits per byte
        for (std::size_t j = 0; j < 4; ++j) {
            std::uint8_t q =
                static_cast<std::uint8_t>(qh[j] * pow3[n]);
            std::int16_t xi = (static_cast<std::uint16_t>(q) * 3) >> 8;
            *y++ = static_cast<float>(xi - 1) * d;
        }
    }
}

/** TQ2_0 (66 bytes): qs[64] | d. Ternary at 2 bits/element: element
 *  (l,m) = (qs[j+m] >> 2l) & 3 in {0,1,2}, value (v-1)*d. Ports ggml
 *  dequantize_row_tq2_0 exactly. */
void dequant_block_tq2_0(const std::byte* blk, float* y) {
    const auto* qs = reinterpret_cast<const std::uint8_t*>(blk);
    const float d = f16_to_f32(load_u16(blk + 64));
    for (std::size_t j = 0; j < 64; j += 32) {
        for (std::size_t l = 0; l < 4; ++l) {
            for (std::size_t m = 0; m < 32; ++m) {
                std::int8_t q =
                    static_cast<std::int8_t>((qs[j + m] >> (l * 2)) & 3);
                *y++ = static_cast<float>(q - 1) * d;
            }
        }
    }
}

using BlockDequantFn = void (*)(const std::byte*, float*);

/** @returns {block_bytes, fn} for 256-elem super-block types. */
std::pair<std::size_t, BlockDequantFn> k_traits(
    gguf::TensorType t) {
    switch (t) {
        case gguf::TensorType::kQ2_K:
            return {84, &dequant_block_q2_k};
        case gguf::TensorType::kQ3_K:
            return {110, &dequant_block_q3_k};
        case gguf::TensorType::kQ4_K:
            return {144, &dequant_block_q4_k};
        case gguf::TensorType::kQ5_K:
            return {176, &dequant_block_q5_k};
        case gguf::TensorType::kQ6_K:
            return {210, &dequant_block_q6_k};
        case gguf::TensorType::kIQ2_XXS:
            return {66, &dequant_block_iq2_xxs};
        case gguf::TensorType::kIQ2_XS:
            return {74, &dequant_block_iq2_xs};
        case gguf::TensorType::kIQ2_S:
            return {82, &dequant_block_iq2_s};
        case gguf::TensorType::kIQ3_S:
            return {110, &dequant_block_iq3_s};
        case gguf::TensorType::kIQ3_XXS:
            return {98, &dequant_block_iq3_xxs};
        case gguf::TensorType::kIQ1_S:
            return {50, &dequant_block_iq1_s};
        case gguf::TensorType::kIQ1_M:
            return {56, &dequant_block_iq1_m};
        case gguf::TensorType::kIQ4_XS:
            return {136, &dequant_block_iq4_xs};
        case gguf::TensorType::kTQ1_0:
            return {54, &dequant_block_tq1_0};
        case gguf::TensorType::kTQ2_0:
            return {66, &dequant_block_tq2_0};
        default:
            return {0, nullptr};
    }
}

/** IQ4_NL (18B/32 elems): d | qs[16]. A 32-element block (not a
 *  super-block): low nibble -> first 16 lanes, high nibble -> next
 *  16, each mapped through the kvalues_iq4nl codebook. Q4_0-style row
 *  path. Matches ggml dequantize_row_iq4_nl. */
float dot_iq4_nl(const std::byte* row, std::span<const float> x) {
    float acc = 0.0f;
    const std::size_t nb = x.size() / 32;
    for (std::size_t b = 0; b < nb; ++b) {
        const std::byte* blk = row + b * kIQ4_NLBlockBytes;
        const float d = f16_to_f32(load_u16(blk));
        const auto* q =
            reinterpret_cast<const std::uint8_t*>(blk + 2);
        float s = 0.0f;
        for (std::size_t i = 0; i < 16; ++i) {
            const float lo = kvalues_iq4nl[q[i] & 0x0f];
            const float hi = kvalues_iq4nl[q[i] >> 4];
            s += lo * x[b * 32 + i] + hi * x[b * 32 + i + 16];
        }
        acc += d * s;
    }
    return acc;
}

// ---- Q8_K activation quantization (ggml parity). ggml's k-quant/IQ
// matvec quantizes the ACTIVATION vector to Q8_K (256-block: f32 scale
// + int8 quants + per-16 sums) and does an integer dot with the
// quantized weight. Matching it makes locus bit-exact with llama.cpp
// on quantized matvec (and enables integer-dot speed). ----

/** ggml's round-to-nearest (the 2^23 magic-add trick); matches the
 *  quantizer bit-for-bit, unlike std::lround (half-away-from-zero). */
int nearest_int(float fval) {
    float val = fval + 12582912.0f;
    std::int32_t i;
    std::memcpy(&i, &val, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}

/** One Q8_K block: 256 int8 quants, an f32 scale, and the sum of each
 *  group of 16 quants (used by the weight-min correction). */
struct BlockQ8K {
    float d;
    std::int8_t qs[256];
    std::int16_t bsums[16];
};

/** Quantizes n floats (n % 256 == 0) into Q8_K blocks. Ports ggml
 *  quantize_row_q8_K_ref exactly (iscale = -127/max). */
void quantize_row_q8_k(const float* x, BlockQ8K* y, std::size_t n) {
    const std::size_t nb = n / 256;
    for (std::size_t i = 0; i < nb; ++i) {
        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < 256; ++j) {
            const float ax = std::fabs(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        if (amax == 0.0f) {
            y[i].d = 0.0f;
            std::memset(y[i].qs, 0, 256);
            std::memset(y[i].bsums, 0, sizeof(y[i].bsums));
            x += 256;
            continue;
        }
        const float iscale = -127.0f / max;
        for (int j = 0; j < 256; ++j) {
            const int v = nearest_int(iscale * x[j]);
            y[i].qs[j] = static_cast<std::int8_t>(std::min(127, v));
        }
        for (int j = 0; j < 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j * 16 + ii];
            }
            y[i].bsums[j] = static_cast<std::int16_t>(sum);
        }
        y[i].d = 1.0f / iscale;
        x += 256;
    }
}

/** Q4_K (144B block) x Q8_K activation integer dot; ports ggml
 *  ggml_vec_dot_q4_K_q8_K_generic. `xq` holds nb Q8_K blocks. */
float vec_dot_q4_k_q8_k(const std::byte* row, const BlockQ8K* xq,
                        std::size_t nb) {
    constexpr std::uint32_t kmask1 = 0x3f3f3f3f;
    constexpr std::uint32_t kmask2 = 0x0f0f0f0f;
    constexpr std::uint32_t kmask3 = 0x03030303;
    std::uint32_t utmp[4];
    const auto* scales = reinterpret_cast<const std::uint8_t*>(&utmp[0]);
    const auto* mins = reinterpret_cast<const std::uint8_t*>(&utmp[2]);
    std::int8_t aux8[256];
    std::int16_t aux16[8];
    float sums[8] = {0};
    std::int32_t aux32[8];
    float sumf = 0.0f;
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 144;
        const float bd = f16_to_f32(load_u16(blk));
        const float bdmin = f16_to_f32(load_u16(blk + 2));
        const auto* q4 =
            reinterpret_cast<const std::uint8_t*>(blk + 16);
        const std::int8_t* q8 = xq[i].qs;
        std::memset(aux32, 0, sizeof(aux32));
        std::int8_t* a = aux8;
        for (int j = 0; j < 4; ++j) {
            for (int l = 0; l < 32; ++l) {
                a[l] = static_cast<std::int8_t>(q4[l] & 0xF);
            }
            a += 32;
            for (int l = 0; l < 32; ++l) {
                a[l] = static_cast<std::int8_t>(q4[l] >> 4);
            }
            a += 32;
            q4 += 32;
        }
        std::memcpy(utmp, blk + 4, 12);  // Q4_K scales[12]
        utmp[3] = ((utmp[2] >> 4) & kmask2) |
                  (((utmp[1] >> 6) & kmask3) << 4);
        const std::uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        int sumi = 0;
        for (int j = 0; j < 16; ++j) {
            sumi += xq[i].bsums[j] * mins[j / 2];
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < 8; ++j) {
            const std::int32_t scale = scales[is++];
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 8; ++l) {
                    aux16[l] = static_cast<std::int16_t>(q8[l] * a[l]);
                }
                for (int l = 0; l < 8; ++l) {
                    aux32[l] += scale * aux16[l];
                }
                q8 += 8;
                a += 8;
            }
        }
        const float d = bd * xq[i].d;
        for (int l = 0; l < 8; ++l) {
            sums[l] += d * static_cast<float>(aux32[l]);
        }
        sumf -= bdmin * xq[i].d * static_cast<float>(sumi);
    }
    for (int l = 0; l < 8; ++l) {
        sumf += sums[l];
    }
    return sumf;
}

/** Q6_K (210B) x Q8_K integer dot; ports ggml_vec_dot_q6_K_q8_K_
 *  generic. No min term. */
float vec_dot_q6_k_q8_k(const std::byte* row, const BlockQ8K* xq,
                        std::size_t nb) {
    std::int8_t aux8[256];
    std::int16_t aux16[8];
    float sums[8] = {0};
    std::int32_t aux32[8];
    float sumf = 0.0f;
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 210;
        const auto* ql = reinterpret_cast<const std::uint8_t*>(blk);
        const auto* qh =
            reinterpret_cast<const std::uint8_t*>(blk + 128);
        const auto* sc =
            reinterpret_cast<const std::int8_t*>(blk + 192);
        const float bd = f16_to_f32(load_u16(blk + 208));
        const std::int8_t* q8 = xq[i].qs;
        std::memset(aux32, 0, sizeof(aux32));
        std::int8_t* a = aux8;
        const std::uint8_t* q4 = ql;
        const std::uint8_t* qhp = qh;
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l + 0] = static_cast<std::int8_t>(
                    ((q4[l] & 0xF) | (((qhp[l] >> 0) & 3) << 4)) -
                    32);
                a[l + 32] = static_cast<std::int8_t>(
                    ((q4[l + 32] & 0xF) | (((qhp[l] >> 2) & 3) << 4)) -
                    32);
                a[l + 64] = static_cast<std::int8_t>(
                    ((q4[l] >> 4) | (((qhp[l] >> 4) & 3) << 4)) - 32);
                a[l + 96] = static_cast<std::int8_t>(
                    ((q4[l + 32] >> 4) | (((qhp[l] >> 6) & 3) << 4)) -
                    32);
            }
            a += 128;
            q4 += 64;
            qhp += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < 16; ++j) {
            const std::int32_t scale = sc[is++];
            for (int k = 0; k < 2; ++k) {
                for (int l = 0; l < 8; ++l) {
                    aux16[l] = static_cast<std::int16_t>(q8[l] * a[l]);
                }
                for (int l = 0; l < 8; ++l) {
                    aux32[l] += scale * aux16[l];
                }
                q8 += 8;
                a += 8;
            }
        }
        const float d = bd * xq[i].d;
        for (int l = 0; l < 8; ++l) {
            sums[l] += d * static_cast<float>(aux32[l]);
        }
    }
    for (int l = 0; l < 8; ++l) {
        sumf += sums[l];
    }
    return sumf;
}

/** Q5_K (176B) x Q8_K integer dot; ports ggml_vec_dot_q5_K_q8_K_
 *  generic (5th bit from qh, min via bsums). */
float vec_dot_q5_k_q8_k(const std::byte* row, const BlockQ8K* xq,
                        std::size_t nb) {
    constexpr std::uint32_t kmask1 = 0x3f3f3f3f;
    constexpr std::uint32_t kmask2 = 0x0f0f0f0f;
    constexpr std::uint32_t kmask3 = 0x03030303;
    std::uint32_t utmp[4];
    const auto* scales = reinterpret_cast<const std::uint8_t*>(&utmp[0]);
    const auto* mins = reinterpret_cast<const std::uint8_t*>(&utmp[2]);
    std::int8_t aux8[256];
    std::int16_t aux16[8];
    float sums[8] = {0};
    std::int32_t aux32[8];
    float sumf = 0.0f;
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 176;
        const float bd = f16_to_f32(load_u16(blk));
        const float bdmin = f16_to_f32(load_u16(blk + 2));
        const auto* hm =
            reinterpret_cast<const std::uint8_t*>(blk + 16);
        const auto* q4 =
            reinterpret_cast<const std::uint8_t*>(blk + 48);
        const std::int8_t* q8 = xq[i].qs;
        std::memset(aux32, 0, sizeof(aux32));
        std::int8_t* a = aux8;
        std::uint8_t m = 1;
        for (int j = 0; j < 4; ++j) {
            for (int l = 0; l < 32; ++l) {
                a[l] = static_cast<std::int8_t>(q4[l] & 0xF);
            }
            for (int l = 0; l < 32; ++l) {
                a[l] = static_cast<std::int8_t>(a[l] +
                                                ((hm[l] & m) ? 16 : 0));
            }
            a += 32;
            m <<= 1;
            for (int l = 0; l < 32; ++l) {
                a[l] = static_cast<std::int8_t>(q4[l] >> 4);
            }
            for (int l = 0; l < 32; ++l) {
                a[l] = static_cast<std::int8_t>(a[l] +
                                                ((hm[l] & m) ? 16 : 0));
            }
            a += 32;
            m <<= 1;
            q4 += 32;
        }
        std::memcpy(utmp, blk + 4, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) |
                  (((utmp[1] >> 6) & kmask3) << 4);
        const std::uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        int sumi = 0;
        for (int j = 0; j < 16; ++j) {
            sumi += xq[i].bsums[j] * mins[j / 2];
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < 8; ++j) {
            const std::int32_t scale = scales[is++];
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 8; ++l) {
                    aux16[l] = static_cast<std::int16_t>(q8[l] * a[l]);
                }
                for (int l = 0; l < 8; ++l) {
                    aux32[l] += scale * aux16[l];
                }
                q8 += 8;
                a += 8;
            }
        }
        const float d = bd * xq[i].d;
        for (int l = 0; l < 8; ++l) {
            sums[l] += d * static_cast<float>(aux32[l]);
        }
        sumf -= bdmin * xq[i].d * static_cast<float>(sumi);
    }
    for (int l = 0; l < 8; ++l) {
        sumf += sums[l];
    }
    return sumf;
}

/** Q2_K (84B) x Q8_K integer dot; ports ggml_vec_dot_q2_K_q8_K_
 *  generic (2-bit quants, 4-bit scale + 4-bit min per sub-block). */
float vec_dot_q2_k_q8_k(const std::byte* row, const BlockQ8K* xq,
                        std::size_t nb) {
    float sumf = 0.0f;
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 84;
        const auto* sc = reinterpret_cast<const std::uint8_t*>(blk);
        const auto* q2 =
            reinterpret_cast<const std::uint8_t*>(blk + 16);
        const float bd = f16_to_f32(load_u16(blk + 80));
        const float bdmin = f16_to_f32(load_u16(blk + 82));
        const std::int8_t* q8 = xq[i].qs;
        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += xq[i].bsums[j] * (sc[j] >> 4);
        }
        const float dall = xq[i].d * bd;
        const float dmin = xq[i].d * bdmin;
        int isum = 0, is = 0;
        for (int k = 0; k < 2; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                int d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l = 0; l < 16; ++l) {
                    isuml += q8[l] * ((q2[l] >> shift) & 3);
                }
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) {
                    isuml += q8[l] * ((q2[l] >> shift) & 3);
                }
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * static_cast<float>(isum) -
                dmin * static_cast<float>(summs);
    }
    return sumf;
}

/** Q3_K (110B) x Q8_K integer dot; ports ggml_vec_dot_q3_K_q8_K_
 *  generic (2-bit quants + hmask 3rd bit, 6-bit scales - 32). */
float vec_dot_q3_k_q8_k(const std::byte* row, const BlockQ8K* xq,
                        std::size_t nb) {
    constexpr std::uint32_t kmask1 = 0x03030303;
    constexpr std::uint32_t kmask2 = 0x0f0f0f0f;
    std::int8_t aux8[256];
    std::int16_t aux16[8];
    float sums[8] = {0};
    std::int32_t aux32[8];
    std::uint32_t auxs[4];
    const auto* scales = reinterpret_cast<const std::int8_t*>(auxs);
    float sumf = 0.0f;
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 110;
        const auto* hm = reinterpret_cast<const std::uint8_t*>(blk);
        const auto* q3 =
            reinterpret_cast<const std::uint8_t*>(blk + 32);
        const float bd = f16_to_f32(load_u16(blk + 108));
        const std::int8_t* q8 = xq[i].qs;
        std::memset(aux32, 0, sizeof(aux32));
        std::int8_t* a = aux8;
        std::uint8_t m = 1;
        const std::uint8_t* q3p = q3;
        for (int j = 0; j < 256; j += 128) {
            for (int sh = 0; sh < 8; sh += 2) {
                for (int l = 0; l < 32; ++l) {
                    a[l] = static_cast<std::int8_t>((q3p[l] >> sh) & 3);
                }
                for (int l = 0; l < 32; ++l) {
                    a[l] = static_cast<std::int8_t>(
                        a[l] - ((hm[l] & m) ? 0 : 4));
                }
                a += 32;
                m <<= 1;
            }
            q3p += 32;
        }
        std::memcpy(auxs, blk + 96, 12);
        const std::uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) |
                  (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) |
                  (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        a = aux8;
        for (int j = 0; j < 16; ++j) {
            for (int k = 0; k < 2; ++k) {
                for (int l = 0; l < 8; ++l) {
                    aux16[l] = static_cast<std::int16_t>(q8[l] * a[l]);
                }
                for (int l = 0; l < 8; ++l) {
                    aux32[l] += (scales[j] - 32) * aux16[l];
                }
                q8 += 8;
                a += 8;
            }
        }
        const float d = bd * xq[i].d;
        for (int l = 0; l < 8; ++l) {
            sums[l] += d * static_cast<float>(aux32[l]);
        }
    }
    for (int l = 0; l < 8; ++l) {
        sumf += sums[l];
    }
    return sumf;
}

/** TQ1_0 (54B) x Q8_K integer dot; ports ggml_vec_dot_tq1_0_q8_K_
 *  generic. The activation is consumed in the ternary fill order. */
float vec_dot_tq1_0_q8_k(const std::byte* row, const BlockQ8K* xq,
                         std::size_t nb) {
    static constexpr std::uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    float sumf = 0.0f;
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 54;
        const auto* qs = reinterpret_cast<const std::uint8_t*>(blk);
        const auto* qh =
            reinterpret_cast<const std::uint8_t*>(blk + 48);
        const float bd = f16_to_f32(load_u16(blk + 52));
        const std::int8_t* q8 = xq[i].qs;
        int sum = 0;
        for (int j = 0; j < 32; j += 32) {
            for (int l = 0; l < 5; ++l) {
                for (int m = 0; m < 32; ++m) {
                    const std::uint8_t q =
                        static_cast<std::uint8_t>(qs[j + m] * pow3[l]);
                    const std::uint16_t xi =
                        (static_cast<std::uint16_t>(q) * 3) >> 8;
                    sum += (xi - 1) * q8[j * 5 + l * 32 + m];
                }
            }
        }
        for (int j = 32; j < 48; j += 16) {
            for (int l = 0; l < 5; ++l) {
                for (int m = 0; m < 16; ++m) {
                    const std::uint8_t q =
                        static_cast<std::uint8_t>(qs[j + m] * pow3[l]);
                    const std::uint16_t xi =
                        (static_cast<std::uint16_t>(q) * 3) >> 8;
                    sum += (xi - 1) * q8[j * 5 + l * 16 + m];
                }
            }
        }
        for (int l = 0; l < 4; ++l) {
            for (int j = 0; j < 4; ++j) {
                const std::uint8_t q =
                    static_cast<std::uint8_t>(qh[j] * pow3[l]);
                const std::uint16_t xi =
                    (static_cast<std::uint16_t>(q) * 3) >> 8;
                sum += (xi - 1) * q8[48 * 5 + l * 4 + j];
            }
        }
        sumf += static_cast<float>(sum) * (bd * xq[i].d);
    }
    return sumf;
}

/** TQ2_0 (66B) x Q8_K integer dot; ports ggml_vec_dot_tq2_0_q8_K_
 *  generic. */
float vec_dot_tq2_0_q8_k(const std::byte* row, const BlockQ8K* xq,
                         std::size_t nb) {
    float sumf = 0.0f;
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 66;
        const auto* qs = reinterpret_cast<const std::uint8_t*>(blk);
        const float bd = f16_to_f32(load_u16(blk + 64));
        const std::int8_t* q8 = xq[i].qs;
        std::int32_t sumi = 0;
        for (int j = 0; j < 64; j += 32) {
            for (int l = 0; l < 4; ++l) {
                for (int k = 0; k < 32; ++k) {
                    sumi += q8[j * 4 + l * 32 + k] *
                            (((qs[j + k] >> (l * 2)) & 3) - 1);
                }
            }
        }
        sumf += static_cast<float>(sumi) * (xq[i].d * bd);
    }
    return sumf;
}

float dot_k_quant(const Mat& w, const std::byte* row,
                  std::span<const float> x) {
    const auto [bytes, fn] = k_traits(w.type);
    float acc = 0.0f;
    float tmp[256];
    for (std::size_t b = 0; b < x.size() / 256; ++b) {
        fn(row + b * bytes, tmp);
        for (std::size_t i = 0; i < 256; ++i) {
            acc += tmp[i] * x[b * 256 + i];
        }
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
        case gguf::TensorType::kQ5_0:
            return w.cols / 32ull * kQ5_0BlockBytes;
        case gguf::TensorType::kIQ4_NL:
            return w.cols / 32ull * kIQ4_NLBlockBytes;
        case gguf::TensorType::kQ2_K:
        case gguf::TensorType::kQ3_K:
        case gguf::TensorType::kQ4_K:
        case gguf::TensorType::kQ5_K:
        case gguf::TensorType::kQ6_K:
        case gguf::TensorType::kIQ2_XXS:
        case gguf::TensorType::kIQ2_XS:
        case gguf::TensorType::kIQ2_S:
        case gguf::TensorType::kIQ3_XXS:
        case gguf::TensorType::kIQ3_S:
        case gguf::TensorType::kIQ1_S:
        case gguf::TensorType::kIQ1_M:
        case gguf::TensorType::kIQ4_XS:
        case gguf::TensorType::kTQ1_0:
        case gguf::TensorType::kTQ2_0:
            return w.cols / 256ull * k_traits(w.type).first;
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

void layernorm(std::span<const float> x, std::span<const float> w,
               float eps, std::span<float> out) {
    assert(x.size() == w.size() && x.size() == out.size());
    const float n = static_cast<float>(x.size());
    float mean = 0.0f;
    for (float v : x) {
        mean += v;
    }
    mean /= n;
    float var = 0.0f;
    for (float v : x) {
        const float d = v - mean;
        var += d * d;
    }
    const float scale = 1.0f / std::sqrt(var / n + eps);
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = (x[i] - mean) * scale * w[i];
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

std::size_t mat_row_bytes(const Mat& w) { return row_bytes(w); }

void matvec_t(const Mat& w, std::span<const float> x,
              std::span<float> out) {
    assert(x.size() == w.rows && out.size() == w.cols);
    std::fill(out.begin(), out.end(), 0.0f);
    static thread_local std::vector<float> tmp;
    tmp.resize(w.cols);
    for (std::uint32_t r = 0; r < w.rows; ++r) {
        const float xr = x[r];
        if (xr == 0.0f) {
            continue;
        }
        dequant_row(w, r, tmp);
        for (std::uint32_t c = 0; c < w.cols; ++c) {
            out[c] += xr * tmp[c];
        }
    }
}

namespace {

/** ggml's YARN correction dim for one beta. */
float yarn_corr_dim(std::uint32_t n_dims,
                    std::uint32_t n_ctx_orig, float beta,
                    float base) {
    return static_cast<float>(n_dims) *
           std::log(static_cast<float>(n_ctx_orig) /
                    (beta * 2.0f *
                     3.14159265358979323846f)) /
           (2.0f * std::log(base));
}

}  // namespace

namespace {

/** YARN-corrected rotation angle for pair index i (of half). */
float yarn_theta(std::uint32_t i, std::uint32_t head_dim,
                 std::uint32_t pos, float freq_base,
                 const Yarn& yarn, float corr_lo,
                 float corr_hi) {
    const float theta_extrap =
        static_cast<float>(pos) *
        std::pow(freq_base, -2.0f * static_cast<float>(i) /
                                static_cast<float>(head_dim));
    if (yarn.freq_scale == 1.0f || yarn.n_ctx_orig == 0) {
        return theta_extrap;
    }
    // Interpolate above corr_hi, extrapolate below corr_lo,
    // ramp in between (ggml rope_yarn with ext_factor = 1).
    const float y = (static_cast<float>(i) - corr_lo) /
                    std::max(0.001f, corr_hi - corr_lo);
    const float ramp =
        1.0f - std::min(1.0f, std::max(0.0f, y));
    const float theta_interp = yarn.freq_scale * theta_extrap;
    return theta_interp * (1.0f - ramp) + theta_extrap * ramp;
}

void yarn_corr_range_impl(std::uint32_t head_dim, float freq_base,
                          const Yarn& yarn, float& lo,
                          float& hi) {
    lo = 0.0f;
    hi = 0.0f;
    if (yarn.freq_scale == 1.0f || yarn.n_ctx_orig == 0) {
        return;
    }
    lo = std::max(
        0.0f, std::floor(yarn_corr_dim(head_dim,
                                       yarn.n_ctx_orig,
                                       yarn.beta_fast,
                                       freq_base)));
    hi = std::min(
        static_cast<float>(head_dim) - 1.0f,
        std::ceil(yarn_corr_dim(head_dim, yarn.n_ctx_orig,
                                yarn.beta_slow, freq_base)));
}

}  // namespace

void rope_neox_yarn(std::span<float> x, std::uint32_t n_heads,
                    std::uint32_t head_dim, std::uint32_t pos,
                    float freq_base, const Yarn& yarn) {
    assert(x.size() ==
           static_cast<std::size_t>(n_heads) * head_dim);
    const std::uint32_t half = head_dim / 2;
    float corr_lo, corr_hi;
    yarn_corr_range_impl(head_dim, freq_base, yarn, corr_lo, corr_hi);
    for (std::uint32_t h = 0; h < n_heads; ++h) {
        float* v =
            x.data() + static_cast<std::size_t>(h) * head_dim;
        for (std::uint32_t i = 0; i < half; ++i) {
            const float theta =
                yarn_theta(i, head_dim, pos, freq_base, yarn,
                           corr_lo, corr_hi);
            const float c = std::cos(theta) * yarn.mscale;
            const float s = std::sin(theta) * yarn.mscale;
            const float x0 = v[i];
            const float x1 = v[i + half];
            v[i] = x0 * c - x1 * s;
            v[i + half] = x0 * s + x1 * c;
        }
    }
}

void yarn_corr_range(std::uint32_t head_dim, float freq_base,
                     const Yarn& yarn, float& lo, float& hi) {
    yarn_corr_range_impl(head_dim, freq_base, yarn, lo, hi);
}

void rope_norm_yarn(std::span<float> x, std::uint32_t n_heads,
                    std::uint32_t head_dim, std::uint32_t pos,
                    float freq_base, const Yarn& yarn) {
    assert(x.size() ==
           static_cast<std::size_t>(n_heads) * head_dim);
    float corr_lo, corr_hi;
    yarn_corr_range_impl(head_dim, freq_base, yarn, corr_lo, corr_hi);
    for (std::uint32_t h = 0; h < n_heads; ++h) {
        float* v =
            x.data() + static_cast<std::size_t>(h) * head_dim;
        for (std::uint32_t i = 0; i < head_dim / 2; ++i) {
            const float theta =
                yarn_theta(i, head_dim, pos, freq_base, yarn,
                           corr_lo, corr_hi);
            const float c = std::cos(theta) * yarn.mscale;
            const float s = std::sin(theta) * yarn.mscale;
            const float x0 = v[2 * i];
            const float x1 = v[2 * i + 1];
            v[2 * i] = x0 * c - x1 * s;
            v[2 * i + 1] = x0 * s + x1 * c;
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
            case gguf::TensorType::kQ4_0:
                out[r] = dot_q4_0(row, x);
                break;
            case gguf::TensorType::kQ5_0:
                out[r] = dot_q5_0(row, x);
                break;
            case gguf::TensorType::kIQ4_NL:
                out[r] = dot_iq4_nl(row, x);
                break;
            default:
                out[r] = dot_k_quant(w, row, x);
                break;
        }
    }
}

void matvec_q8k(const Mat& w, std::span<const float> x,
                std::span<float> out) {
    assert(x.size() == w.cols && out.size() == w.rows);
    using T = gguf::TensorType;
    const bool ported = (w.type == T::kQ2_K || w.type == T::kQ3_K ||
                         w.type == T::kQ4_K || w.type == T::kQ5_K ||
                         w.type == T::kQ6_K || w.type == T::kTQ1_0 ||
                         w.type == T::kTQ2_0) &&
                        x.size() % 256 == 0;
    if (!ported) {  // types without a Q8_K dot yet: keep the f32 path
        matvec(w, x, out);
        return;
    }
    const std::size_t nb = x.size() / 256;
    std::vector<BlockQ8K> xq(nb);
    quantize_row_q8_k(x.data(), xq.data(), x.size());
    const std::size_t stride = row_bytes(w);
    for (std::uint32_t r = 0; r < w.rows; ++r) {
        const std::byte* row = w.data + r * stride;
        switch (w.type) {
            case T::kQ2_K:
                out[r] = vec_dot_q2_k_q8_k(row, xq.data(), nb);
                break;
            case T::kQ3_K:
                out[r] = vec_dot_q3_k_q8_k(row, xq.data(), nb);
                break;
            case T::kQ4_K:
                out[r] = vec_dot_q4_k_q8_k(row, xq.data(), nb);
                break;
            case T::kQ5_K:
                out[r] = vec_dot_q5_k_q8_k(row, xq.data(), nb);
                break;
            case T::kQ6_K:
                out[r] = vec_dot_q6_k_q8_k(row, xq.data(), nb);
                break;
            case T::kTQ1_0:
                out[r] = vec_dot_tq1_0_q8_k(row, xq.data(), nb);
                break;
            case T::kTQ2_0:
                out[r] = vec_dot_tq2_0_q8_k(row, xq.data(), nb);
                break;
            default:
                break;
        }
    }
}

void matvec_batch_scalar(const Mat& w, std::span<const float> x_batch,
                         std::span<float> out_batch,
                         std::uint32_t n) {
    assert(x_batch.size() ==
               static_cast<std::size_t>(n) * w.cols &&
           out_batch.size() ==
               static_cast<std::size_t>(w.rows) * n);
    const std::size_t stride = row_bytes(w);
    const std::size_t xc = w.cols;
    // Row-major output (out[r*n + t]) and the same per-row dot
    // dispatch matvec() uses, so each result is bit-for-bit a
    // matvec() of that row against token t. The reference re-reads
    // the row per token; the SIMD kernels amortize that read.
    for (std::uint32_t r = 0; r < w.rows; ++r) {
        const std::byte* row = w.data + r * stride;
        float* o = out_batch.data() + static_cast<std::size_t>(r) * n;
        for (std::uint32_t t = 0; t < n; ++t) {
            const std::span<const float> x =
                x_batch.subspan(static_cast<std::size_t>(t) * xc, xc);
            switch (w.type) {
                case gguf::TensorType::kF32:
                    o[t] = dot_f32(row, x);
                    break;
                case gguf::TensorType::kF16:
                    o[t] = dot_f16(row, x);
                    break;
                case gguf::TensorType::kQ8_0:
                    o[t] = dot_q8_0(row, x);
                    break;
                case gguf::TensorType::kQ4_0:
                    o[t] = dot_q4_0(row, x);
                    break;
                case gguf::TensorType::kQ5_0:
                    o[t] = dot_q5_0(row, x);
                    break;
                case gguf::TensorType::kIQ4_NL:
                    o[t] = dot_iq4_nl(row, x);
                    break;
                default:
                    o[t] = dot_k_quant(w, row, x);
                    break;
            }
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
        case gguf::TensorType::kQ4_0: {
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
        case gguf::TensorType::kQ5_0: {
            for (std::uint32_t b = 0; b < w.cols / 32; ++b) {
                dequant_block_q5_0(p + b * kQ5_0BlockBytes,
                                   out.data() + b * 32);
            }
            break;
        }
        case gguf::TensorType::kIQ4_NL: {
            for (std::uint32_t b = 0; b < w.cols / 32; ++b) {
                const std::byte* blk = p + b * kIQ4_NLBlockBytes;
                const float d = f16_to_f32(load_u16(blk));
                const auto* q =
                    reinterpret_cast<const std::uint8_t*>(blk + 2);
                for (std::uint32_t i = 0; i < 16; ++i) {
                    out[b * 32 + i] = d * kvalues_iq4nl[q[i] & 0x0f];
                    out[b * 32 + i + 16] =
                        d * kvalues_iq4nl[q[i] >> 4];
                }
            }
            break;
        }
        default: {
            const auto [bytes, fn] = k_traits(w.type);
            for (std::uint32_t b = 0; b < w.cols / 256; ++b) {
                fn(p + b * bytes, out.data() + b * 256);
            }
            break;
        }
    }
}

}  // namespace locus::backend
