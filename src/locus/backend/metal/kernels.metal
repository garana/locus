// The Metal backend's kernels (i#865). Compiled from this source when
// the context opens (no prebuilt metallib): CMake embeds the file into
// the binary as a string.

#include <metal_stdlib>
using namespace metal;

struct AddParams {
    uint n;
};

// out[i] = a[i] + b[i] for i < n: the smoke-test kernel.
kernel void add_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant AddParams& p [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        out[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------------------
// Matvec (i#866): out[out_off + r] (+)= scale * dot(row r of W, x).
//
// A threadgroup of 128 threads (4 simdgroups) takes 8 rows, 16 lanes
// per row (two rows per simdgroup); lanes stride over the row's units
// and the 16 partial sums are added with simd shuffles within each
// half of the simdgroup. W starts w_off bytes into its buffer, a multiple
// of 8 for Q4_K (8-byte loads) and 16 for F32 (float4); Q8_0 and Q6_K
// read bytes. x_off is a multiple of 4 (float4 loads).

struct MvParams {
    ulong w_off;    // bytes into the weight buffer
    ulong out_off;  // floats into the output buffer
    ulong expert_bytes;  // routed: w_off + route[route_slot] * this
    uint rows;
    uint cols;
    uint accumulate;
    uint x_off;
    float scale;
    uint route_slot;  // kNoRoute: w_off as is
    // Several slots in one dispatch (MoE's k experts, i#868): slot s
    // takes threadgroups [s, s + 1) * groups-a-slot, the expert at
    // route[route_slot + s], x at + s * x_slot_stride and out at
    // + s * out_slot_stride. 1 slot otherwise.
    uint n_slots;
    uint x_slot_stride;
    uint out_slot_stride;
    uint slot_div;  // x of slot s is at + (s / slot_div) * x_slot_stride
};

constant constexpr uint kNoRoute = 0xFFFFFFFFu;

// The weight's byte offset: w_off, plus a routed expert's (MoE on the
// GPU, i#868: the expert id read from the route buffer at buffer 7).
inline ulong mv_w_off(constant MvParams& p, device const uint* route,
                      uint slot) {
    return p.route_slot == kNoRoute
               ? p.w_off
               : p.w_off + ulong(route[p.route_slot + slot]) * p.expert_bytes;
}

constant constexpr uint kMvLanes = 16;
constant constexpr uint kMvRows = 8;

// The sum of v over the 16 lanes of this half of the simdgroup.
inline float half_simd_sum(float v) {
    v += simd_shuffle_xor(v, 8);
    v += simd_shuffle_xor(v, 4);
    v += simd_shuffle_xor(v, 2);
    v += simd_shuffle_xor(v, 1);
    return v;
}

inline void mv_store(device float* out, constant MvParams& p, uint slot,
                     uint r, uint lane, float acc) {
    const float s = half_simd_sum(acc);
    if (lane == 0 && r < p.rows) {
        device float* o =
            out + p.out_off + ulong(slot) * p.out_slot_stride + r;
        const float v = p.scale * s;
        *o = p.accumulate != 0 ? *o + v : v;
    }
}

inline float hsum(float4 v) { return v.x + v.y + v.z + v.w; }

inline float f16_at(device const uchar* b) {
    return float(as_type<half>(ushort(b[0] | (b[1] << 8))));
}

// F32: units of 4 columns.
kernel void mv_f32(device const uchar* w [[buffer(0)]],
                   device const float* xb [[buffer(1)]],
                   device float* out [[buffer(2)]],
                   constant MvParams& p [[buffer(3)]],
                   device const uint* route [[buffer(7)]],
                   uint tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid % kMvLanes;
    const uint gps = (p.rows + kMvRows - 1) / kMvRows;  // groups a slot
    const uint slot = tg / gps;
    const uint r = tg % gps * kMvRows + tid / kMvLanes;
    const ulong wo = mv_w_off(p, route, slot);  // the weight's bytes
    device const float* xs =
        xb + p.x_off + ulong(slot / p.slot_div) * p.x_slot_stride;
    float acc = 0.0f;
    if (r < p.rows) {
        device const float4* row =
            (device const float4*)(w + wo +
                                      (ulong)r * p.cols * 4);
        device const float4* x = (device const float4*)(xs);
        for (uint u = lane; u < p.cols / 4; u += kMvLanes) {
            acc += dot(row[u], x[u]);
        }
    }
    mv_store(out, p, slot, r, lane, acc);
}

// Q8_0: 34-byte blocks (f16 d, 32 int8). Units are blocks; the block
// stride is not a multiple of 4, so the quants load as bytes.
kernel void mv_q8_0(device const uchar* w [[buffer(0)]],
                    device const float* xb [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant MvParams& p [[buffer(3)]],
                    device const uint* route [[buffer(7)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid % kMvLanes;
    const uint gps = (p.rows + kMvRows - 1) / kMvRows;  // groups a slot
    const uint slot = tg / gps;
    const uint r = tg % gps * kMvRows + tid / kMvLanes;
    const ulong wo = mv_w_off(p, route, slot);  // the weight's bytes
    device const float* xs =
        xb + p.x_off + ulong(slot / p.slot_div) * p.x_slot_stride;
    float acc = 0.0f;
    if (r < p.rows) {
        const uint nb = p.cols / 32;
        device const uchar* row = w + wo + (ulong)r * nb * 34;
        device const float* x = xs;
        for (uint b = lane; b < nb; b += kMvLanes) {
            device const uchar* blk = row + b * 34;
            device const char* q = (device const char*)(blk + 2);
            device const float4* xv = (device const float4*)(x + b * 32);
            float s = 0.0f;
            for (uint k = 0; k < 8; ++k) {
                const float4 qv(q[4 * k], q[4 * k + 1], q[4 * k + 2],
                                q[4 * k + 3]);
                s += dot(qv, xv[k]);
            }
            acc += f16_at(blk) * s;
        }
    }
    mv_store(out, p, slot, r, lane, acc);
}

// F16: units of 4 columns (8-byte loads: w_off a multiple of 8).
kernel void mv_f16(device const uchar* w [[buffer(0)]],
                   device const float* xb [[buffer(1)]],
                   device float* out [[buffer(2)]],
                   constant MvParams& p [[buffer(3)]],
                   device const uint* route [[buffer(7)]],
                   uint tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid % kMvLanes;
    const uint gps = (p.rows + kMvRows - 1) / kMvRows;  // groups a slot
    const uint slot = tg / gps;
    const uint r = tg % gps * kMvRows + tid / kMvLanes;
    const ulong wo = mv_w_off(p, route, slot);  // the weight's bytes
    device const float* xs =
        xb + p.x_off + ulong(slot / p.slot_div) * p.x_slot_stride;
    float acc = 0.0f;
    if (r < p.rows) {
        device const half4* row =
            (device const half4*)(w + wo +
                                      (ulong)r * p.cols * 2);
        device const float4* x = (device const float4*)(xs);
        for (uint u = lane; u < p.cols / 4; u += kMvLanes) {
            acc += dot(float4(row[u]), x[u]);
        }
    }
    mv_store(out, p, slot, r, lane, acc);
}

// BF16: units of 4 columns (8-byte loads), widened by a shift.
kernel void mv_bf16(device const uchar* w [[buffer(0)]],
                    device const float* xb [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant MvParams& p [[buffer(3)]],
                    device const uint* route [[buffer(7)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid % kMvLanes;
    const uint gps = (p.rows + kMvRows - 1) / kMvRows;  // groups a slot
    const uint slot = tg / gps;
    const uint r = tg % gps * kMvRows + tid / kMvLanes;
    const ulong wo = mv_w_off(p, route, slot);  // the weight's bytes
    device const float* xs =
        xb + p.x_off + ulong(slot / p.slot_div) * p.x_slot_stride;
    float acc = 0.0f;
    if (r < p.rows) {
        device const ushort4* row =
            (device const ushort4*)(w + wo +
                                      (ulong)r * p.cols * 2);
        device const float4* x = (device const float4*)(xs);
        for (uint u = lane; u < p.cols / 4; u += kMvLanes) {
            const uint4 v = uint4(row[u]) << 16;
            acc += dot(as_type<float4>(v), x[u]);
        }
    }
    mv_store(out, p, slot, r, lane, acc);
}

// Q4_0: 18-byte blocks of 32 (f16 d, 16 bytes): element j is the low
// nibble of byte j minus 8, element j + 16 the high one.
kernel void mv_q4_0(device const uchar* w [[buffer(0)]],
                    device const float* xb [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant MvParams& p [[buffer(3)]],
                    device const uint* route [[buffer(7)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid % kMvLanes;
    const uint gps = (p.rows + kMvRows - 1) / kMvRows;  // groups a slot
    const uint slot = tg / gps;
    const uint r = tg % gps * kMvRows + tid / kMvLanes;
    const ulong wo = mv_w_off(p, route, slot);  // the weight's bytes
    device const float* xs =
        xb + p.x_off + ulong(slot / p.slot_div) * p.x_slot_stride;
    float acc = 0.0f;
    if (r < p.rows) {
        const uint nb = p.cols / 32;
        device const uchar* row = w + wo + (ulong)r * nb * 18;
        device const float* x = xs;
        for (uint b = lane; b < nb; b += kMvLanes) {
            device const uchar* blk = row + b * 18;
            device const uchar* q = blk + 2;
            device const float4* xv = (device const float4*)(x + b * 32);
            float s = 0.0f;
            for (uint k = 0; k < 4; ++k) {
                const uchar4 v(q[4 * k], q[4 * k + 1], q[4 * k + 2],
                               q[4 * k + 3]);
                s += dot(float4(v & uchar4(0xF)) - 8.0f, xv[k]);
                s += dot(float4(v >> 4) - 8.0f, xv[k + 4]);
            }
            acc += f16_at(blk) * s;
        }
    }
    mv_store(out, p, slot, r, lane, acc);
}

// Q5_0: 22-byte blocks of 32 (f16 d, 32-bit qh, 16 bytes): element j
// is (low nibble of byte j | bit j of qh << 4) - 16, element j + 16
// (high nibble | bit j + 16 << 4) - 16.
kernel void mv_q5_0(device const uchar* w [[buffer(0)]],
                    device const float* xb [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant MvParams& p [[buffer(3)]],
                    device const uint* route [[buffer(7)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid % kMvLanes;
    const uint gps = (p.rows + kMvRows - 1) / kMvRows;  // groups a slot
    const uint slot = tg / gps;
    const uint r = tg % gps * kMvRows + tid / kMvLanes;
    const ulong wo = mv_w_off(p, route, slot);  // the weight's bytes
    device const float* xs =
        xb + p.x_off + ulong(slot / p.slot_div) * p.x_slot_stride;
    float acc = 0.0f;
    if (r < p.rows) {
        const uint nb = p.cols / 32;
        device const uchar* row = w + wo + (ulong)r * nb * 22;
        device const float* x = xs;
        for (uint b = lane; b < nb; b += kMvLanes) {
            device const uchar* blk = row + b * 22;
            const uint qh = uint(blk[2]) | (uint(blk[3]) << 8) |
                            (uint(blk[4]) << 16) | (uint(blk[5]) << 24);
            device const uchar* q = blk + 6;
            device const float* xv = x + b * 32;
            float s = 0.0f;
            for (uint j = 0; j < 16; ++j) {
                const int lo = int((q[j] & 0xF) | (((qh >> j) & 1) << 4));
                const int hi =
                    int((q[j] >> 4) | (((qh >> (j + 16)) & 1) << 4));
                s += float(lo - 16) * xv[j] + float(hi - 16) * xv[j + 16];
            }
            acc += f16_at(blk) * s;
        }
    }
    mv_store(out, p, slot, r, lane, acc);
}

// Q4_K: 144-byte super-blocks of 256 (f16 d and dmin, 12 packed 6-bit
// scale / min pairs, 128 nibble bytes). A unit is one pair of
// sub-blocks 2j, 2j + 1, which share 32 nibble bytes (low, high).

// ggml's get_scale_min_k4 for sub-block j.
inline void q4k_scale_min(uint j, device const uchar* q, thread float& sc,
                          thread float& mn) {
    if (j < 4) {
        sc = float(q[j] & 63);
        mn = float(q[j + 4] & 63);
    } else {
        sc = float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        mn = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// The threadgroup is 2 simdgroups of 2 rows each (kQ4Rows a group).
// Lane l works on blocks l / 8, + 4, ...: in each it takes 8 elements
// from four sub-blocks, those at 8 (l % 4) of sub-blocks 2 jq, 2 jq + 1,
// 2 jq + 4 and 2 jq + 5 (jq = l % 8 / 4): two 8-byte runs of nibbles.
// The 32 activations and their four sums are loaded once and serve
// both rows. (The layout of ggml's Metal Q4_K matvec.)
constant constexpr uint kQ4Rows = 4;

inline float dot8(float4 a, float4 b, uint2 q, uint sh) {
    const float4 q0 = float4((as_type<uchar4>(q.x) >> sh) & uchar4(0xF));
    const float4 q1 = float4((as_type<uchar4>(q.y) >> sh) & uchar4(0xF));
    return dot(q0, a) + dot(q1, b);
}

kernel void mv_q4_k(device const uchar* w [[buffer(0)]],
                    device const float* xb [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant MvParams& p [[buffer(3)]],
                    device const uint* route [[buffer(7)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint gps = (p.rows + kQ4Rows - 1) / kQ4Rows;  // groups a slot
    const uint slot = tg / gps;
    const uint r0 = tg % gps * kQ4Rows + sg * 2;
    const ulong wo = mv_w_off(p, route, slot);  // the weight's bytes
    device const float* xs =
        xb + p.x_off + ulong(slot / p.slot_div) * p.x_slot_stride;
    const uint nb = p.cols / 256;
    const ulong row_bytes = ulong(nb) * 144;
    const uint ix = lane / 8, jq = (lane % 8) / 4, ir = lane % 4;
    device const uchar* w0 = w + wo + ulong(r0) * row_bytes;
    device const float* x = xs + 64 * jq + 8 * ir;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint b = ix; b < nb; b += 4) {
        device const float4* y = (device const float4*)(x + b * 256);
        // Elements of sub-blocks 2 jq, 2 jq + 1, 2 jq + 4, 2 jq + 5.
        const float4 a0 = y[0], a1 = y[1], b0 = y[8], b1 = y[9];
        const float4 c0 = y[32], c1 = y[33], e0 = y[40], e1 = y[41];
        const float s0 = hsum(a0 + a1), s1 = hsum(b0 + b1);
        const float s2 = hsum(c0 + c1), s3 = hsum(e0 + e1);
        for (uint i = 0; i < 2; ++i) {
            if (r0 + i >= p.rows) {
                break;
            }
            device const uchar* blk = w0 + i * row_bytes + b * 144;
            const half2 dm = *(device const half2*)blk;
            float sc0, mn0, sc1, mn1, sc2, mn2, sc3, mn3;
            q4k_scale_min(2 * jq, blk + 4, sc0, mn0);
            q4k_scale_min(2 * jq + 1, blk + 4, sc1, mn1);
            q4k_scale_min(2 * jq + 4, blk + 4, sc2, mn2);
            q4k_scale_min(2 * jq + 5, blk + 4, sc3, mn3);
            device const uchar* q = blk + 16 + 32 * jq + 8 * ir;
            const uint2 lo = *(device const uint2*)q;
            const uint2 hi = *(device const uint2*)(q + 64);
            const float v = float(dm.x) * (sc0 * dot8(a0, a1, lo, 0) +
                                           sc1 * dot8(b0, b1, lo, 4) +
                                           sc2 * dot8(c0, c1, hi, 0) +
                                           sc3 * dot8(e0, e1, hi, 4)) -
                            float(dm.y) * (mn0 * s0 + mn1 * s1 + mn2 * s2 +
                                           mn3 * s3);
            if (i == 0) {
                acc0 += v;
            } else {
                acc1 += v;
            }
        }
    }
    const float t0 = simd_sum(acc0), t1 = simd_sum(acc1);
    if (lane == 0) {
        for (uint i = 0; i < 2 && r0 + i < p.rows; ++i) {
            device float* o =
                out + p.out_off + ulong(slot) * p.out_slot_stride + r0 + i;
            const float v = p.scale * (i == 0 ? t0 : t1);
            *o = p.accumulate != 0 ? *o + v : v;
        }
    }
}

// Q5_K: 176-byte super-blocks of 256 (f16 d and dmin, 12 packed
// scale / min bytes as Q4_K's, 32 qh bytes, 128 nibble bytes). Pair j
// (sub-blocks 2 j and 2 j + 1) takes nibble bytes 32 j ..: element
// 64 j + l is (low nibble of byte l | bit 2 j of qh[l] << 4), element
// 64 j + 32 + l (high nibble | bit 2 j + 1 << 4). A unit is one pair.
kernel void mv_q5_k(device const uchar* w [[buffer(0)]],
                    device const float* xb [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant MvParams& p [[buffer(3)]],
                    device const uint* route [[buffer(7)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid % kMvLanes;
    const uint gps = (p.rows + kMvRows - 1) / kMvRows;  // groups a slot
    const uint slot = tg / gps;
    const uint r = tg % gps * kMvRows + tid / kMvLanes;
    const ulong wo = mv_w_off(p, route, slot);  // the weight's bytes
    device const float* xs =
        xb + p.x_off + ulong(slot / p.slot_div) * p.x_slot_stride;
    float acc = 0.0f;
    if (r < p.rows) {
        const uint nb = p.cols / 256;
        device const uchar* row = w + wo + (ulong)r * nb * 176;
        device const float* x = xs;
        for (uint u = lane; u < nb * 4; u += kMvLanes) {
            const uint b = u >> 2, j = u & 3;
            device const uchar* blk = row + b * 176;
            float sc0, mn0, sc1, mn1;
            q4k_scale_min(2 * j, blk + 4, sc0, mn0);
            q4k_scale_min(2 * j + 1, blk + 4, sc1, mn1);
            device const uchar* qh = blk + 16;
            device const uchar* q = blk + 48 + 32 * j;
            device const float* xl = x + b * 256 + 64 * j;
            float sq0 = 0.0f, sx0 = 0.0f, sq1 = 0.0f, sx1 = 0.0f;
            for (uint l = 0; l < 32; ++l) {
                const uint h = qh[l] >> (2 * j);
                const float a = xl[l], c = xl[l + 32];
                sq0 += float((q[l] & 0xF) | ((h & 1) << 4)) * a;
                sq1 += float((q[l] >> 4) | ((h & 2) << 3)) * c;
                sx0 += a;
                sx1 += c;
            }
            const float d = f16_at(blk), dmin = f16_at(blk + 2);
            acc += d * sc0 * sq0 - dmin * mn0 * sx0;
            acc += d * sc1 * sq1 - dmin * mn1 * sx1;
        }
    }
    mv_store(out, p, slot, r, lane, acc);
}

// Q6_K: 210-byte super-blocks of 256 (128 ql, 64 qh, 16 int8 scales,
// f16 d at 208). Half n (0, 1) of a block holds elements 128 n ..; a
// unit is one (n, i) with i = 0, 1: the 16 l in [16 i, 16 i + 16) of
// that half, each giving elements l, l + 32, l + 64 and l + 96 with
// scales sc[8 n + i + {0, 2, 4, 6}]. The block stride is not a multiple
// of 4, so the bytes load one at a time.
kernel void mv_q6_k(device const uchar* w [[buffer(0)]],
                    device const float* xb [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant MvParams& p [[buffer(3)]],
                    device const uint* route [[buffer(7)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid % kMvLanes;
    const uint gps = (p.rows + kMvRows - 1) / kMvRows;  // groups a slot
    const uint slot = tg / gps;
    const uint r = tg % gps * kMvRows + tid / kMvLanes;
    const ulong wo = mv_w_off(p, route, slot);  // the weight's bytes
    device const float* xs =
        xb + p.x_off + ulong(slot / p.slot_div) * p.x_slot_stride;
    float acc = 0.0f;
    if (r < p.rows) {
        const uint nb = p.cols / 256;
        device const uchar* row = w + wo + (ulong)r * nb * 210;
        device const float* x = xs;
        for (uint u = lane; u < nb * 4; u += kMvLanes) {
            const uint b = u >> 2;
            const uint n = (u >> 1) & 1;
            const uint i = u & 1;
            device const uchar* blk = row + b * 210;
            device const uchar* ql = blk + 64 * n + 16 * i;
            device const uchar* qh = blk + 128 + 32 * n + 16 * i;
            device const char* sc =
                (device const char*)(blk + 192) + 8 * n + i;
            device const float* xs = x + b * 256 + 128 * n + 16 * i;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (uint l = 0; l < 16; ++l) {
                const uint a = ql[l], c = ql[l + 32], h = qh[l];
                const int q0 = int((a & 0xF) | ((h & 3) << 4)) - 32;
                const int q1 = int((c & 0xF) | (((h >> 2) & 3) << 4)) - 32;
                const int q2 = int((a >> 4) | (((h >> 4) & 3) << 4)) - 32;
                const int q3 = int((c >> 4) | (((h >> 6) & 3) << 4)) - 32;
                s0 += float(q0) * xs[l];
                s1 += float(q1) * xs[l + 32];
                s2 += float(q2) * xs[l + 64];
                s3 += float(q3) * xs[l + 96];
            }
            acc += f16_at(blk + 208) *
                   (float(sc[0]) * s0 + float(sc[2]) * s1 +
                    float(sc[4]) * s2 + float(sc[6]) * s3);
        }
    }
    mv_store(out, p, slot, r, lane, acc);
}

// ---------------------------------------------------------------------
// The text forward's other steps (i#866). Offsets count floats.

// The sum of v over a threadgroup of up to 1024 threads (32
// simdgroups); every thread gets it. `part` holds 32 floats.
inline float group_sum(float v, threadgroup float* part, uint tid,
                       uint n_threads) {
    v = simd_sum(v);
    const uint sg = tid / 32, n_sg = (n_threads + 31) / 32;
    if (tid % 32 == 0) {
        part[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint i = 0; i < n_sg; ++i) {
        s += part[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

inline float group_max(float v, threadgroup float* part, uint tid,
                       uint n_threads) {
    v = simd_max(v);
    const uint sg = tid / 32, n_sg = (n_threads + 31) / 32;
    if (tid % 32 == 0) {
        part[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY;
    for (uint i = 0; i < n_sg; ++i) {
        m = max(m, part[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

struct NormParams {
    uint n;
    float eps;
};

// out = x / sqrt(mean(x^2) + eps) * w, one threadgroup.
kernel void rmsnorm_f32(device const float* x [[buffer(0)]],
                        device const float* w [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant NormParams& p [[buffer(3)]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint nt [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    float s = 0.0f;
    for (uint i = tid; i < p.n; i += nt) {
        s += x[i] * x[i];
    }
    s = group_sum(s, part, tid, nt);
    const float inv = 1.0f / sqrt(s / float(p.n) + p.eps);
    for (uint i = tid; i < p.n; i += nt) {
        out[i] = x[i] * inv * w[i];
    }
}

struct RopeParams {
    ulong off;      // the first head's first float
    uint n_heads;
    uint head_dim;
    uint neox;      // NEOX pairs (i, i + head_dim / 2), else (2 i, 2 i + 1)
};

// In place, pair i of each head rotated by cs[2 i] (cos) and
// cs[2 i + 1] (sin): elements (2 i, 2 i + 1) (rope_norm), or (i,
// i + head_dim / 2) with neox (the same angle per pair): one thread a
// pair.
kernel void rope_norm(device float* v [[buffer(0)]],
                      device const float* cs [[buffer(1)]],
                      constant RopeParams& p [[buffer(2)]],
                      uint g [[thread_position_in_grid]]) {
    const uint half_d = p.head_dim / 2;
    if (g >= p.n_heads * half_d) {
        return;
    }
    const uint h = g / half_d, i = g % half_d;
    device float* hv = v + p.off + h * p.head_dim;
    device float* e0 = p.neox != 0 ? hv + i : hv + 2 * i;
    device float* e1 = p.neox != 0 ? hv + i + half_d : hv + 2 * i + 1;
    const float c = cs[2 * i], s = cs[2 * i + 1];
    const float a = *e0, b = *e1;
    *e0 = a * c - b * s;
    *e1 = a * s + b * c;
}

// Paged attention for one query over positions [0, n_pos): position t
// of the sequence is at blocks[t / bt] * block_floats + layer_off +
// (t % bt) * kv_dim (K), and v_off past it (V); head h reads KV head
// h / (n_heads / n_kv_heads).
struct AttnParams {
    ulong block_floats;
    ulong layer_off;
    ulong v_off;
    uint n_pos;
    uint block_tokens;
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    float kq_scale;
};

inline ulong kv_row(constant AttnParams& p, device const uint* blocks,
                    uint t) {
    return ulong(blocks[t / p.block_tokens]) * p.block_floats +
           p.layer_off + ulong(t % p.block_tokens) * p.n_kv_heads *
                             p.head_dim;
}

// scores[h * n_pos + t] = kq_scale * dot(q_h, k_t): one thread each.
kernel void attn_scores(device const float* q [[buffer(0)]],
                        device const float* pool [[buffer(1)]],
                        device const uint* blocks [[buffer(2)]],
                        device float* scores [[buffer(3)]],
                        constant AttnParams& p [[buffer(4)]],
                        uint g [[thread_position_in_grid]]) {
    if (g >= p.n_heads * p.n_pos) {
        return;
    }
    const uint h = g / p.n_pos, t = g % p.n_pos;
    const uint kvh = h / (p.n_heads / p.n_kv_heads);
    device const float4* k = (device const float4*)(
        pool + kv_row(p, blocks, t) + kvh * p.head_dim);
    device const float4* qh = (device const float4*)(q + h * p.head_dim);
    float s = 0.0f;
    for (uint i = 0; i < p.head_dim / 4; ++i) {
        s += dot(qh[i], k[i]);
    }
    scores[g] = s * p.kq_scale;
}

// Softmax of each head's n_pos scores in place: one threadgroup a head.
kernel void attn_softmax(device float* scores [[buffer(0)]],
                         constant AttnParams& p [[buffer(1)]],
                         uint h [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    device float* s = scores + h * p.n_pos;
    float m = -INFINITY;
    for (uint t = tid; t < p.n_pos; t += nt) {
        m = max(m, s[t]);
    }
    m = group_max(m, part, tid, nt);
    float sum = 0.0f;
    for (uint t = tid; t < p.n_pos; t += nt) {
        const float e = exp(s[t] - m);
        s[t] = e;
        sum += e;
    }
    sum = group_sum(sum, part, tid, nt);
    const float inv = 1.0f / sum;
    for (uint t = tid; t < p.n_pos; t += nt) {
        s[t] *= inv;
    }
}

// out[h * head_dim + d] = sum_t p[h, t] * v_t[kvh * head_dim + d]: one
// thread each.
kernel void attn_values(device const float* scores [[buffer(0)]],
                        device const float* pool [[buffer(1)]],
                        device const uint* blocks [[buffer(2)]],
                        device float* out [[buffer(3)]],
                        constant AttnParams& p [[buffer(4)]],
                        uint g [[thread_position_in_grid]]) {
    if (g >= p.n_heads * p.head_dim) {
        return;
    }
    const uint h = g / p.head_dim, d = g % p.head_dim;
    const uint kvh = h / (p.n_heads / p.n_kv_heads);
    device const float* s = scores + h * p.n_pos;
    float acc = 0.0f;
    for (uint t = 0; t < p.n_pos; ++t) {
        acc += s[t] * pool[kv_row(p, blocks, t) + p.v_off +
                           kvh * p.head_dim + d];
    }
    out[g] = acc;
}

struct SiluParams {
    uint n;
};

// gate[i] = silu(gate[i]) * up[i].
kernel void silu_mul(device float* gate [[buffer(0)]],
                     device const float* up [[buffer(1)]],
                     constant SiluParams& p [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        const float g = gate[i];
        gate[i] = g / (1.0f + exp(-g)) * up[i];
    }
}

// The three attention passes in one kernel for up to kAttnMaxPos
// positions: one threadgroup a head computes its scores into
// threadgroup memory, takes their softmax, then sums the values (each
// thread a slice of positions for its dimension, the slices added
// through threadgroup memory). Same AttnParams and buffers as
// attn_scores, with the output at index 3.
constant constexpr uint kAttnMaxPos = 4096;
constant constexpr uint kAttnGroup = 256;

kernel void attn_fused(device const float* q [[buffer(0)]],
                       device const float* pool [[buffer(1)]],
                       device const uint* blocks [[buffer(2)]],
                       device float* out [[buffer(3)]],
                       constant AttnParams& p [[buffer(4)]],
                       uint h [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float s[kAttnMaxPos];
    threadgroup float part[32];
    threadgroup float4 red[kAttnGroup];
    const uint kvh = h / (p.n_heads / p.n_kv_heads);
    device const float4* qh = (device const float4*)(q + h * p.head_dim);
    float m = -INFINITY;
    for (uint t = tid; t < p.n_pos; t += kAttnGroup) {
        device const float4* k = (device const float4*)(
            pool + kv_row(p, blocks, t) + kvh * p.head_dim);
        float d = 0.0f;
        for (uint i = 0; i < p.head_dim / 4; ++i) {
            d += dot(qh[i], k[i]);
        }
        d *= p.kq_scale;
        s[t] = d;
        m = max(m, d);
    }
    m = group_max(m, part, tid, kAttnGroup);
    float sum = 0.0f;
    for (uint t = tid; t < p.n_pos; t += kAttnGroup) {
        const float e = exp(s[t] - m);
        s[t] = e;
        sum += e;
    }
    sum = group_sum(sum, part, tid, kAttnGroup);
    const float inv = 1.0f / sum;
    // Values: head_dim / 4 threads a position (a float4 each), each
    // slice of them over every (kAttnGroup / (head_dim / 4))-th one.
    const uint n4 = p.head_dim / 4, per = kAttnGroup / n4;
    const uint c = tid % n4, slice = tid / n4;
    float4 acc = 0.0f;
    if (slice < per) {
        for (uint t = slice; t < p.n_pos; t += per) {
            acc += s[t] * ((device const float4*)(
                              pool + kv_row(p, blocks, t) + p.v_off +
                              kvh * p.head_dim))[c];
        }
    }
    red[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slice == 0) {
        float4 v = 0.0f;
        for (uint j = 0; j < per; ++j) {
            v += red[j * n4 + c];
        }
        ((device float4*)(out + h * p.head_dim))[c] = v * inv;
    }
}

struct AttnSplitParams {
    AttnParams a;
    uint n_parts;  // kAttnGroup positions a part
};

// Split-K decode attention, pass 1: threadgroup g is head g / n_parts
// over part g % n_parts's kAttnGroup positions (a thread a position).
// It writes the part's max score m, its exp sum l and its unnormalized
// value sums into part[g * (head_dim + 4) ..]: head_dim sums, m, l
// (the stride a float4 multiple).
kernel void attn_split(device const float* q [[buffer(0)]],
                       device const float* pool [[buffer(1)]],
                       device const uint* blocks [[buffer(2)]],
                       device float* part_out [[buffer(3)]],
                       constant AttnSplitParams& sp [[buffer(4)]],
                       uint g [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    constant AttnParams& p = sp.a;
    threadgroup float s[kAttnGroup];
    threadgroup ulong rows[kAttnGroup];
    threadgroup float part[32];
    threadgroup float4 red[kAttnGroup];
    const uint h = g / sp.n_parts, pi = g % sp.n_parts;
    const uint kvh = h / (p.n_heads / p.n_kv_heads);
    const uint t0 = pi * kAttnGroup;
    const uint cnt = min(kAttnGroup, p.n_pos - t0);
    device const float4* qh = (device const float4*)(q + h * p.head_dim);
    float d = -INFINITY;
    if (tid < cnt) {
        const ulong row = kv_row(p, blocks, t0 + tid) + kvh * p.head_dim;
        rows[tid] = row;
        device const float4* k = (device const float4*)(pool + row);
        d = 0.0f;
        for (uint i = 0; i < p.head_dim / 4; ++i) {
            d += dot(qh[i], k[i]);
        }
        d *= p.kq_scale;
    }
    const float m = group_max(d, part, tid, kAttnGroup);
    const float e = tid < cnt ? exp(d - m) : 0.0f;
    s[tid] = e;
    const float l = group_sum(e, part, tid, kAttnGroup);
    // Values: n4 = head_dim / 4 threads a position (a float4 each),
    // kAttnGroup / n4 slices of the positions.
    const uint n4 = p.head_dim / 4, per = kAttnGroup / n4;
    const uint c = tid % n4, slice = tid / n4;
    float4 acc = 0.0f;
    if (slice < per) {
        for (uint t = slice; t < cnt; t += per) {
            acc += s[t] *
                   ((device const float4*)(pool + rows[t] + p.v_off))[c];
        }
    }
    red[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* o = part_out + ulong(g) * (p.head_dim + 4);
    if (slice == 0) {
        float4 v = 0.0f;
        for (uint j = 0; j < per; ++j) {
            v += red[j * n4 + c];
        }
        ((device float4*)o)[c] = v;
    }
    if (tid == 0) {
        o[p.head_dim] = m;
        o[p.head_dim + 1] = l;
    }
}

// Pass 2, a thread a (head, dimension): the parts rescaled to the
// head's max, out = sum exp(m_p - M) acc_p / sum exp(m_p - M) l_p.
kernel void attn_merge(device const float* part_out [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant AttnSplitParams& sp [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    constant AttnParams& p = sp.a;
    if (i >= p.n_heads * p.head_dim) {
        return;
    }
    const uint h = i / p.head_dim, c = i % p.head_dim;
    const uint w = p.head_dim + 4;
    device const float* b = part_out + ulong(h) * sp.n_parts * w;
    float mx = -INFINITY;
    for (uint j = 0; j < sp.n_parts; ++j) {
        mx = max(mx, b[j * w + p.head_dim]);
    }
    float num = 0.0f, den = 0.0f;
    for (uint j = 0; j < sp.n_parts; ++j) {
        const float f = exp(b[j * w + p.head_dim] - mx);
        num += f * b[j * w + c];
        den += f * b[j * w + p.head_dim + 1];
    }
    out[i] = num / den;
}

// ---------------------------------------------------------------------
// Prefill matmul (i#866, from #799): Y[t][r] = scale * (W X)[r][t] (+ Y)
// for Q4_K W rows and f32 token rows of X and Y. A threadgroup of 128
// (4 simdgroups) takes 64 weight rows x 32 tokens and walks k in stages
// of 32 (one sub-block): the threads dequantize the weight tile and
// stage the activations in threadgroup memory (float), then each
// simdgroup runs a 32 x 16 piece as eight 8 x 8 simdgroup_matrix tiles.
// Rows and tokens past the matrix load as zero and are not stored.


struct MmBlockQ4K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

struct MmParams {
    ulong w_off;     // bytes
    ulong x_off;     // floats
    ulong out_off;   // floats
    uint rows, cols, n_tok;
    uint x_stride, out_stride;
    uint accumulate;
    float scale;
    uint row_tiles;  // threadgroup t is tile (t % row_tiles, t / row_tiles)
    // mm_id_*: x of slot s is row s / slot_div; expert e's matrix at
    // w_off + e * expert_bytes.
    uint slot_div;
    ulong expert_bytes;
};

constant constexpr uint kMmTM = 64;
constant constexpr uint kMmTN = 32;
constant constexpr uint kMmTK = 32;
constant constexpr uint kMmPA = kMmTK + 8;
constant constexpr uint kMmPB = kMmTK + 8;
constant constexpr uint kMmPO = kMmTM + 4;

// Sub-block j's 6-bit scale and min (ggml's get_scale_min_k4).
inline void mm_scale_min(uint j, device const uchar* q, thread uint& sc,
                      thread uint& mn) {
    if (j < 4) {
        sc = q[j] & 63;
        mn = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// The weight loaders: each fills da[0 .. 16) with the dequantized
// weights k0 + ah .. k0 + ah + 15 of a row (ah 0 or 16; a k stage of
// 32 lies in one sub-block or block), and gives a row's bytes.
struct MmLoadQ4K {
    static ulong row_bytes(uint cols) { return ulong(cols / 256) * 144; }
    static void load(device const uchar* row, uint k0, uint ah,
                     threadgroup float* da) {
        device const MmBlockQ4K& blk =
            ((device const MmBlockQ4K*)row)[k0 / 256];
        const uint j = (k0 % 256) / 32;
        uint sc, mn;
        mm_scale_min(j, blk.scales, sc, mn);
        const float d = float(blk.d) * float(sc);
        const float mm = float(blk.dmin) * float(mn);
        device const uchar* q = blk.qs + 32 * (j / 2) + ah;
        const uint sh = (j & 1) * 4;
        for (uint i = 0; i < 16; ++i) {
            da[i] = d * float((q[i] >> sh) & 0xF) - mm;
        }
    }
};

struct MmLoadQ6K {
    static ulong row_bytes(uint cols) { return ulong(cols / 256) * 210; }
    // Stage k0 is quarter qq of half hn of its block: l = ah + i (the
    // layout as mv_q6_k; 210-byte blocks, read bytewise).
    static void load(device const uchar* row, uint k0, uint ah,
                     threadgroup float* da) {
        device const uchar* blk = row + (k0 / 256) * 210;
        const uint e0 = k0 % 256, hn = e0 / 128, qq = (e0 % 128) / 32;
        device const uchar* ql = blk + 64 * hn + 32 * (qq & 1) + ah;
        device const uchar* qh = blk + 128 + 32 * hn + ah;
        const float d =
            f16_at(blk + 208) *
            float(((device const char*)(blk + 192))[8 * hn + ah / 16 +
                                                   2 * qq]);
        const uint lsh = (qq >> 1) * 4, hsh = 2 * qq;
        for (uint i = 0; i < 16; ++i) {
            const int q = int(((ql[i] >> lsh) & 0xF) |
                              (((qh[i] >> hsh) & 3) << 4)) -
                          32;
            da[i] = d * float(q);
        }
    }
};

struct MmLoadQ8_0 {
    static ulong row_bytes(uint cols) { return ulong(cols / 32) * 34; }
    // Stage k0 is block k0 / 32: d, then 32 int8.
    static void load(device const uchar* row, uint k0, uint ah,
                     threadgroup float* da) {
        device const uchar* blk = row + (k0 / 32) * 34;
        const float d = f16_at(blk);
        device const char* qv = (device const char*)(blk + 2) + ah;
        for (uint i = 0; i < 16; ++i) {
            da[i] = d * float(qv[i]);
        }
    }
};

// Q4_0: 18-byte blocks (f16 d, 16 bytes): element j is the low nibble
// of byte j less 8, element j + 16 its high nibble less 8.
struct MmLoadQ4_0 {
    static ulong row_bytes(uint cols) { return ulong(cols / 32) * 18; }
    static void load(device const uchar* row, uint k0, uint ah,
                     threadgroup float* da) {
        device const uchar* blk = row + (k0 / 32) * 18;
        const float d = f16_at(blk);
        device const uchar* q = blk + 2;
        const uint sh = ah == 0 ? 0 : 4;
        for (uint i = 0; i < 16; ++i) {
            da[i] = d * (float((q[i] >> sh) & 0xF) - 8.0f);
        }
    }
};

// Q5_0: 22-byte blocks (f16 d, 32-bit qh, 16 bytes): as Q4_0 with bit
// j of qh the fifth bit of element j, less 16.
struct MmLoadQ5_0 {
    static ulong row_bytes(uint cols) { return ulong(cols / 32) * 22; }
    static void load(device const uchar* row, uint k0, uint ah,
                     threadgroup float* da) {
        device const uchar* blk = row + (k0 / 32) * 22;
        const float d = f16_at(blk);
        const uint qh = uint(blk[2]) | (uint(blk[3]) << 8) |
                        (uint(blk[4]) << 16) | (uint(blk[5]) << 24);
        device const uchar* q = blk + 6;
        const uint sh = ah == 0 ? 0 : 4;
        for (uint i = 0; i < 16; ++i) {
            const uint v = ((q[i] >> sh) & 0xF) | (((qh >> (ah + i)) & 1) << 4);
            da[i] = d * (float(v) - 16.0f);
        }
    }
};

// The float weights (the image and encoder models carry them): 16
// consecutive values of a row, F32 as is, F16 and BF16 widened.
struct MmLoadF32 {
    static ulong row_bytes(uint cols) { return ulong(cols) * 4; }
    static void load(device const uchar* row, uint k0, uint ah,
                     threadgroup float* da) {
        device const float* v = (device const float*)row + k0 + ah;
        for (uint i = 0; i < 16; ++i) {
            da[i] = v[i];
        }
    }
};

struct MmLoadF16 {
    static ulong row_bytes(uint cols) { return ulong(cols) * 2; }
    static void load(device const uchar* row, uint k0, uint ah,
                     threadgroup float* da) {
        device const half* v = (device const half*)row + k0 + ah;
        for (uint i = 0; i < 16; ++i) {
            da[i] = float(v[i]);
        }
    }
};

struct MmLoadBF16 {
    static ulong row_bytes(uint cols) { return ulong(cols) * 2; }
    static void load(device const uchar* row, uint k0, uint ah,
                     threadgroup float* da) {
        device const ushort* v = (device const ushort*)row + k0 + ah;
        for (uint i = 0; i < 16; ++i) {
            da[i] = as_type<float>(uint(v[i]) << 16);
        }
    }
};

// The tile shared by every weight type, its loader L the one difference.
// With G (mm_id_*, n_tok the slot count n), column tile tg.y is
// moe_tiles' entry idx[n + 3 tg.y ..] = (expert, first, count): its
// columns are the slots idx[first ..] (sorted by expert), their x rows
// slot / slot_div, their outputs at row slot.
template <typename L, bool G>
inline void mm_tile(device const uchar* wbase, device const float* xbase,
                    device float* ybase, constant MmParams& p, uint tgi,
                    uint tid, uint sg, threadgroup float* sa,
                    threadgroup float* sb, threadgroup float* so,
                    device const uint* idx) {
    const uint2 tg = uint2(tgi % p.row_tiles, tgi / p.row_tiles);
    device const uchar* w = wbase + p.w_off;
    device const float* x = xbase + p.x_off;
    device float* y = ybase + p.out_off;
    const uint m0 = tg.x * kMmTM;
    uint n0 = tg.y * kMmTN;
    uint n_tok = p.n_tok;
    device const uint* perm = idx;
    if (G) {
        device const uint* tile = idx + p.n_tok + 3 * tg.y;
        n_tok = tile[2];
        if (n_tok == 0) {
            return;  // past the last tile: the whole threadgroup
        }
        w += ulong(tile[0]) * p.expert_bytes;
        perm = idx + tile[1];
        n0 = 0;
    }
    const uint rm = (sg & 1) * 32;
    const uint rn = (sg >> 1) * 16;
    simdgroup_float8x8 acc[4][2];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

    // Loaders: A, row tid / 2, 16 elements at (tid & 1) * 16; B, token
    // tid / 4, 8 elements at (tid & 3) * 8.
    const uint ar = tid >> 1;
    const uint ah = (tid & 1) * 16;
    const uint bt = tid >> 2;
    const uint bk = (tid & 3) * 8;
    const bool a_in = m0 + ar < p.rows;
    const bool b_in = n0 + bt < n_tok;
    device const uchar* wrow =
        w + (a_in ? ulong(m0 + ar) * L::row_bytes(p.cols) : 0);
    const uint xr = !b_in ? 0 : G ? perm[bt] / p.slot_div : n0 + bt;
    device const float* xrow = x + ulong(xr) * p.x_stride;

    for (uint k0 = 0; k0 < p.cols; k0 += kMmTK) {
        threadgroup float* da = sa + ar * kMmPA + ah;
        if (a_in) {
            L::load(wrow, k0, ah, da);
        } else {
            for (uint i = 0; i < 16; ++i) da[i] = 0.0f;
        }
        threadgroup float* db = sb + bt * kMmPB + bk;
        if (b_in) {
            device const float* src = xrow + k0 + bk;
            for (uint i = 0; i < 8; ++i) db[i] = src[i];
        } else {
            for (uint i = 0; i < 8; ++i) db[i] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < kMmTK; kk += 8) {
            simdgroup_float8x8 a[4], bm[2];
            for (uint i = 0; i < 4; ++i)
                simdgroup_load(a[i], sa + (rm + 8 * i) * kMmPA + kk, kMmPA);
            // B is k x tokens: the [token][k] tile, loaded transposed.
            for (uint j = 0; j < 2; ++j)
                simdgroup_load(bm[j], sb + (rn + 8 * j) * kMmPB + kk, kMmPB,
                               ulong2(0, 0), true);
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 2; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], bm[j],
                                                  acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The tile as [token][row] in threadgroup memory, then each thread
    // stores its share inside the matrix.
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 2; ++j)
            simdgroup_store(acc[i][j], so + (rn + 8 * j) * kMmPO + rm + 8 * i,
                            kMmPO, ulong2(0, 0), true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < kMmTM * kMmTN; e += 128) {
        const uint t = e / kMmTM, r = e % kMmTM;
        if (n0 + t < n_tok && m0 + r < p.rows) {
            const uint orow = G ? perm[t] : n0 + t;
            device float* o = y + ulong(orow) * p.out_stride + m0 + r;
            const float v = p.scale * so[t * kMmPO + r];
            *o = p.accumulate != 0 ? *o + v : v;
        }
    }
}

#define MM_KERNEL(name, loader)                                          \
    kernel void name(device const uchar* wbase [[buffer(0)]],           \
                     device const float* xbase [[buffer(1)]],           \
                     device float* ybase [[buffer(2)]],                 \
                     constant MmParams& p [[buffer(3)]],                \
                     uint tgi [[threadgroup_position_in_grid]],         \
                     uint tid [[thread_index_in_threadgroup]],          \
                     uint sg [[simdgroup_index_in_threadgroup]]) {      \
        threadgroup float sa[kMmTM * kMmPA];                            \
        threadgroup float sb[kMmTN * kMmPB];                            \
        threadgroup float so[kMmTN * kMmPO];                            \
        mm_tile<loader, false>(wbase, xbase, ybase, p, tgi, tid, sg,    \
                               sa, sb, so, nullptr);                    \
    }                                                                   \
    kernel void name##_id(device const uchar* wbase [[buffer(0)]],      \
                          device const float* xbase [[buffer(1)]],      \
                          device float* ybase [[buffer(2)]],            \
                          constant MmParams& p [[buffer(3)]],           \
                          device const uint* idx [[buffer(4)]],         \
                          uint tgi [[threadgroup_position_in_grid]],    \
                          uint tid [[thread_index_in_threadgroup]],     \
                          uint sg [[simdgroup_index_in_threadgroup]]) { \
        threadgroup float sa[kMmTM * kMmPA];                            \
        threadgroup float sb[kMmTN * kMmPB];                            \
        threadgroup float so[kMmTN * kMmPO];                            \
        mm_tile<loader, true>(wbase, xbase, ybase, p, tgi, tid, sg, sa, \
                              sb, so, idx);                             \
    }

MM_KERNEL(mm_q4_k, MmLoadQ4K)
MM_KERNEL(mm_q6_k, MmLoadQ6K)
MM_KERNEL(mm_q8_0, MmLoadQ8_0)
MM_KERNEL(mm_q4_0, MmLoadQ4_0)
MM_KERNEL(mm_q5_0, MmLoadQ5_0)
MM_KERNEL(mm_f32, MmLoadF32)
MM_KERNEL(mm_f16, MmLoadF16)
MM_KERNEL(mm_bf16, MmLoadBF16)

struct MoeTilesParams {
    uint n_slots;    // m k: the route's ids at route[0 .. n_slots)
    uint n_expert;
    uint max_tiles;  // n_slots / kMmTN + n_expert
};

// The slots grouped by expert for mm_id_*: idx[0 .. n) the slots in
// expert order (each expert's in slot order), then from idx[n] 3 uints
// a tile: (expert, its first slot's index into idx, count up to kMmTN),
// zero counts past the last. One threadgroup, a thread an expert at a
// time; deterministic.
kernel void moe_tiles(device const uint* route [[buffer(0)]],
                      device uint* idx [[buffer(1)]],
                      constant MoeTilesParams& p [[buffer(2)]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint nt [[threads_per_threadgroup]]) {
    threadgroup uint cnt[1024];
    threadgroup uint tiles_at[1024];
    const uint n = p.n_slots;
    for (uint e = tid; e < p.n_expert; e += nt) {
        uint c = 0;
        for (uint s = 0; s < n; ++s) {
            c += route[s] == e ? 1u : 0u;
        }
        cnt[e] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint first = 0, tl = 0;
        for (uint e = 0; e < p.n_expert; ++e) {
            const uint c = cnt[e];
            cnt[e] = first;
            tiles_at[e] = tl;
            first += c;
            tl += (c + kMmTN - 1) / kMmTN;
        }
        for (uint i = tl; i < p.max_tiles; ++i) {
            idx[n + 3 * i + 2] = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < p.n_expert; e += nt) {
        const uint first = cnt[e];
        uint c = 0;
        for (uint s = 0; s < n; ++s) {
            if (route[s] == e) {
                idx[first + c] = s;
                ++c;
            }
        }
        for (uint j = 0; j * kMmTN < c; ++j) {
            device uint* tile = idx + n + 3 * (tiles_at[e] + j);
            tile[0] = e;
            tile[1] = first + j * kMmTN;
            tile[2] = min(c - j * kMmTN, kMmTN);
        }
    }
}

// ---------------------------------------------------------------------
// The prefill forward's row kernels (i#866): n token rows at once.

struct RowNormParams {
    uint n;       // row width
    uint stride;  // floats between rows (in and out)
    float eps;
};

// rmsnorm_f32 on row t = threadgroup t: out[t] = x[t] / rms * w.
kernel void rmsnorm_rows(device const float* x [[buffer(0)]],
                         device const float* w [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         constant RowNormParams& p [[buffer(3)]],
                         uint t [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    device const float* xr = x + ulong(t) * p.stride;
    device float* o = out + ulong(t) * p.stride;
    float s = 0.0f;
    for (uint i = tid; i < p.n; i += nt) {
        s += xr[i] * xr[i];
    }
    s = group_sum(s, part, tid, nt);
    const float inv = 1.0f / sqrt(s / float(p.n) + p.eps);
    for (uint i = tid; i < p.n; i += nt) {
        o[i] = xr[i] * inv * w[i];
    }
}

struct RopeRowsParams {
    ulong off;    // token 0's first float
    uint n_tok;
    uint n_heads;
    uint head_dim;
    uint stride;  // floats between token rows
    uint neox;    // NEOX pairs, as rope_norm
};

// rope_norm on n token rows: token t's heads at off + t * stride, its
// (cos, sin) table at cs + t * head_dim. One thread a pair.
kernel void rope_rows(device float* v [[buffer(0)]],
                      device const float* cs [[buffer(1)]],
                      constant RopeRowsParams& p [[buffer(2)]],
                      uint g [[thread_position_in_grid]]) {
    const uint half_d = p.head_dim / 2, per = p.n_heads * half_d;
    if (g >= p.n_tok * per) {
        return;
    }
    const uint t = g / per, j = g % per;
    const uint h = j / half_d, i = j % half_d;
    device float* hv = v + p.off + ulong(t) * p.stride + h * p.head_dim;
    device float* e0 = p.neox != 0 ? hv + i : hv + 2 * i;
    device float* e1 = p.neox != 0 ? hv + i + half_d : hv + 2 * i + 1;
    device const float* c = cs + t * p.head_dim + 2 * i;
    const float a = *e0, b = *e1;
    *e0 = a * c[0] - b * c[1];
    *e1 = a * c[1] + b * c[0];
}

struct BiasParams {
    ulong off;     // token 0's first float
    uint n;        // the bias's length
    uint n_tok;
    uint stride;   // floats between token rows
};

// v[off + t * stride + i] += b[i] for n_tok rows: the q / k / v
// projection biases (qwen2).
kernel void add_bias(device float* v [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     constant BiasParams& p [[buffer(2)]],
                     uint g [[thread_position_in_grid]]) {
    if (g >= p.n_tok * p.n) {
        return;
    }
    const uint t = g / p.n, i = g % p.n;
    v[p.off + ulong(t) * p.stride + i] += b[i];
}

struct ScatterParams {
    ulong v_off;  // V's offset past K in a pool row
    uint n_tok;
    uint kv_dim;
};

// The chunk's K and V rows (n_tok rows of kv_dim) into the KV pool:
// token t's K row at rows[t], its V row v_off past it.
kernel void kv_scatter(device const float* k [[buffer(0)]],
                       device const float* v [[buffer(1)]],
                       device float* pool [[buffer(2)]],
                       device const ulong* rows [[buffer(3)]],
                       constant ScatterParams& p [[buffer(4)]],
                       uint g [[thread_position_in_grid]]) {
    if (g >= p.n_tok * p.kv_dim) {
        return;
    }
    const uint t = g / p.kv_dim, i = g % p.kv_dim;
    pool[rows[t] + i] = k[g];
    pool[rows[t] + p.v_off + i] = v[g];
}

struct AttnRowsParams {
    AttnParams a;    // n_pos unused: query t sees base + t + 1
    uint base;       // the chunk's first position
    uint q_stride;   // floats between the queries' (and outputs') rows
};

// Causal attention for a chunk: threadgroup g is query t = g / n_heads,
// head h = g % n_heads, over positions [0, base + t]; as attn_fused,
// up to kAttnMaxPos positions.
kernel void attn_rows(device const float* q [[buffer(0)]],
                      device const float* pool [[buffer(1)]],
                      device const uint* blocks [[buffer(2)]],
                      device float* out [[buffer(3)]],
                      constant AttnRowsParams& pr [[buffer(4)]],
                      uint g [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float s[kAttnMaxPos];
    threadgroup float part[32];
    threadgroup float red[kAttnGroup];
    constant AttnParams& p = pr.a;
    const uint t = g / p.n_heads, h = g % p.n_heads;
    const uint n_pos = pr.base + t + 1;
    const uint kvh = h / (p.n_heads / p.n_kv_heads);
    device const float4* qh =
        (device const float4*)(q + ulong(t) * pr.q_stride + h * p.head_dim);
    float m = -INFINITY;
    for (uint j = tid; j < n_pos; j += kAttnGroup) {
        device const float4* k = (device const float4*)(
            pool + kv_row(p, blocks, j) + kvh * p.head_dim);
        float d = 0.0f;
        for (uint i = 0; i < p.head_dim / 4; ++i) {
            d += dot(qh[i], k[i]);
        }
        d *= p.kq_scale;
        s[j] = d;
        m = max(m, d);
    }
    m = group_max(m, part, tid, kAttnGroup);
    float sum = 0.0f;
    for (uint j = tid; j < n_pos; j += kAttnGroup) {
        const float e = exp(s[j] - m);
        s[j] = e;
        sum += e;
    }
    sum = group_sum(sum, part, tid, kAttnGroup);
    const float inv = 1.0f / sum;
    const uint per = kAttnGroup / p.head_dim;
    const uint d = tid % p.head_dim, slice = tid / p.head_dim;
    float acc = 0.0f;
    if (slice < per) {
        for (uint j = slice; j < n_pos; j += per) {
            acc += s[j] * pool[kv_row(p, blocks, j) + p.v_off +
                               kvh * p.head_dim + d];
        }
    }
    red[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slice == 0) {
        float v = 0.0f;
        for (uint j = 0; j < per; ++j) {
            v += red[j * p.head_dim + d];
        }
        out[ulong(t) * pr.q_stride + h * p.head_dim + d] = v * inv;
    }
}


// Causal attention for a chunk, tiled: threadgroup g takes queries
// 8 (g / n_heads) .. + 8 of head g % n_heads, one simdgroup a query.
// The positions go by in chunks of min(kTileFloats / head_dim, 32), a
// position a lane, whose K and V rows are staged in threadgroup memory
// once for all 8 queries; each query keeps an online softmax (running
// max m, sum l, and its output dims d = lane + 32 i in registers).
// head_dim is at most 128 and a multiple of 32. Same parameters and
// buffers as attn_rows.
constant constexpr uint kTileQ = 8;
constant constexpr uint kTileFloats = 2048;

kernel void attn_rows_tiled(device const float* q [[buffer(0)]],
                            device const float* pool [[buffer(1)]],
                            device const uint* blocks [[buffer(2)]],
                            device float* out [[buffer(3)]],
                            constant AttnRowsParams& pr [[buffer(4)]],
                            uint g [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float kt[kTileFloats];
    threadgroup float vt[kTileFloats];
    threadgroup float qs[kTileQ * 128];
    constant AttnParams& p = pr.a;
    // A chunk is at most 32 positions: lane j scores position c + j.
    const uint hd = p.head_dim, ch = min(kTileFloats / hd, 32u);
    const uint h = g % p.n_heads, t0 = (g / p.n_heads) * kTileQ;
    const uint kvh = h / (p.n_heads / p.n_kv_heads);
    const uint t = t0 + sg;  // this simdgroup's query
    const bool live = t < p.n_pos;  // n_pos: the chunk's token count
    // The 8 queries' rows, staged once.
    for (uint i = tid; i < kTileQ * hd; i += kTileQ * 32) {
        const uint qi = i / hd;
        qs[i] = t0 + qi < p.n_pos
                    ? q[ulong(t0 + qi) * pr.q_stride + h * hd + i % hd]
                    : 0.0f;
    }
    const uint last = pr.base + min(t0 + kTileQ, p.n_pos);  // positions
    const uint mine = pr.base + t + 1;  // this query sees [0, mine)
    float m = -INFINITY, l = 0.0f;
    float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint nd = hd / 32;
    for (uint c = 0; c < last; c += ch) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < ch * hd; i += kTileQ * 32) {
            const uint j = c + i / hd;
            if (j < last) {
                const ulong r = kv_row(p, blocks, j) + kvh * hd + i % hd;
                kt[i] = pool[r];
                vt[i] = pool[r + p.v_off];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!live || c >= mine) {
            continue;
        }
        // Lane j scores position c + j.
        const uint j = c + lane;
        float sc = -INFINITY;
        if (lane < ch && j < mine) {
            threadgroup const float4* kr =
                (threadgroup const float4*)(kt + lane * hd);
            threadgroup const float4* qr =
                (threadgroup const float4*)(qs + sg * hd);
            float d = 0.0f;
            for (uint i = 0; i < hd / 4; ++i) {
                d += dot(qr[i], kr[i]);
            }
            sc = d * p.kq_scale;
        }
        const float cm = simd_max(sc);
        const float mn = max(m, cm);
        const float e = sc == -INFINITY ? 0.0f : exp(sc - mn);
        const float scale = m == -INFINITY ? 0.0f : exp(m - mn);
        l = l * scale + simd_sum(e);
        for (uint i = 0; i < nd; ++i) {
            o[i] *= scale;
        }
        const uint nj = min(ch, mine - c);
        for (uint jj = 0; jj < nj; ++jj) {
            const float pj = simd_shuffle(e, jj);
            for (uint i = 0; i < nd; ++i) {
                o[i] += pj * vt[jj * hd + lane + 32 * i];
            }
        }
        m = mn;
    }
    if (live) {
        const float inv = 1.0f / l;
        for (uint i = 0; i < nd; ++i) {
            out[ulong(t) * pr.q_stride + h * hd + lane + 32 * i] = o[i] * inv;
        }
    }
}


// Causal attention for a chunk on the simdgroup matrices: threadgroup
// g takes queries 32 (g / n_heads) .. + 32 of head g % n_heads, a
// simdgroup (of 4) 8 of them, over 32-position chunks. A chunk's K
// rows are staged, each simdgroup takes its 8 x 32 scores Q K^T (Q,
// pre-scaled by kq_scale, kept in registers), the rows' online softmax
// turns them into P and rescales the output O (8 x head_dim, in
// registers) by a diagonal matrix, then the V rows replace K in the
// same threadgroup memory and O += P V. head_dim is a multiple of 8,
// at most 128. Same parameters and buffers as attn_rows.
constant constexpr uint kFaQ = 32;  // queries a threadgroup
constant constexpr uint kFaC = 32;  // positions a chunk

kernel void attn_rows_mma(device const float* q [[buffer(0)]],
                          device const float* pool [[buffer(1)]],
                          device const uint* blocks [[buffer(2)]],
                          device float* out [[buffer(3)]],
                          constant AttnRowsParams& pr [[buffer(4)]],
                          uint g [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float kv[kFaC * 128];
    threadgroup float spall[4 * 8 * kFaC];
    threadgroup float dgall[4 * 64];
    constant AttnParams& p = pr.a;
    const uint hd = p.head_dim, nd = hd / 8;
    const uint h = g % p.n_heads, t0 = (g / p.n_heads) * kFaQ;
    const uint kvh = h / (p.n_heads / p.n_kv_heads);
    const uint q0 = t0 + sg * 8;  // this simdgroup's first query
    threadgroup float* sp = spall + sg * 8 * kFaC;
    threadgroup float* dg = dgall + sg * 64;

    // Q, scaled, as nd 8 x 8 matrices through dg (rows past the chunk
    // as zero).
    simdgroup_float8x8 qm[16], om[16];
    for (uint k = 0; k < nd; ++k) {
        for (uint e = lane; e < 64; e += 32) {
            const uint r = e / 8, c = e % 8;
            dg[e] = q0 + r < p.n_pos
                        ? q[ulong(q0 + r) * pr.q_stride + h * hd + k * 8 +
                            c] * p.kq_scale
                        : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(qm[k], dg, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        om[k] = simdgroup_float8x8(0.0f);
    }
    float m[8], l[8];
    for (uint r = 0; r < 8; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }
    // Positions [0, last) are seen by some query of the block.
    const uint last = pr.base + min(t0 + kFaQ, p.n_pos);
    const uint mine_end = pr.base + min(q0 + 8, p.n_pos);  // this sg's
    for (uint c = 0; c < last; c += kFaC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < kFaC * hd; i += 128) {
            const uint j = c + i / hd;
            kv[i] = j < last ? pool[kv_row(p, blocks, j) + kvh * hd + i % hd]
                             : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool act = q0 < p.n_pos && c < mine_end;
        if (act) {
            // S = Q K^T, 8 x 32, into sp.
            for (uint cb = 0; cb < kFaC / 8; ++cb) {
                simdgroup_float8x8 acc(0.0f);
                for (uint k = 0; k < nd; ++k) {
                    simdgroup_float8x8 kb;
                    simdgroup_load(kb, kv + cb * 8 * hd + k * 8, hd,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(acc, qm[k], kb, acc);
                }
                simdgroup_store(acc, sp + cb * 8, kFaC);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            // The rows' online softmax: lane j holds position c + j.
            for (uint e = lane; e < 64; e += 32) {
                dg[e] = 0.0f;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint j = c + lane;
            for (uint r = 0; r < 8; ++r) {
                const uint qi = q0 + r;
                const bool see = qi < p.n_pos && j <= pr.base + qi;
                const float s = see ? sp[r * kFaC + lane] : -INFINITY;
                const float mn = max(m[r], simd_max(s));
                const float e = s == -INFINITY ? 0.0f : exp(s - mn);
                const float sc = m[r] == -INFINITY ? 0.0f : exp(m[r] - mn);
                l[r] = l[r] * sc + simd_sum(e);
                m[r] = mn;
                sp[r * kFaC + lane] = e;
                if (lane == 0) {
                    dg[r * 9] = sc;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 dm;
            simdgroup_load(dm, dg, 8);
            for (uint k = 0; k < nd; ++k) {
                simdgroup_float8x8 t;
                simdgroup_multiply(t, dm, om[k]);
                om[k] = t;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < kFaC * hd; i += 128) {
            const uint j = c + i / hd;
            kv[i] = j < last ? pool[kv_row(p, blocks, j) + p.v_off +
                                    kvh * hd + i % hd]
                             : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (act) {
            // O += P V.
            simdgroup_float8x8 pm[kFaC / 8];
            for (uint kb = 0; kb < kFaC / 8; ++kb) {
                simdgroup_load(pm[kb], sp + kb * 8, kFaC);
            }
            for (uint n = 0; n < nd; ++n) {
                for (uint kb = 0; kb < kFaC / 8; ++kb) {
                    simdgroup_float8x8 vb;
                    simdgroup_load(vb, kv + kb * 8 * hd + n * 8, hd);
                    simdgroup_multiply_accumulate(om[n], pm[kb], vb, om[n]);
                }
            }
        }
    }
    // O / l, through dg, rows inside the chunk.
    for (uint n = 0; n < nd; ++n) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(om[n], dg, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64; e += 32) {
            const uint r = e / 8, c = e % 8;
            if (q0 + r < p.n_pos) {
                out[ulong(q0 + r) * pr.q_stride + h * hd + n * 8 + c] =
                    dg[e] / l[r];
            }
        }
    }
}


struct AxpyParams {
    uint n;
    float a;
};

// x[i] += a * y[i]: a routed expert's output into the residual, by its
// mixing weight (MoE, i#868).
kernel void axpy(device float* x [[buffer(0)]],
                 device const float* y [[buffer(1)]],
                 constant AxpyParams& p [[buffer(2)]],
                 uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        x[i] += p.a * y[i];
    }
}

struct AxpyRouteParams {
    uint n;
    uint slot;  // the weight is the float at route[slot]
};

// axpy with its weight from the route buffer (a routed expert's mixing
// weight, or the shared expert's gate), so no host round trip.
kernel void axpy_routed(device float* x [[buffer(0)]],
                        device const float* y [[buffer(1)]],
                        device const uint* route [[buffer(2)]],
                        constant AxpyRouteParams& p [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        x[i] += as_type<float>(route[p.slot]) * y[i];
    }
}

struct RouteParams {
    uint n_expert;
    uint k;            // experts a token
    uint softmax;      // 1: softmax gating, 0: sigmoid
    uint bias_on_logit;
    uint has_bias;     // exp_probs_b (selection bias) given
    uint norm;         // normalize the picked weights to sum 1
    float scale;       // expert_weights_scale
    uint shared_gate;  // also the gate: sigmoid(logits[n_expert])
    uint n_tok;        // tokens, a threadgroup each
    uint logit_stride; // floats between the tokens' logit rows
};

// moe_select on the GPU for one token (one threadgroup of 1024):
// gating, the selection score (plus the bias), the top k by repeated
// argmax (ties to the lowest id, as the CPU's strict >), and the
// weights. For token t (threadgroup t of n_tok): route[t k .. t k + k)
// gets its ids, route[n_tok k + t k ..] its weights as float bits, and
// route[2 n_tok k + t] its shared expert's sigmoid gate when asked (one
// token: ids at 0, weights at k, the gate at 2 k).
// n_expert at most 1024; no group-limited routing (the CPU keeps it).
// Its softmax adds in another order than the CPU's, so on a near-tie
// between two experts the pick can differ from the CPU forward's.
kernel void moe_route(device const float* logits_all [[buffer(0)]],
                      device const float* bias [[buffer(1)]],
                      device uint* route_all [[buffer(2)]],
                      constant RouteParams& p [[buffer(3)]],
                      uint tok [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint nt [[threads_per_threadgroup]]) {
    device const float* logits = logits_all + ulong(tok) * p.logit_stride;
    device uint* ids = route_all + tok * p.k;
    device uint* wts = route_all + (p.n_tok + tok) * p.k;
    threadgroup float prob[1024];
    threadgroup float sel[1024];
    threadgroup float part[32];
    threadgroup float bv[32];
    threadgroup uint bi[32];
    const bool live = tid < p.n_expert;
    const float lg = live ? logits[tid] : -INFINITY;
    float pr;
    if (p.softmax != 0) {
        const float m = group_max(lg, part, tid, nt);
        const float e = live ? exp(lg - m) : 0.0f;
        const float s = group_sum(e, part, tid, nt);
        pr = e / s;
    } else {
        pr = 1.0f / (1.0f + exp(-lg));
    }
    float sc = p.bias_on_logit != 0 ? lg : pr;
    if (live && p.has_bias != 0) {
        sc += bias[tid];
    }
    if (live) {
        prob[tid] = pr;
        sel[tid] = sc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float wsum = 0.0f;
    for (uint j = 0; j < p.k; ++j) {
        // The argmax of sel, ties to the lowest id.
        float v = live ? sel[tid] : -INFINITY;
        uint id = live ? tid : 0xFFFFFFFFu;
        for (uint o = 16; o > 0; o >>= 1) {
            const float v2 = simd_shuffle_down(v, o);
            const uint i2 = simd_shuffle_down(id, o);
            if (v2 > v || (v2 == v && i2 < id)) {
                v = v2;
                id = i2;
            }
        }
        if (tid % 32 == 0) {
            bv[tid / 32] = v;
            bi[tid / 32] = id;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float best = bv[0];
            uint at = bi[0];
            for (uint g = 1; g < (nt + 31) / 32; ++g) {
                if (bv[g] > best || (bv[g] == best && bi[g] < at)) {
                    best = bv[g];
                    at = bi[g];
                }
            }
            ids[j] = at;
            sel[at] = -INFINITY;  // taken
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        for (uint j = 0; j < p.k; ++j) {
            wsum += prob[ids[j]];
        }
        const float div = p.norm != 0 ? wsum : 1.0f;
        for (uint j = 0; j < p.k; ++j) {
            wts[j] = as_type<uint>(prob[ids[j]] / div * p.scale);
        }
        if (p.shared_gate != 0) {
            const float g = logits[p.n_expert];
            route_all[2 * p.n_tok * p.k + tok] =
                as_type<uint>(1.0f / (1.0f + exp(-g)));
        }
    }
}

struct HeadNormParams {
    ulong off;     // token 0's first float
    uint n_tok;
    uint n_heads;
    uint head_dim;
    uint stride;   // floats between token rows
    float eps;
};

// Per-head RMS norm in place (qwen3's q / k norm): each head's
// head_dim values times 1 / rms and the head_dim-wide weight w. A
// simdgroup a (token, head).
kernel void head_rmsnorm(device float* v [[buffer(0)]],
                         device const float* w [[buffer(1)]],
                         constant HeadNormParams& p [[buffer(2)]],
                         uint g [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    if (g >= p.n_tok * p.n_heads) {
        return;
    }
    const uint t = g / p.n_heads, h = g % p.n_heads;
    device float* x =
        v + p.off + ulong(t) * p.stride + ulong(h) * p.head_dim;
    float s = 0.0f;
    for (uint i = lane; i < p.head_dim; i += 32) {
        s += x[i] * x[i];
    }
    s = simd_sum(s);
    const float inv = 1.0f / sqrt(s / float(p.head_dim) + p.eps);
    for (uint i = lane; i < p.head_dim; i += 32) {
        x[i] = x[i] * inv * w[i];
    }
}

struct MoeSumParams {
    uint n;      // the residual's width
    uint k;      // experts a token
    uint n_tok;  // token t: outputs at eout[(t k + j) n ..], weights at
                 // route[n_tok k + t k + j], its residual row x[t n ..]
};

// x[i] += sum_j w_j * eout[j * n + i]: the k experts of a token into
// the residual in one dispatch, weights from moe_route.
kernel void moe_sum(device float* x [[buffer(0)]],
                    device const float* eout [[buffer(1)]],
                    device const uint* route [[buffer(2)]],
                    constant MoeSumParams& p [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= p.n * p.n_tok) {
        return;
    }
    const uint t = i / p.n, c = i % p.n;
    float acc = 0.0f;
    for (uint j = 0; j < p.k; ++j) {
        const uint s = t * p.k + j;
        acc += as_type<float>(route[p.n_tok * p.k + s]) *
               eout[ulong(s) * p.n + c];
    }
    x[i] += acc;
}

struct SharedAddParams {
    uint n;       // the residual's width
    uint n_tok;
    uint gated;   // weight token t by route[gate_at + t], else 1
    uint gate_at;
};

// x[t] += g_t * y[t] for n_tok rows: the shared expert's output into
// the residual by each token's gate (qwen2moe) or 1.
kernel void shared_add(device float* x [[buffer(0)]],
                       device const float* y [[buffer(1)]],
                       device const uint* route [[buffer(2)]],
                       constant SharedAddParams& p [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.n * p.n_tok) {
        return;
    }
    const float g =
        p.gated != 0 ? as_type<float>(route[p.gate_at + i / p.n]) : 1.0f;
    x[i] += g * y[i];
}

// ---------------------------------------------------------------------
// conv2d (i#1050): the image VAEs' 2D convolution, groups 1, as an
// implicit GEMM on the simdgroup matrices: out[oc][p] = bias[oc] +
// sum_k W[oc][k] B[k][p], k = (ci, ky, kx) in the weight's order and B
// the input pixels under the kernel (zero outside the plane).

struct Conv2dParams {
    uint in_ch, in_h, in_w, out_ch, oh, ow, kh, kw, stride, pad;
    uint has_bias, K, P;  // K = in_ch kh kw, P = oh ow
};

constant constexpr uint kCvT = 64;  // a tile's out channels
constant constexpr uint kCvP = 32;  // a tile's pixels
constant constexpr uint kCvK = 16;  // the K step staged a time
// The staged tiles' row strides, padded off a multiple of 32 floats so
// a simdgroup_load's 8 rows fall in different banks.
constant constexpr uint kCvPA = kCvK + 4;
constant constexpr uint kCvPB = kCvP + 4;

// tab[k]: k's input offset (ci in_h + ky) in_w + kx, then ky << 16 | kx,
// so the staging divides nothing. Threadgroup g (128 threads) takes
// out channels 64 (g / tiles_p) .. +64 and pixels 32 (g % tiles_p) ..
// +32; simdgroup sg a 32 x 16 quarter as 4 x 2 blocks of 8 x 8 (more
// accumulators a simdgroup ran slower: 64 x 64 tiles took twice as
// long). Each K step stages W's 64 x 16 and B's 16 x 32 in threadgroup
// memory.
kernel void conv2d_gemm(device const float* in [[buffer(0)]],
                        device const float* wt [[buffer(1)]],
                        device const float* bias [[buffer(2)]],
                        device float* out [[buffer(3)]],
                        device const uint2* tab [[buffer(4)]],
                        constant Conv2dParams& p [[buffer(5)]],
                        uint g [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float ta[kCvT * kCvPA];  // W: [oc][k]
    threadgroup float tb[kCvK * kCvPB];  // B: [k][pixel]
    threadgroup float dgall[4 * 64];
    threadgroup float* d = dgall + sg * 64;
    const uint tiles_p = (p.P + kCvP - 1) / kCvP;
    const uint p0 = (g % tiles_p) * kCvP, oc0 = (g / tiles_p) * kCvT;
    const uint sr = (sg & 1) * 32, sc = (sg >> 1) * 16;

    // The bias, broadcast along the pixels, starts the sums.
    simdgroup_float8x8 acc[4][2];
    for (uint i = 0; i < 4; ++i) {
        for (uint e = lane; e < 64; e += 32) {
            const uint oc = oc0 + sr + i * 8 + e / 8;
            d[e] = p.has_bias != 0 && oc < p.out_ch ? bias[oc] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 bm;
        simdgroup_load(bm, d, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 2; ++j) {
            acc[i][j] = bm;
        }
    }

    // B: this thread's pixel (128 threads over 32 pixels) and its rows
    // bk, bk + 4, ...; W: column tid % 16 of rows tid / 16, + 8, ...
    // (16 lanes read a row's 16 floats together).
    const uint bp = tid % kCvP, bk = tid / kCvP;
    const uint pix = p0 + bp;
    const bool pin = pix < p.P;
    const uint oy = pin ? pix / p.ow : 0, ox = pin ? pix % p.ow : 0;
    const int iy0 = int(oy * p.stride) - int(p.pad);
    const int ix0 = int(ox * p.stride) - int(p.pad);
    const long base = long(iy0) * p.in_w + ix0;
    const uint ac = tid % kCvK, ar0 = tid / kCvK;
    for (uint k0 = 0; k0 < p.K; k0 += kCvK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ar = ar0; ar < kCvT; ar += 128 / kCvK) {
            const uint k = k0 + ac, oc = oc0 + ar;
            ta[ar * kCvPA + ac] =
                oc < p.out_ch && k < p.K ? wt[ulong(oc) * p.K + k] : 0.0f;
        }
        for (uint i = 0; i < kCvK / 4; ++i) {
            const uint kr = bk + 4 * i, k = k0 + kr;
            float v = 0.0f;
            if (pin && k < p.K) {
                const uint2 t = tab[k];
                const int iy = iy0 + int(t.y >> 16);
                const int ix = ix0 + int(t.y & 0xffffu);
                if (iy >= 0 && iy < int(p.in_h) && ix >= 0 &&
                    ix < int(p.in_w)) {
                    v = in[base + long(t.x)];
                }
            }
            tb[kr * kCvPB + bp] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < kCvK; kk += 8) {
            simdgroup_float8x8 a[4], b[2];
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(a[i], ta + (sr + i * 8) * kCvPA + kk, kCvPA);
            }
            for (uint j = 0; j < 2; ++j) {
                simdgroup_load(b[j], tb + kk * kCvPB + sc + j * 8, kCvPB);
            }
            for (uint i = 0; i < 4; ++i) {
                for (uint j = 0; j < 2; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j],
                                                  acc[i][j]);
                }
            }
        }
    }

    // Whole blocks straight out; a block over the channel or pixel tail
    // through d, its inside rows and columns only.
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 2; ++j) {
            const uint r0 = oc0 + sr + i * 8, c0 = p0 + sc + j * 8;
            if (r0 >= p.out_ch || c0 >= p.P) {
                continue;
            }
            if (r0 + 8 <= p.out_ch && c0 + 8 <= p.P) {
                simdgroup_store(acc[i][j], out + ulong(r0) * p.P + c0, p.P);
                continue;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(acc[i][j], d, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                const uint r = r0 + e / 8, c = c0 + e % 8;
                if (r < p.out_ch && c < p.P) {
                    out[ulong(r) * p.P + c] = d[e];
                }
            }
        }
    }
}
