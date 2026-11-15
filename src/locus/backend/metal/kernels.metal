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
// half of the simdgroup. W starts at the bound buffer's offset, 16-byte
// aligned (so Q4_K's word loads are aligned); x_off is a multiple of 4
// (float4 loads).

struct MvParams {
    uint rows;
    uint cols;
    uint out_off;
    uint accumulate;
    uint x_off;
    float scale;
};

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

inline void mv_store(device float* out, constant MvParams& p, uint r,
                     uint lane, float acc) {
    const float s = half_simd_sum(acc);
    if (lane == 0 && r < p.rows) {
        device float* o = out + p.out_off + r;
        const float v = p.scale * s;
        *o = p.accumulate != 0 ? *o + v : v;
    }
}

inline float hsum(float4 v) { return v.x + v.y + v.z + v.w; }

inline float f16_at(device const uchar* b) {
    return float(as_type<half>(ushort(b[0] | (b[1] << 8))));
}

// F32: units of 4 columns.
kernel void mv_f32(device const float* w [[buffer(0)]],
                   device const float* xb [[buffer(1)]],
                   device float* out [[buffer(2)]],
                   constant MvParams& p [[buffer(3)]],
                   uint tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid % kMvLanes;
    const uint r = tg * kMvRows + tid / kMvLanes;
    float acc = 0.0f;
    if (r < p.rows) {
        device const float4* row =
            (device const float4*)(w + (ulong)r * p.cols);
        device const float4* x = (device const float4*)(xb + p.x_off);
        for (uint u = lane; u < p.cols / 4; u += kMvLanes) {
            acc += dot(row[u], x[u]);
        }
    }
    mv_store(out, p, r, lane, acc);
}

// Q8_0: 34-byte blocks (f16 d, 32 int8). Units are blocks; the block
// stride is not a multiple of 4, so the quants load as bytes.
kernel void mv_q8_0(device const uchar* w [[buffer(0)]],
                    device const float* xb [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant MvParams& p [[buffer(3)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid % kMvLanes;
    const uint r = tg * kMvRows + tid / kMvLanes;
    float acc = 0.0f;
    if (r < p.rows) {
        const uint nb = p.cols / 32;
        device const uchar* row = w + (ulong)r * nb * 34;
        device const float* x = xb + p.x_off;
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
    mv_store(out, p, r, lane, acc);
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

kernel void mv_q4_k(device const uchar* w [[buffer(0)]],
                    device const float* xb [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant MvParams& p [[buffer(3)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid % kMvLanes;
    const uint r = tg * kMvRows + tid / kMvLanes;
    float acc = 0.0f;
    if (r < p.rows) {
        const uint nb = p.cols / 256;
        device const uchar* row = w + (ulong)r * nb * 144;
        device const float* x = xb + p.x_off;
        for (uint u = lane; u < nb * 4; u += kMvLanes) {
            const uint b = u >> 2;
            const uint j = u & 3;
            device const uchar* blk = row + b * 144;
            const half2 dm = *(device const half2*)blk;
            float sc0, mn0, sc1, mn1;
            q4k_scale_min(2 * j, blk + 4, sc0, mn0);
            q4k_scale_min(2 * j + 1, blk + 4, sc1, mn1);
            device const uint* q = (device const uint*)(blk + 16 + j * 32);
            device const float4* xl =
                (device const float4*)(x + b * 256 + j * 64);
            device const float4* xh = xl + 8;
            float sq0 = 0.0f, sx0 = 0.0f, sq1 = 0.0f, sx1 = 0.0f;
            for (uint k = 0; k < 8; ++k) {
                const uint v = q[k];
                const float4 lo(v & 0xF, (v >> 8) & 0xF, (v >> 16) & 0xF,
                                (v >> 24) & 0xF);
                const float4 hi((v >> 4) & 0xF, (v >> 12) & 0xF,
                                (v >> 20) & 0xF, (v >> 28) & 0xF);
                const float4 a = xl[k];
                const float4 c = xh[k];
                sq0 += dot(lo, a);
                sx0 += hsum(a);
                sq1 += dot(hi, c);
                sx1 += hsum(c);
            }
            const float d = float(dm.x), dmin = float(dm.y);
            acc += d * sc0 * sq0 - dmin * mn0 * sx0;
            acc += d * sc1 * sq1 - dmin * mn1 * sx1;
        }
    }
    mv_store(out, p, r, lane, acc);
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
                    uint tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid % kMvLanes;
    const uint r = tg * kMvRows + tid / kMvLanes;
    float acc = 0.0f;
    if (r < p.rows) {
        const uint nb = p.cols / 256;
        device const uchar* row = w + (ulong)r * nb * 210;
        device const float* x = xb + p.x_off;
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
    mv_store(out, p, r, lane, acc);
}
