// CUDA attention over the paged KV cache: the device mirror of the
// cache (alloc, store, load, free) and cuda_kv_attend, a two-pass GQA
// attention (per-chunk partials, then a combine in chunk order, so the
// bits do not depend on how many queries share a launch). A shape the
// kernels do not take returns false and the host attends.

#include "locus/backend/cuda/cuda_common.cuh"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string_view>
#include <vector>

namespace locus::backend {

using namespace cuda_detail;

void* cuda_kv_mirror_alloc(std::size_t n_floats) {
    ensure_device();
    void* p = nullptr;
    if (n_floats == 0 ||
        !cuda_ok(cudaMalloc(&p, n_floats * sizeof(float)))) {
        return nullptr;
    }
    // Zeroed, so a row read before its first store is 0, not garbage.
    if (!cuda_ok(cudaMemset(p, 0, n_floats * sizeof(float)))) {
        cudaFree(p);
        return nullptr;
    }
    return p;
}

bool cuda_kv_mirror_store(void* dev, std::size_t off, const float* src,
                          std::size_t n) {
    ensure_device();
    return cuda_ok(cudaMemcpy(static_cast<float*>(dev) + off, src,
                              n * sizeof(float), cudaMemcpyHostToDevice));
}

bool cuda_kv_mirror_load(const void* dev, std::size_t off, float* dst,
                         std::size_t n) {
    ensure_device();
    return cuda_ok(cudaMemcpy(dst, static_cast<const float*>(dev) + off,
                              n * sizeof(float), cudaMemcpyDeviceToHost));
}

void cuda_kv_mirror_free(void* dev) {
    ensure_device();
    cudaFree(dev);
}

namespace {

constexpr unsigned kAttendThreads = 256;

/** Fixed-order tree reduction over the block (max when is_max, else
 *  sum); every thread gets the result. red holds blockDim floats. */
__device__ float block_reduce(float v, float* red, bool is_max) {
    const unsigned tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] = is_max ? fmaxf(red[tid], red[tid + s])
                              : red[tid] + red[tid + s];
        }
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

/** Positions per chunk of gqa_attend_partial. Fixed, so a query's chunks
 *  (and so its bits) do not depend on how many queries share a launch. */
constexpr std::uint32_t kAttendChunk = 128;

/**
 * GQA attention, first pass: one block per (KV head, query, chunk of
 * kAttendChunk positions). The block scores the query's `group` heads
 * of that KV head on the chunk's positions (one warp per K row, read
 * once for all of the group), and writes per head the chunk's max m,
 * sum l of exp(s - m) and weighted value sum acc, for gqa_attend_combine.
 * A chunk past the query's positions writes nothing.
 *
 * part layout, per (query, KV head, chunk, head of the group): m, l, then
 * hd floats of acc. Shared memory: group * hd floats of q, group *
 * kAttendChunk scores, blockDim floats for the reductions and the value
 * pass's partial sums (group * blockDim), kAttendChunk row offsets.
 * The group size is a template parameter, so its loops unroll and the
 * accumulators stay in registers. Needs hd dividing blockDim.
 */
template <std::uint32_t group>
__global__ void gqa_attend_partial(
    const float* __restrict__ kv, const std::uint32_t* __restrict__ blocks,
    std::size_t block_stride, std::size_t layer_off, std::size_t v_off,
    std::uint32_t bt, std::uint32_t kv_dim, std::uint32_t hd, float scale,
    const std::uint32_t* __restrict__ qpos,
    const std::uint32_t* __restrict__ qtab, const float* __restrict__ q,
    std::size_t q_stride, float* __restrict__ part) {
    extern __shared__ float sh[];
    const std::uint32_t kvh = blockIdx.x, qi = blockIdx.y, c = blockIdx.z;
    const std::uint32_t npos = qpos[qi] + 1;
    blocks += qtab[qi];  // this query's sequence
    const std::uint32_t t0 = c * kAttendChunk;
    if (t0 >= npos) {
        return;  // the whole block leaves: no barrier is skipped
    }
    const std::uint32_t n = min(kAttendChunk, npos - t0);
    const unsigned tid = threadIdx.x, nt = blockDim.x;
    const unsigned lane = tid % 32, warp = tid / 32, nw = nt / 32;
    float* qs = sh;
    float* sc = qs + group * hd;
    float* red = sc + group * kAttendChunk;  // group * nt floats
    // Rounded up to an even float count, so the offsets are 8-byte aligned.
    const std::size_t rows_at = (std::size_t{group} *
                                     (hd + kAttendChunk + nt) + 1) & ~std::size_t{1};
    auto* rows = reinterpret_cast<std::size_t*>(sh + rows_at);
    const std::size_t head0 = std::size_t{kvh} * group * hd;
    const float* qrow = q + qi * q_stride + head0;
    for (std::uint32_t i = tid; i < group * hd; i += nt) {
        qs[i] = qrow[i];
    }
    // The chunk's K row offsets, once (V is v_off past each).
    for (std::uint32_t j = tid; j < n; j += nt) {
        const std::uint32_t t = t0 + j;
        rows[j] = blocks[t / bt] * block_stride + layer_off +
                  std::size_t{t % bt} * kv_dim + std::size_t{kvh} * hd;
    }
    __syncthreads();

    // Scores: warp w takes positions w, w + nw, ... of the chunk; its
    // lanes split the row and a fixed xor tree sums them.
    for (std::uint32_t j = warp; j < n; j += nw) {
        const float* k = kv + rows[j];
        float s[group];
#pragma unroll
        for (std::uint32_t g = 0; g < group; ++g) {
            s[g] = 0.0f;
        }
        for (std::uint32_t i = lane; i < hd; i += 32) {
            const float kk = __ldg(k + i);
#pragma unroll
            for (std::uint32_t g = 0; g < group; ++g) {
                s[g] += qs[g * hd + i] * kk;
            }
        }
        // The group's xor trees interleave, so their shuffles overlap.
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
            for (std::uint32_t g = 0; g < group; ++g) {
                s[g] += __shfl_xor_sync(0xffffffffu, s[g], o);
            }
        }
        if (lane == 0) {
#pragma unroll
            for (std::uint32_t g = 0; g < group; ++g) {
                sc[g * kAttendChunk + j] = s[g] * scale;
            }
        }
    }
    __syncthreads();

    float* pq = part + ((std::size_t{qi} * gridDim.x + kvh) * gridDim.z + c) *
                           group * (hd + 2);
    for (std::uint32_t g = 0; g < group; ++g) {
        float* s = sc + g * kAttendChunk;
        float m = -INFINITY;
        for (std::uint32_t j = tid; j < n; j += nt) {
            m = fmaxf(m, s[j]);
        }
        m = block_reduce(m, red, true);
        float l = 0.0f;
        for (std::uint32_t j = tid; j < n; j += nt) {
            const float e = expf(s[j] - m);
            s[j] = e;
            l += e;
        }
        l = block_reduce(l, red, false);
        if (tid == 0) {
            pq[g * (hd + 2)] = m;
            pq[g * (hd + 2) + 1] = l;
        }
    }
    __syncthreads();

    // Weighted value sum: thread (p, d) takes dimension d of positions
    // p, p + P, ... (P = nt / hd slices) for every head of the group, so
    // each V float is read once; the P slices then add in slice order.
    const std::uint32_t P = nt / hd, p = tid / hd, d = tid % hd;
    float acc[group];
#pragma unroll
    for (std::uint32_t g = 0; g < group; ++g) {
        acc[g] = 0.0f;
    }
#pragma unroll 4
    for (std::uint32_t j = p; j < n; j += P) {
        const float v = __ldg(kv + rows[j] + v_off + d);
#pragma unroll
        for (std::uint32_t g = 0; g < group; ++g) {
            acc[g] += sc[g * kAttendChunk + j] * v;
        }
    }
#pragma unroll
    for (std::uint32_t g = 0; g < group; ++g) {
        red[g * nt + tid] = acc[g];
    }
    __syncthreads();
    for (std::uint32_t i = tid; i < group * hd; i += nt) {
        const std::uint32_t g = i / hd, dd = i % hd;
        float s = 0.0f;
        for (std::uint32_t k = 0; k < P; ++k) {
            s += red[g * nt + k * hd + dd];
        }
        pq[g * (hd + 2) + 2 + dd] = s;
    }
}

/**
 * GQA attention, second pass: one block per (KV head, query) merges the
 * query's chunks in chunk order, rescaling each by exp(m_c - M) for the
 * overall max M, and writes acc / l into out.
 */
__global__ void gqa_attend_combine(const float* __restrict__ part,
                                   std::uint32_t hd, std::uint32_t group,
                                   const std::uint32_t* __restrict__ qpos,
                                   std::uint32_t chunks,
                                   float* __restrict__ out,
                                   std::size_t q_stride) {
    const std::uint32_t kvh = blockIdx.x, qi = blockIdx.y;
    const std::uint32_t npos = qpos[qi] + 1;
    const std::uint32_t used = (npos + kAttendChunk - 1) / kAttendChunk;
    const std::size_t head0 = std::size_t{kvh} * group * hd;
    const float* pq =
        part + (std::size_t{qi} * gridDim.x + kvh) * chunks * group * (hd + 2);
    const std::size_t per_chunk = std::size_t{group} * (hd + 2);
    for (std::uint32_t i = threadIdx.x; i < group * hd; i += blockDim.x) {
        const std::uint32_t g = i / hd, d = i % hd;
        float mx = -INFINITY;
        for (std::uint32_t c = 0; c < used; ++c) {
            mx = fmaxf(mx, pq[c * per_chunk + g * (hd + 2)]);
        }
        float l = 0.0f, acc = 0.0f;
        for (std::uint32_t c = 0; c < used; ++c) {
            const float* p = pq + c * per_chunk + g * (hd + 2);
            const float w = expf(p[0] - mx);
            l += p[1] * w;
            acc += p[2 + d] * w;
        }
        out[qi * q_stride + head0 + i] = acc / l;
    }
}

/** Query-heads a flash block takes: up to 32 (the tile's queries times
 *  the group). */
constexpr std::uint32_t kFlashHeads = 32;

/**
 * Flash attention for a prefill chunk of one sequence (i#1254): one
 * block per (KV head, tile of QT = kFlashHeads / group queries). K and
 * V are staged in shared memory KT positions at a time, read once for
 * the tile's QT * group query-heads; a thread takes whole (query-head,
 * position) dots, so no shuffle tree; each query-head keeps a running
 * max and sum (online softmax) and its output in registers, so there is
 * no partial buffer and no second pass. A position past a query is
 * masked. Its sums run in another order than gqa_attend_partial's, so a
 * prefill's attention is close to, not bit for bit, a decode's
 * (DESIGN 16).
 *
 * Shared memory: H * (HD + 1) floats of q and KT * (HD + 1) of the K or
 * V tile (rows padded by one float, so lanes on different rows hit
 * different banks), H * KT scores, 3 H for the running max, sum and
 * rescale; KT row offsets, computed once a tile. A thread scores one
 * query-head against 4 positions, and sums its kPer heads' outputs
 * from one V float a position (i#1254).
 */
template <std::uint32_t HD>
__global__ void gqa_attend_flash(
    const float* __restrict__ kv, const std::uint32_t* __restrict__ blocks,
    std::size_t block_stride, std::size_t layer_off, std::size_t v_off,
    std::uint32_t bt, std::uint32_t kv_dim, float scale,
    const std::uint32_t* __restrict__ qpos, std::uint32_t n_q,
    std::uint32_t group, const float* __restrict__ q, std::size_t q_stride,
    float* __restrict__ out) {
    constexpr std::uint32_t KT = HD <= 64 ? 32 : 16;
    constexpr std::uint32_t KS = HD + 1;  // padded row stride
    constexpr std::uint32_t kPer = kFlashHeads * HD / kAttendThreads;
    constexpr std::uint32_t JG = 4;  // positions a thread scores at once
    extern __shared__ float sh[];
    __shared__ std::size_t rowoff[KT];  // the tile's K row offsets
    const std::uint32_t qt = kFlashHeads / group;
    const std::uint32_t H = qt * group;
    const std::uint32_t kvh = blockIdx.x, q0 = blockIdx.y * qt;
    const std::uint32_t nq = min(qt, n_q - q0);
    const unsigned tid = threadIdx.x, nt = blockDim.x;
    const unsigned lane = tid % 32, warp = tid / 32, nw = nt / 32;
    float* qs = sh;                    // [h][KS], h = query * group + g
    float* kvs = qs + H * KS;          // [KT][KS]
    float* ps = kvs + KT * KS;         // [h][KT]
    float* mh = ps + H * KT;           // running max a query-head
    float* lh = mh + H;                // running sum
    float* rs = lh + H;                // this tile's rescale
    const std::size_t head0 = std::size_t{kvh} * group * HD;
    std::uint32_t last = 0;  // the tile's positions: up to its last query
    for (std::uint32_t i = 0; i < nq; ++i) {
        last = max(last, qpos[q0 + i] + 1);
    }
    for (std::uint32_t e = tid; e < H * HD; e += nt) {
        const std::uint32_t h = e / HD, i = h / group;
        // Rows padded to KS: the score loop's 4 query-heads a warp read
        // different banks.
        qs[h * KS + e % HD] =
            i < nq ? q[(q0 + i) * q_stride + head0 +
                       std::size_t{h % group} * HD + e % HD]
                   : 0.0f;
    }
    for (std::uint32_t h = tid; h < H; h += nt) {
        mh[h] = -INFINITY;
        lh[h] = 0.0f;
    }
    float acc[kPer];
#pragma unroll
    for (std::uint32_t k = 0; k < kPer; ++k) {
        acc[k] = 0.0f;
    }
    const auto row_of = [&](std::uint32_t t) {
        return blocks[t / bt] * block_stride + layer_off +
               std::size_t{t % bt} * kv_dim + std::size_t{kvh} * HD;
    };
    for (std::uint32_t t0 = 0; t0 < last; t0 += KT) {
        const std::uint32_t n = min(KT, last - t0);
        __syncthreads();  // the last tile's V, scores and rows are used
        // Each position's row offset once a tile (a division by bt each),
        // not once a staged float.
        if (tid < KT) {
            rowoff[tid] = tid < n ? row_of(t0 + tid) : 0;
        }
        __syncthreads();
        for (std::uint32_t e = tid; e < KT * HD; e += nt) {
            const std::uint32_t j = e / HD, d = e % HD;
            kvs[j * KS + d] = j < n ? __ldg(kv + rowoff[j] + d) : 0.0f;
        }
        __syncthreads();
        // Scores: a thread a query-head and JG positions, so each q
        // float read serves JG dots.
        for (std::uint32_t e = tid; e < H * (KT / JG); e += nt) {
            const std::uint32_t h = e / (KT / JG);
            const std::uint32_t j0 = (e % (KT / JG)) * JG;
            const std::uint32_t i = h / group;
            const float* qr = qs + h * KS;
            float d[JG];
#pragma unroll
            for (std::uint32_t g = 0; g < JG; ++g) {
                d[g] = 0.0f;
            }
#pragma unroll 16
            for (std::uint32_t x = 0; x < HD; ++x) {
                const float qv = qr[x];
#pragma unroll
                for (std::uint32_t g = 0; g < JG; ++g) {
                    d[g] += qv * kvs[(j0 + g) * KS + x];
                }
            }
#pragma unroll
            for (std::uint32_t g = 0; g < JG; ++g) {
                const std::uint32_t j = j0 + g;
                ps[h * KT + j] =
                    i < nq && j < n && t0 + j <= qpos[q0 + i]
                        ? d[g] * scale
                        : -INFINITY;
            }
        }
        __syncthreads();
        // Online softmax: warp w takes query-heads w, w + nw, ...
        for (std::uint32_t h = warp; h < H; h += nw) {
            float s = lane < KT ? ps[h * KT + lane] : -INFINITY;
            float mx = s;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            }
            const float m_old = mh[h];
            const float m_new = fmaxf(m_old, mx);
            const float p = s == -INFINITY ? 0.0f : expf(s - m_new);
            float sum = p;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            if (lane < KT) {
                ps[h * KT + lane] = p;
            }
            __syncwarp();
            if (lane == 0) {
                const float r =
                    m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
                rs[h] = r;
                lh[h] = lh[h] * r + sum;
                mh[h] = m_new;
            }
        }
        __syncthreads();
        // V into the same tile buffer.
        for (std::uint32_t e = tid; e < KT * HD; e += nt) {
            const std::uint32_t j = e / HD, d = e % HD;
            kvs[j * KS + d] =
                j < n ? __ldg(kv + rowoff[j] + v_off + d) : 0.0f;
        }
        __syncthreads();
        // Outputs: thread owns (h, d) = e / HD, e % HD for e = tid + k nt
        // (one d, kPer query-heads), so each V float read serves kPer
        // heads; per head the sum runs over j in order, as before.
#pragma unroll
        for (std::uint32_t k = 0; k < kPer; ++k) {
            const std::uint32_t h = (tid + k * nt) / HD;
            if (h < H) {
                acc[k] *= rs[h];
            }
        }
        const std::uint32_t dcol = tid % HD;
        for (std::uint32_t j = 0; j < n; ++j) {
            const float vv = kvs[j * KS + dcol];
#pragma unroll
            for (std::uint32_t k = 0; k < kPer; ++k) {
                const std::uint32_t h = (tid + k * nt) / HD;
                if (h < H) {
                    acc[k] += ps[h * KT + j] * vv;
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (std::uint32_t k = 0; k < kPer; ++k) {
        const std::uint32_t e = tid + k * nt;
        const std::uint32_t h = e / HD, d = e % HD;
        const std::uint32_t i = h / group;
        if (h < H && i < nq) {
            out[(q0 + i) * q_stride + head0 + std::size_t{h % group} * HD +
                d] = acc[k] / lh[h];
        }
    }
}

/** Grow-only device scratch for cuda_kv_attend; callers hold mu. */
struct AttendScratch {
    std::mutex mu;
    void* q = nullptr;
    void* out = nullptr;
    void* blocks = nullptr;
    void* part = nullptr;
    void* meta = nullptr;  // n_q positions, then n_q table offsets
    std::size_t q_bytes = 0, out_bytes = 0, blocks_bytes = 0,
                part_bytes = 0, meta_bytes = 0;

    static bool grow(void*& p, std::size_t& have, std::size_t want) {
        if (want <= have) {
            return true;
        }
        cudaFree(p);
        p = nullptr;
        have = 0;
        if (!cuda_ok(cudaMalloc(&p, want))) {
            p = nullptr;
            return false;
        }
        have = want;
        return true;
    }
};

AttendScratch& attend_scratch() {
    static AttendScratch s;
    return s;
}

}  // namespace

bool cuda_kv_attend(const KvAttend& a) {
    if (a.mirror == nullptr || a.n_q == 0 || a.n_kv_heads == 0 ||
        a.head_dim == 0 || a.block_tokens == 0 ||
        a.n_heads % a.n_kv_heads != 0) {
        return false;
    }
    const std::uint32_t group = a.n_heads / a.n_kv_heads;
    // Each query's position and the start of its sequence's block table:
    // one sequence (base + i, 0) or several (q_pos, q_table). The same
    // numbers give the same bits whichever way they came.
    std::vector<std::uint32_t> meta(std::size_t{a.n_q} * 2);
    std::uint32_t max_pos = 0;
    for (std::uint32_t i = 0; i < a.n_q; ++i) {
        const std::uint32_t pos = a.q_pos ? a.q_pos[i] : a.base + i;
        const std::uint32_t tab = a.q_table ? a.q_table[i] : 0;
        if (std::size_t{tab} + pos / a.block_tokens >= a.n_blocks) {
            return false;  // a position past the block table
        }
        meta[i] = pos;
        meta[a.n_q + i] = tab;
        max_pos = std::max(max_pos, pos + 1);
    }
    // Groups 1 to 8: Qwen2.5-0.5B's is 7 and Llama 3.2 3B's 3 (i#1196).
    if (group < 1 || group > 8 || a.n_heads % a.n_kv_heads != 0 ||
        a.head_dim > kAttendThreads ||
        kAttendThreads % a.head_dim != 0) {
        return false;  // a shape the kernel does not take: host attends
    }
    const std::size_t shared =
        ((std::size_t{group} * (a.head_dim + kAttendChunk + kAttendThreads) +
          1) & ~std::size_t{1}) * sizeof(float) +
        kAttendChunk * sizeof(std::size_t);
    if (shared > 48 * 1024) {
        return false;  // q heads too wide for one block
    }
    const std::uint32_t chunks = (max_pos + kAttendChunk - 1) / kAttendChunk;
    ensure_device();
    AttendScratch& s = attend_scratch();
    std::lock_guard<std::mutex> lk(s.mu);
    const std::size_t width = std::size_t{a.n_heads} * a.head_dim;
    const std::size_t q_floats = (a.n_q - 1) * a.q_stride + width;
    const std::size_t q_bytes = q_floats * sizeof(float);
    if (!AttendScratch::grow(s.q, s.q_bytes, q_bytes) ||
        !AttendScratch::grow(s.out, s.out_bytes, q_bytes) ||
        !AttendScratch::grow(s.blocks, s.blocks_bytes,
                             a.n_blocks * sizeof(std::uint32_t)) ||
        !AttendScratch::grow(s.part, s.part_bytes,
                             std::size_t{a.n_q} * a.n_heads * chunks *
                                 (a.head_dim + 2) * sizeof(float)) ||
        !AttendScratch::grow(s.meta, s.meta_bytes,
                             meta.size() * sizeof(std::uint32_t))) {
        return false;
    }
    // device_io: q and out are the caller's device buffers (a CUDA
    // forward), so neither is copied.
    const float* dq =
        a.device_io ? a.q : static_cast<const float*>(s.q);
    float* dout = a.device_io ? a.out : static_cast<float*>(s.out);
    // Tables the caller already put on the device skip their upload (a
    // pageable upload waits for the queued work).
    const bool dev_tables = a.device_io && a.dev_blocks != nullptr &&
                            a.dev_meta != nullptr;
    if ((!a.device_io &&
         !cuda_ok(cudaMemcpy(s.q, a.q, q_bytes, cudaMemcpyHostToDevice))) ||
        (!dev_tables &&
         (!cuda_ok(cudaMemcpy(s.blocks, a.blocks,
                              a.n_blocks * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice)) ||
          !cuda_ok(cudaMemcpy(s.meta, meta.data(),
                              meta.size() * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice))))) {
        return false;
    }
    const auto* dblocks = static_cast<const std::uint32_t*>(
        dev_tables ? a.dev_blocks : s.blocks);
    const auto* qpos = static_cast<const std::uint32_t*>(
        dev_tables ? a.dev_meta : s.meta);
    const auto* qtab = qpos + a.n_q;
    // A prefill chunk of one sequence (several queries, one block table)
    // takes the flash kernel (i#1254); decode and a batch of sequences
    // keep the two passes below. LOCUS_CUDA_FLASH=0 keeps them for all.
    static const bool flash_on = [] {
        const char* e = std::getenv("LOCUS_CUDA_FLASH");
        return e == nullptr || std::string_view(e) != "0";
    }();
    bool one_seq = a.n_q > 1;
    for (std::uint32_t i = 1; one_seq && i < a.n_q; ++i) {
        one_seq = meta[a.n_q + i] == meta[a.n_q];
    }
    const auto flash = [&](auto kernel, std::uint32_t hd) {
        const std::uint32_t kt = hd <= 64 ? 32 : 16;
        const std::uint32_t h = kFlashHeads / group * group;
        const std::size_t sh =
            (std::size_t{h} * (hd + 1) + std::size_t{kt} * (hd + 1) +
             std::size_t{h} * kt + 3 * std::size_t{h}) * sizeof(float);
        if (sh > 48 * 1024) {
            return false;
        }
        const std::uint32_t qt = kFlashHeads / group;
        // The sequence's table starts at its offset in the blocks array
        // (the queries share it: one_seq), as qtab does for the two-pass
        // kernels; the n_blocks check above covers tab + pos / bt.
        kernel<<<dim3(a.n_kv_heads, (a.n_q + qt - 1) / qt), kAttendThreads,
                 sh>>>(static_cast<const float*>(a.mirror),
                       dblocks + meta[a.n_q],
                       a.block_stride, a.layer_off, a.v_off, a.block_tokens,
                       a.n_kv_heads * a.head_dim, a.kq_scale, qpos, a.n_q,
                       group, dq, a.q_stride, dout);
        return true;
    };
    bool flashed = false;
    if (flash_on && one_seq) {
        if (a.head_dim == 64) {
            flashed = flash(gqa_attend_flash<64>, 64);
        } else if (a.head_dim == 128) {
            flashed = flash(gqa_attend_flash<128>, 128);
        }
    }
    const auto partial = [&](auto kernel) {
        kernel<<<dim3(a.n_kv_heads, a.n_q, chunks), kAttendThreads,
                 shared>>>(
            static_cast<const float*>(a.mirror), dblocks, a.block_stride,
            a.layer_off, a.v_off, a.block_tokens, a.n_kv_heads * a.head_dim,
            a.head_dim, a.kq_scale, qpos, qtab, dq, a.q_stride,
            static_cast<float*>(s.part));
    };
    if (!flashed) {
        switch (group) {
            case 1: partial(gqa_attend_partial<1>); break;
            case 2: partial(gqa_attend_partial<2>); break;
            case 3: partial(gqa_attend_partial<3>); break;
            case 4: partial(gqa_attend_partial<4>); break;
            case 5: partial(gqa_attend_partial<5>); break;
            case 6: partial(gqa_attend_partial<6>); break;
            case 7: partial(gqa_attend_partial<7>); break;
            case 8: partial(gqa_attend_partial<8>); break;
            default: return false;  // checked above; other sizes: host
        }
        gqa_attend_combine<<<dim3(a.n_kv_heads, a.n_q), kAttendThreads>>>(
            static_cast<const float*>(s.part), a.head_dim, group, qpos,
            chunks, dout, a.q_stride);
    }
    if (!cuda_ok(cudaGetLastError())) {
        return false;
    }
    if (a.device_io) {
        return true;  // left on the device; the caller syncs
    }
    // Row by row: the gaps between q_stride rows stay the caller's.
    const std::size_t pitch = a.q_stride * sizeof(float);
    return cuda_ok(cudaMemcpy2D(a.out, pitch, s.out, pitch,
                                width * sizeof(float), a.n_q,
                                cudaMemcpyDeviceToHost));
}

}  // namespace locus::backend
