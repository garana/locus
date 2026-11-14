// The Metal Q4_K prefill matmul (i#806); see metal_matmul.hpp.
#include "locus/backend/vulkan/metal_matmul.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace locus::backend::vk {

namespace {

// Y = scale * W X (+ Y), W in Q4_K rows, X and Y f32 token rows. A
// threadgroup of 128 threads (4 simdgroups) takes 64 weight rows x 32
// tokens and walks k in stages of 32 (one Q4_K sub-block): the threads
// dequantize the weight tile and stage the activations in threadgroup
// memory (float, so the products are f32), then each simdgroup runs a
// 32 x 16 piece as eight 8 x 8 simdgroup_matrix tiles. Rows and tokens
// past the matrix load as zero and are not stored, so any shape works.
constexpr const char* kSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct BlockQ4K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

struct Params {
    ulong w_off;     // bytes
    ulong x_off;     // floats
    ulong out_off;   // floats
    uint rows, cols, n_tok;
    uint x_stride, out_stride;
    uint accumulate;
    float scale;
};

constant constexpr uint TM = 64;
constant constexpr uint TN = 32;
constant constexpr uint TK = 32;
constant constexpr uint PA = TK + 8;
constant constexpr uint PB = TK + 8;
constant constexpr uint PO = TM + 4;

// Sub-block j's 6-bit scale and min (ggml's get_scale_min_k4).
inline void scale_min(uint j, device const uchar* q, thread uint& sc,
                      thread uint& mn) {
    if (j < 4) {
        sc = q[j] & 63;
        mn = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

kernel void mm_q4k(device const uchar* wbase [[buffer(0)]],
                   device const float* xbase [[buffer(1)]],
                   device float* ybase [[buffer(2)]],
                   constant Params& p [[buffer(3)]],
                   uint2 tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[TM * PA];
    threadgroup float sb[TN * PB];
    threadgroup float so[TN * PO];
    device const BlockQ4K* w = (device const BlockQ4K*)(wbase + p.w_off);
    device const float* x = xbase + p.x_off;
    device float* y = ybase + p.out_off;
    const uint m0 = tg.x * TM;
    const uint n0 = tg.y * TN;
    const uint nb = p.cols / 256;
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
    const bool b_in = n0 + bt < p.n_tok;
    device const BlockQ4K* wrow = w + (a_in ? (m0 + ar) * nb : 0);
    device const float* xrow = x + (b_in ? (n0 + bt) * p.x_stride : 0);

    for (uint k0 = 0; k0 < p.cols; k0 += TK) {
        threadgroup float* da = sa + ar * PA + ah;
        if (a_in) {
            device const BlockQ4K& blk = wrow[k0 / 256];
            const uint j = (k0 % 256) / 32;
            uint sc, mn;
            scale_min(j, blk.scales, sc, mn);
            const float d = float(blk.d) * float(sc);
            const float mm = float(blk.dmin) * float(mn);
            device const uchar* q = blk.qs + 32 * (j / 2) + ah;
            const uint sh = (j & 1) * 4;
            for (uint i = 0; i < 16; ++i) {
                da[i] = d * float((q[i] >> sh) & 0xF) - mm;
            }
        } else {
            for (uint i = 0; i < 16; ++i) da[i] = 0.0f;
        }
        threadgroup float* db = sb + bt * PB + bk;
        if (b_in) {
            device const float* src = xrow + k0 + bk;
            for (uint i = 0; i < 8; ++i) db[i] = src[i];
        } else {
            for (uint i = 0; i < 8; ++i) db[i] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < TK; kk += 8) {
            simdgroup_float8x8 a[4], bm[2];
            for (uint i = 0; i < 4; ++i)
                simdgroup_load(a[i], sa + (rm + 8 * i) * PA + kk, PA);
            // B is k x tokens: the [token][k] tile, loaded transposed.
            for (uint j = 0; j < 2; ++j)
                simdgroup_load(bm[j], sb + (rn + 8 * j) * PB + kk, PB,
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
            simdgroup_store(acc[i][j], so + (rn + 8 * j) * PO + rm + 8 * i,
                            PO, ulong2(0, 0), true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < TM * TN; e += 128) {
        const uint t = e / TM, r = e % TM;
        if (n0 + t < p.n_tok && m0 + r < p.rows) {
            device float* o = y + (n0 + t) * p.out_stride + m0 + r;
            const float v = p.scale * so[t * PO + r];
            *o = p.accumulate != 0 ? *o + v : v;
        }
    }
}
)MSL";

struct Params {
    std::uint64_t w_off, x_off, out_off;
    std::uint32_t rows, cols, n_tok;
    std::uint32_t x_stride, out_stride;
    std::uint32_t accumulate;
    float scale;
};

class MetalMatmulImpl final : public MetalMatmul {
public:
    MetalMatmulImpl(id<MTLCommandQueue> q, id<MTLComputePipelineState> p)
        : queue_(q), pso_(p) {}

    void commit_q4k(const MetalQ4kArgs& a) override {
        @autoreleasepool {
            const Params p{a.w_off,      a.x_off,    a.out_off,
                           a.rows,       a.cols,     a.n_tok,
                           a.x_stride,   a.out_stride,
                           a.accumulate ? 1u : 0u, a.scale};
            // The buffers outlive the command buffer (the Vulkan side
            // frees them only after its batch completes), so it need
            // not retain them.
            id<MTLCommandBuffer> cb =
                [queue_ commandBufferWithUnretainedReferences];
            id<MTLSharedEvent> ev = (__bridge id<MTLSharedEvent>)a.event;
            if (ev != nil) {
                [cb encodeWaitForEvent:ev value:a.wait_value];
            }
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso_];
            [enc setBuffer:(__bridge id<MTLBuffer>)a.w offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)a.x offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)a.out offset:0 atIndex:2];
            [enc setBytes:&p length:sizeof p atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((a.rows + 63) / 64,
                                                  (a.n_tok + 31) / 32, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            if (ev != nil) {
                [cb encodeSignalEvent:ev value:a.signal_value];
            }
            [cb commit];
        }
    }

private:
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> pso_;
};

}  // namespace

std::unique_ptr<MetalMatmul> MetalMatmul::create(void* mtl_device,
                                                 void* mtl_queue) {
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)mtl_device;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)mtl_queue;
        if (dev == nil || q == nil) {
            return nullptr;
        }
        MTLCompileOptions* opt = [MTLCompileOptions new];
        opt.mathMode = MTLMathModeSafe;
        NSError* err = nil;
        id<MTLLibrary> lib =
            [dev newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                              options:opt
                                error:&err];
        if (lib == nil) {
            NSLog(@"locus: metal matmul: %@", err);
            return nullptr;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"mm_q4k"];
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:fn error:&err];
        if (pso == nil) {
            NSLog(@"locus: metal matmul: %@", err);
            return nullptr;
        }
        return std::make_unique<MetalMatmulImpl>(q, pso);
    }
}

}  // namespace locus::backend::vk
