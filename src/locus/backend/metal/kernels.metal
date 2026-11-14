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
