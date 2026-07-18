#include <unordered_map>

#include "cppllm/backend/variants.hpp"
#include "cppllm/backend/vulkan/context.hpp"

namespace cppllm::backend {

namespace {

/**
 * Process-lifetime GPU state for the vulkan backend: one context,
 * weight buffers uploaded on first use (keyed by the mmap'd
 * weight pointer, which is stable for the model's lifetime), and
 * grow-only scratch buffers for activations.
 *
 * Single-threaded by contract: ops are called only from the
 * engine/model thread.
 */
struct State {
    vk::VulkanContext ctx;
    std::unordered_map<const std::byte*, vk::VulkanContext::Buffer>
        weights;
    vk::VulkanContext::Buffer x{}, out{};
    std::size_t x_cap = 0, out_cap = 0;

    vk::VulkanContext::Buffer weight_buffer(const Mat& w) {
        auto it = weights.find(w.data);
        if (it != weights.end()) {
            return it->second;
        }
        const std::size_t bytes =
            static_cast<std::size_t>(w.rows) * w.cols *
            sizeof(float);
        auto buf = ctx.create_buffer(bytes);
        ctx.write_buffer(
            buf, {w.data, bytes});
        weights.emplace(w.data, buf);
        return buf;
    }

    void ensure_scratch(std::size_t x_bytes,
                        std::size_t out_bytes) {
        if (x_bytes > x_cap) {
            if (x_cap != 0) {
                ctx.destroy_buffer(x);
            }
            x = ctx.create_buffer(x_bytes);
            x_cap = x_bytes;
        }
        if (out_bytes > out_cap) {
            if (out_cap != 0) {
                ctx.destroy_buffer(out);
            }
            out = ctx.create_buffer(out_bytes);
            out_cap = out_bytes;
        }
    }
};

State& state() {
    static State s;
    return s;
}

}  // namespace

bool vulkan_backend_usable() {
    static const bool ok = vk::VulkanContext::available();
    return ok;
}

void matvec_vulkan(const Mat& w, std::span<const float> x,
                   std::span<float> out) {
    if (w.type != gguf::TensorType::kF32) {
        // Quantized/f16 weight kernels are still CPU (registry
        // description says so); scalar keeps full coverage.
        matvec(w, x, out);
        return;
    }
    State& s = state();
    auto wb = s.weight_buffer(w);
    s.ensure_scratch(x.size_bytes(), out.size_bytes());
    s.ctx.write_buffer(s.x, std::as_bytes(x));
    s.ctx.matvec_f32(wb, w.rows, w.cols, s.x, s.out);
    s.ctx.read_buffer(s.out, std::as_writable_bytes(out));
}

}  // namespace cppllm::backend
