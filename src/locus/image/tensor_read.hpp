#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/gguf/gguf.hpp"

/** Tensor lookups the image and embedding model loaders share. */
namespace locus::image {

/**
 * @param g The model's GGUF.
 * @param model The model's name, the prefix of the error.
 * @param name The tensor's name.
 * @returns The tensor's info, never null.
 * @throws gguf::Error "<model>: missing tensor <name>" when it is absent.
 */
inline const gguf::TensorInfo* require_tensor(const gguf::GgufFile& g,
                                              std::string_view model,
                                              const std::string& name) {
    const gguf::TensorInfo* ti = g.find_tensor(name);
    if (ti == nullptr) {
        throw gguf::Error(std::string(model) + ": missing tensor " + name);
    }
    return ti;
}

/**
 * A small F32, F16 or BF16 tensor (a norm weight, a bias), as floats.
 *
 * @param g The model's GGUF.
 * @param model The model's name, the prefix of the error.
 * @param name The tensor's name.
 * @param n The number of values it must have.
 * @returns Its n values, widened to F32.
 * @throws gguf::Error when it is absent, has another count, or another
 *     type.
 */
inline std::vector<float> read_floats(const gguf::GgufFile& g,
                                      std::string_view model,
                                      const std::string& name,
                                      std::size_t n) {
    const gguf::TensorInfo* ti = require_tensor(g, model, name);
    const bool f32 = ti->type == gguf::TensorType::kF32;
    const bool f16 = ti->type == gguf::TensorType::kF16;
    const bool bf16 = ti->type == gguf::TensorType::kBF16;
    if ((!f32 && !f16 && !bf16) || ti->nelements() != n) {
        throw gguf::Error(std::string(model) + ": " + name +
                          " must be F32, F16 or BF16 with " +
                          std::to_string(n) + " values");
    }
    const auto bytes = g.tensor_data(*ti);
    std::vector<float> out(n);
    if (f32) {
        std::memcpy(out.data(), bytes.data(), n * sizeof(float));
        return out;
    }
    for (std::size_t i = 0; i < n; ++i) {
        std::uint16_t h;
        std::memcpy(&h, bytes.data() + 2 * i, 2);
        if (f16) {
            out[i] = backend::f16_to_f32(h);
        } else {
            const std::uint32_t u = std::uint32_t{h} << 16;
            std::memcpy(&out[i], &u, sizeof(float));
        }
    }
    return out;
}

}  // namespace locus::image
