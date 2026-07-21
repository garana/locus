#pragma once

#include <string>

#include "locus/gguf/gguf.hpp"
#include "locus/model/llama.hpp"

/** Shared loader helpers for arch hooks and the model loader. */
namespace locus::model::load_util {

inline std::uint32_t need_uint(const gguf::GgufFile& g,
                               const std::string& key) {
    auto v = g.get_uint(key);
    if (!v) {
        throw gguf::Error("missing metadata: " + key);
    }
    return static_cast<std::uint32_t>(*v);
}

inline float get_float(const gguf::GgufFile& g,
                       const std::string& key, float fallback) {
    const gguf::Value* v = g.find(key);
    if (v == nullptr) {
        return fallback;
    }
    if (v->type != gguf::ValueType::kFloat32 &&
        v->type != gguf::ValueType::kFloat64) {
        throw gguf::Error(key + " is not a float");
    }
    return static_cast<float>(v->f);
}

/** Fetches a tensor as a Mat, enforcing its 2-D shape. */
inline backend::Mat need_mat(const gguf::GgufFile& g,
                             const std::string& name,
                             std::uint32_t cols,
                             std::uint32_t rows) {
    const gguf::TensorInfo* t = g.find_tensor(name);
    if (t == nullptr) {
        throw gguf::Error("missing tensor: " + name);
    }
    if (t->ne[0] != cols || t->ne[1] != rows || t->ne[2] != 1 ||
        t->ne[3] != 1) {
        throw gguf::Error("unexpected shape for tensor: " + name);
    }
    return backend::Mat{t->type, g.tensor_data(*t).data(), rows,
                        cols};
}

/** Fetches a 3-D expert tensor, enforcing its shape. */
inline LlamaModel::ExpertMat need_mat3(const gguf::GgufFile& g,
                                       const std::string& name,
                                       std::uint32_t cols,
                                       std::uint32_t rows,
                                       std::uint32_t n_expert) {
    const gguf::TensorInfo* t = g.find_tensor(name);
    if (t == nullptr) {
        throw gguf::Error("missing tensor: " + name);
    }
    if (t->ne[0] != cols || t->ne[1] != rows ||
        t->ne[2] != n_expert || t->ne[3] != 1) {
        throw gguf::Error("unexpected shape for tensor: " + name);
    }
    LlamaModel::ExpertMat em;
    em.base = backend::Mat{t->type, g.tensor_data(*t).data(),
                           rows, cols};
    em.expert_bytes = t->nbytes / n_expert;
    return em;
}

/** Fetches a 1-D F32 tensor (norm weights). */
inline std::span<const float> need_vec(const gguf::GgufFile& g,
                                       const std::string& name,
                                       std::uint32_t len) {
    const gguf::TensorInfo* t = g.find_tensor(name);
    if (t == nullptr) {
        throw gguf::Error("missing tensor: " + name);
    }
    if (t->type != gguf::TensorType::kF32 || t->ne[0] != len ||
        t->nelements() != len) {
        throw gguf::Error("norm tensor must be F32 1-D: " + name);
    }
    return {reinterpret_cast<const float*>(
                g.tensor_data(*t).data()),
            len};
}

}  // namespace locus::model::load_util
