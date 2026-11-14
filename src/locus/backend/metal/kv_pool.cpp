// KV pools in shared Metal buffers (i#866); see kv_pool.hpp.
#include "locus/backend/metal/kv_pool.hpp"

#include <exception>
#include <mutex>
#include <unordered_map>

namespace locus::backend::metal {

namespace {

// Leaked, as device_mutex(): a cache freed at exit still finds it.
std::unordered_map<const float*, Context::Buffer>& pools() {
    static auto* p = new std::unordered_map<const float*, Context::Buffer>;
    return *p;
}

}  // namespace

float* alloc_kv(std::size_t n_floats) {
    if (!Context::usable()) {
        return nullptr;
    }
    std::lock_guard lk(device_mutex());
    Context::Buffer b;
    try {
        b = Context::get().alloc(n_floats * sizeof(float));  // zeroed
    } catch (const std::exception&) {
        return nullptr;  // the cache then allocates on the heap
    }
    auto* p = static_cast<float*>(b.data);
    pools().emplace(p, b);
    return p;
}

void free_kv(float* pool) {
    if (pool == nullptr) {
        return;
    }
    std::lock_guard lk(device_mutex());
    const auto it = pools().find(pool);
    if (it != pools().end()) {
        Context::get().free(it->second);
        pools().erase(it);
    }
}

const Context::Buffer* kv_buffer(const float* pool) {
    const auto it = pools().find(pool);
    return it == pools().end() ? nullptr : &it->second;
}

}  // namespace locus::backend::metal
