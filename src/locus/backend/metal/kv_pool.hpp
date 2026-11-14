#pragma once

#include <cstddef>

#include "locus/backend/metal/context.hpp"

namespace locus::backend::metal {

/**
 * A KV pool in a shared Metal buffer (Ops.alloc_kv, i#866): the paged
 * cache's host pointer is the buffer's contents, so the CPU path and
 * the GPU forward read and write the same memory.
 * @returns The zeroed pool, or nullptr when Metal is not usable or the
 *     device cannot allocate it.
 */
float* alloc_kv(std::size_t n_floats);

/** Releases a pool alloc_kv returned (Ops.free_kv). */
void free_kv(float* pool);

/**
 * @returns The buffer of the alloc_kv pool that starts at `pool`, or
 *     nullptr when `pool` is not one. The caller holds device_mutex()
 *     for as long as it uses the buffer.
 */
const Context::Buffer* kv_buffer(const float* pool);

}  // namespace locus::backend::metal
