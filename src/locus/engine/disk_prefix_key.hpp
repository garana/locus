#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "locus/engine/disk_prefix_store.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/kv_quant.hpp"
#include "locus/model/transformer.hpp"

namespace locus::engine {

/**
 * The disk prefix cache key for model `m`, loaded from `g` (the file
 * at `path`), with KV type `kv_type` (i#416).
 *
 * The model part hashes every shard's GGUF header, metadata, tensor
 * table and size, with the first file's mtime: hashing the weights
 * would take about a minute for a 10 GB model on a slow disk, and the
 * mtime makes a file replaced under the same name a miss. The override
 * part hashes what changes K or V bits besides the weights: context
 * length, RoPE and YaRN settings, Q8_K activations, the device name
 * and, on Vulkan, f16 prefill staging and the batch-invariant kernels.
 * The backend is the one running `m`, so call this after use_backend
 * and the Vulkan mode setters.
 *
 * @param[out] block_bytes Bytes of one KV block of this geometry.
 * @throws std::filesystem::filesystem_error when `path` cannot be
 *     stat'ed.
 */
DiskKey disk_key_for(const gguf::GgufFile& g, const std::string& path,
                     const model::TransformerModel& m, kv::KvType kv_type,
                     std::size_t& block_bytes);

/**
 * Opens the disk prefix store under `dir` for `m` (disk_key_for), with
 * a cap of `cap_bytes`.
 */
std::shared_ptr<DiskPrefixStore> open_disk_prefix(
    const std::string& dir, std::uint64_t cap_bytes,
    const gguf::GgufFile& g, const std::string& path,
    const model::TransformerModel& m, kv::KvType kv_type);

}  // namespace locus::engine
