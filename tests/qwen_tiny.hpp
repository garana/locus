#pragma once

// Tiny Qwen2.5-VL models for the synthetic goldens: the language model
// of scripts/make-qwen-mrope-golden.py and the vision tower of
// scripts/make-qwen-vision-goldens.py synth, built as GGUF images in
// memory and filled from the scripts' LCG in sorted tensor-name order.
// Keep the sizes in sync with the scripts.

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "gguf_builder.hpp"

namespace qwen_tiny {

using Shape = std::vector<std::uint64_t>;

// The language model (make-qwen-mrope-golden.py).
constexpr std::uint32_t kE = 64, kHeads = 4, kKv = 2, kFf = 96,
                        kLayers = 2, kVocab = 64;
constexpr std::int32_t kSections[] = {2, 3, 3, 0};
constexpr std::int32_t kImageId = 60, kStartId = 61;

// The vision tower (make-qwen-vision-goldens.py SYNTH).
constexpr std::uint32_t kVw = 32, kVheads = 2, kVffn = 40, kVdepth = 4,
                        kVwa = 2;

/** The scripts' LCG: n values, advancing state. */
inline std::vector<float> lcg(std::uint32_t& state, std::size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * 0.5f;
    }
    return v;
}

/** The GGUF image of `shapes`, F32, LCG-filled from seed 12345 in
 *  sorted name order; `kvs` writes the n_kv metadata entries. */
template <class Kvs>
std::vector<std::byte> build(const std::map<std::string, Shape>& shapes,
                             std::uint64_t n_kv, Kvs&& kvs) {
    std::uint32_t state = 12345;
    std::map<std::string, std::vector<float>> data;
    for (const auto& [name, shape] : shapes) {
        std::uint64_t n = 1;
        for (auto d : shape) n *= d;
        data[name] = lcg(state, n);
    }
    GgufBuilder g;
    g.header(data.size(), n_kv);
    kvs(g);
    std::uint64_t offset = 0;
    for (const auto& [name, v] : data) {
        const Shape& s = shapes.at(name);
        const std::vector<std::uint64_t> dims(s.rbegin(), s.rend());
        g.tensor(name, dims, 0 /* F32 */, offset);
        offset = (offset + v.size() * 4 + 31) & ~31ull;
    }
    g.pad();
    for (const auto& [name, v] : data) {
        for (float x : v) g.f32(x);
        while (g.bytes().size() % 32 != 0) g.zeros(1);
    }
    auto span = g.bytes();
    return {span.begin(), span.end()};
}

/** The tiny language model under `arch` (qwen2vl, or qwen2 without the
 *  rope sections), with `vocab` rows (the real tokenizer's 151936 for
 *  the pipeline test). */
inline std::vector<std::byte> lm(const std::string& arch = "qwen2vl",
                                 std::uint64_t vocab = kVocab) {
    const std::uint64_t hd = kE / kHeads;
    std::map<std::string, Shape> shapes;
    shapes["token_embd.weight"] = {vocab, kE};
    shapes["output_norm.weight"] = {kE};
    shapes["output.weight"] = {vocab, kE};
    for (std::uint32_t i = 0; i < kLayers; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        shapes[b + "attn_norm.weight"] = {kE};
        shapes[b + "ffn_norm.weight"] = {kE};
        shapes[b + "attn_q.weight"] = {kE, kE};
        shapes[b + "attn_q.bias"] = {kE};
        for (const char* t : {"attn_k", "attn_v"}) {
            shapes[b + t + ".weight"] = {kKv * hd, kE};
            shapes[b + t + ".bias"] = {kKv * hd};
        }
        shapes[b + "attn_output.weight"] = {kE, kE};
        shapes[b + "ffn_gate.weight"] = {kFf, kE};
        shapes[b + "ffn_up.weight"] = {kFf, kE};
        shapes[b + "ffn_down.weight"] = {kE, kFf};
    }
    const bool vl = arch == "qwen2vl";
    return build(shapes, vl ? 10 : 9, [&](GgufBuilder& g) {
        g.kv_string("general.architecture", arch);
        g.kv_u32(arch + ".embedding_length", kE);
        g.kv_u32(arch + ".block_count", kLayers);
        g.kv_u32(arch + ".attention.head_count", kHeads);
        g.kv_u32(arch + ".attention.head_count_kv", kKv);
        g.kv_u32(arch + ".feed_forward_length", kFf);
        g.kv_u32(arch + ".context_length", 128);
        g.kv_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        g.kv_f32(arch + ".rope.freq_base", 10000.0f);
        if (vl) g.kv_i32_array(arch + ".rope.dimension_sections", kSections);
    });
}

/** The tiny tower as an mmproj GGUF image, its merger projecting to
 *  `out` (24 in the vision golden, kE to feed the language model). */
inline std::vector<std::byte> mmproj(std::uint32_t out,
                                     const char* projector =
                                         "qwen2.5vl_merger") {
    std::map<std::string, Shape> shapes;
    const std::uint64_t w = kVw, f = kVffn, m = 4 * kVw;
    shapes["v.patch_embd.weight"] = {w, 3, 14, 14};
    shapes["v.patch_embd.weight.1"] = {w, 3, 14, 14};
    shapes["v.post_ln.weight"] = {w};
    shapes["mm.0.weight"] = {m, m};
    shapes["mm.0.bias"] = {m};
    shapes["mm.2.weight"] = {out, m};
    shapes["mm.2.bias"] = {out};
    for (std::uint32_t i = 0; i < kVdepth; ++i) {
        const std::string b = "v.blk." + std::to_string(i) + ".";
        shapes[b + "ln1.weight"] = {w};
        shapes[b + "ln2.weight"] = {w};
        for (const char* t : {"attn_q", "attn_k", "attn_v", "attn_out"}) {
            shapes[b + t + ".weight"] = {w, w};
            shapes[b + t + ".bias"] = {w};
        }
        for (const char* t : {"ffn_gate", "ffn_up"}) {
            shapes[b + t + ".weight"] = {f, w};
            shapes[b + t + ".bias"] = {f};
        }
        shapes[b + "ffn_down.weight"] = {w, f};
        shapes[b + "ffn_down.bias"] = {w};
    }
    return build(shapes, 12, [&](GgufBuilder& g) {
        g.kv_string("general.architecture", "clip");
        g.kv_string("clip.projector_type", projector);
        g.kv_u32("clip.vision.patch_size", 14);
        g.kv_u32("clip.vision.embedding_length", kVw);
        g.kv_u32("clip.vision.feed_forward_length", kVffn);
        g.kv_u32("clip.vision.attention.head_count", kVheads);
        g.kv_u32("clip.vision.projection_dim", out);
        g.kv_u32("clip.vision.block_count", kVdepth);
        g.kv_u32("clip.vision.n_wa_pattern", kVwa);
        g.kv_f32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
        const float mean[] = {0.5f, 0.5f, 0.5f}, sd[] = {0.25f, 0.25f, 0.25f};
        g.kv_f32_array("clip.vision.image_mean", mean);
        g.kv_f32_array("clip.vision.image_std", sd);
    });
}

}  // namespace qwen_tiny
