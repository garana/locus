#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"

namespace locus::image {

/**
 * One LoRA's change to one linear layer W (out x in): y = W x + b plus
 * strength * B (A x), with A (r x in) and B (out x r, alpha / r folded
 * in by scripts/convert-lora.py), both F32.
 */
struct LoraPart {
    backend::Mat a;  ///< r x in
    backend::Mat b;  ///< out x r
    float strength = 1.0f;
};

/**
 * The LoRAs of one request (i#372): "flux_lora" GGUFs, each at its own
 * strength, keyed by the DiT layer each part changes (the weight's
 * name without ".weight"). Several LoRAs on one layer add up. The
 * matrices stay in the GGUF mappings, so each GgufFile must outlive
 * the set and every model it is attached to.
 */
class LoraSet {
  public:
    /**
     * Adds one LoRA.
     *
     * @param g A GGUF whose general.architecture is "flux_lora".
     * @param strength The user's scale on top of alpha / r.
     * @throws gguf::Error on a wrong architecture, a missing or unpaired
     *     tensor, or a shape that is not r x in and out x r.
     */
    void add(const gguf::GgufFile& g, float strength);

    /** @returns The parts on layer, empty when no LoRA changes it. */
    std::span<const LoraPart> parts(std::string_view layer) const;

    /** @returns Every layer some LoRA changes, sorted. */
    std::vector<std::string> layers() const;

    /** @returns Whether no LoRA was added. */
    bool empty() const { return by_layer_.empty(); }

  private:
    std::map<std::string, std::vector<LoraPart>, std::less<>> by_layer_;
};

/** One LoRA a request asks for, by name, at a strength. */
struct LoraPick {
    std::string name;
    float strength = 1.0f;
};

/**
 * The LoRAs a server offers (--lora-dir): every "*.gguf" in one
 * directory, each a "flux_lora" GGUF named by its file name without
 * ".gguf". The files stay mapped for the library's life.
 */
class LoraLibrary {
  public:
    /**
     * @throws gguf::Error when the directory holds no "*.gguf", or one
     *     of them is not a usable flux_lora (wrong architecture, an
     *     unpaired tensor, mismatched ranks).
     * @throws std::filesystem::filesystem_error when it cannot be read.
     */
    static std::unique_ptr<LoraLibrary> load_dir(const std::string& dir);

    /** @returns The LoRA names, sorted. */
    std::vector<std::string> names() const;

    /**
     * @returns A set of the picked LoRAs at their strengths; it points
     *     into this library, which must outlive it.
     * @throws std::invalid_argument naming an unknown LoRA and listing
     *     the known ones.
     */
    LoraSet select(std::span<const LoraPick> picks) const;

  private:
    std::map<std::string, std::unique_ptr<gguf::GgufFile>, std::less<>>
        files_;
};

/**
 * out += sum of part.strength * B (A x) over parts, for n token-major
 * rows: x is n x in, out n x out. Runs on op, as the layer's own
 * matrix does.
 */
void apply_lora(const backend::Ops& op, std::span<const LoraPart> parts,
                std::span<const float> x, std::uint32_t n,
                std::span<float> out);

}  // namespace locus::image
