#include "locus/image/lora.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

#include "locus/image/dense.hpp"

namespace locus::image {

namespace {

constexpr std::string_view kA = ".lora_a", kB = ".lora_b";

/** @returns name's F32 matrix, rows x cols as GGUF lists them. */
backend::Mat mat(const gguf::GgufFile& g, const gguf::TensorInfo& t) {
    if (t.type != gguf::TensorType::kF32 || t.n_dims != 2 ||
        t.ne[0] == 0 || t.ne[1] == 0 || t.ne[0] > (1u << 20) ||
        t.ne[1] > (1u << 20)) {
        throw gguf::Error("lora: " + t.name + " must be a 2D F32 matrix");
    }
    return backend::Mat{t.type, g.tensor_data(t).data(),
                        static_cast<std::uint32_t>(t.ne[1]),
                        static_cast<std::uint32_t>(t.ne[0])};
}

}  // namespace

void LoraSet::add(const gguf::GgufFile& g, float strength) {
    if (g.get_string("general.architecture") != "flux_lora") {
        throw gguf::Error("lora: general.architecture is not flux_lora");
    }
    std::map<std::string, LoraPart, std::less<>> found;
    std::map<std::string, int, std::less<>> seen;  // 1 = A, 2 = B
    for (const gguf::TensorInfo& t : g.tensors()) {
        const std::string_view n = t.name;
        const bool is_a = n.ends_with(kA), is_b = n.ends_with(kB);
        if (!is_a && !is_b) {
            throw gguf::Error("lora: unexpected tensor " + t.name);
        }
        const std::string layer(n.substr(0, n.size() - kA.size()));
        LoraPart& p = found[layer];
        (is_a ? p.a : p.b) = mat(g, t);
        seen[layer] |= is_a ? 1 : 2;
    }
    if (found.empty()) throw gguf::Error("lora: no tensors");
    for (auto& [layer, p] : found) {
        if (seen[layer] != 3) {
            throw gguf::Error("lora: " + layer + " needs lora_a and lora_b");
        }
        if (p.a.rows != p.b.cols) {
            throw gguf::Error("lora: " + layer +
                              " has A and B of different ranks");
        }
        p.strength = strength;
    }
    // Only now, so a bad file leaves the set as it was.
    for (auto& [layer, p] : found) by_layer_[layer].push_back(p);
}

std::span<const LoraPart> LoraSet::parts(std::string_view layer) const {
    const auto it = by_layer_.find(layer);
    if (it == by_layer_.end()) return {};
    return it->second;
}

std::vector<std::string> LoraSet::layers() const {
    std::vector<std::string> out;
    for (const auto& [layer, p] : by_layer_) out.push_back(layer);
    return out;
}

std::unique_ptr<LoraLibrary> LoraLibrary::load_dir(const std::string& dir) {
    auto lib = std::make_unique<LoraLibrary>();
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (!e.is_regular_file() || e.path().extension() != ".gguf") {
            continue;
        }
        auto g = std::make_unique<gguf::GgufFile>(
            gguf::GgufFile::open(e.path().string()));
        LoraSet check;  // refuses a file that is not a usable flux_lora
        try {
            check.add(*g, 1.0f);
        } catch (const gguf::Error& err) {
            throw gguf::Error(e.path().string() + ": " + err.what());
        }
        lib->files_[e.path().stem().string()] = std::move(g);
    }
    if (lib->files_.empty()) {
        throw gguf::Error("lora: no *.gguf in " + dir);
    }
    return lib;
}

std::vector<std::string> LoraLibrary::names() const {
    std::vector<std::string> out;
    for (const auto& [name, g] : files_) out.push_back(name);
    return out;
}

LoraSet LoraLibrary::select(std::span<const LoraPick> picks) const {
    LoraSet set;
    for (const LoraPick& p : picks) {
        const auto it = files_.find(p.name);
        if (it == files_.end()) {
            std::string known;
            for (const auto& [name, g] : files_) {
                known += (known.empty() ? "" : ", ") + name;
            }
            throw std::invalid_argument("unknown lora \"" + p.name +
                                        "\"; this server has: " + known);
        }
        set.add(*it->second, p.strength);
    }
    return set;
}

void apply_lora(const backend::Ops& op, std::span<const LoraPart> parts,
                std::span<const float> x, std::uint32_t n,
                std::span<float> out) {
    for (const LoraPart& p : parts) {
        std::vector<float> ax(std::size_t{n} * p.a.rows);
        std::vector<float> bax(std::size_t{n} * p.b.rows);
        dense::linear(op, p.a, {}, x, n, ax);
        dense::linear(op, p.b, {}, ax, n, bax);
        for (std::size_t i = 0; i < bax.size(); ++i) {
            out[i] += p.strength * bax[i];
        }
    }
}

}  // namespace locus::image
