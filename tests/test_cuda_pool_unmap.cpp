#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"

using locus::model::TransformerModel;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/models/";

/** The logits after feeding `n` tokens (1, 2, ...) on the model's
 *  backend. */
std::vector<float> run(const TransformerModel& model, int n) {
    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(model.hparams().n_vocab);
    for (int t = 0; t < n; ++t) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(1 + t, cache, seq, ws, logits);
    }
    cache.release(seq);
    return logits;
}

}  // namespace

// i#1164: the CUDA weight pool keys pages by host address and lived
// for the process, so a freed model's pages held device memory (about
// 800 MB of a 2 GB card after Llama 3.2 1B) and a model mapped later at
// a reused address could be served the old pages. An unmap hook now
// drops a file's pages when it is unmapped.
TEST_CASE("cuda drops a model file's pages when it is unmapped",
          "[cuda][e2e]") {
    const auto* cuda = locus::backend::find_backend("cuda");
    if (cuda == nullptr || !cuda->available) {
        SKIP("no usable CUDA device");
    }
    const std::string a_path = kDir + "llama-3.2-1b-q4_k_m.gguf";
    const std::string b_path = kDir + "llama-3.2-1b-q8_0.gguf";
    if (!std::filesystem::exists(a_path) || !std::filesystem::exists(b_path)) {
        SKIP("llama-3.2-1b q4_k_m or q8_0 absent");
    }
    locus::backend::cuda_pool_reset();
    REQUIRE(locus::backend::cuda_pool_resident().first == 0);
    {
        auto g = locus::gguf::GgufFile::open(a_path);
        auto model = TransformerModel::load(g);
        model.use_backend(*cuda);
        run(model, 3);
        const auto [pages, bytes] = locus::backend::cuda_pool_resident();
        INFO("resident while A runs: " << pages << " pages, "
                                       << (bytes >> 20) << " MB");
        CHECK(pages > 0);
    }
    // A's GGUF is unmapped: none of its pages stay.
    CHECK(locus::backend::cuda_pool_resident() ==
          std::pair<std::size_t, std::size_t>{0, 0});

    // Another model of other contents (mapped wherever the kernel puts
    // it, A's range included) gives its own output on CUDA: the CPU's
    // within quantization noise, the same top token.
    auto g = locus::gguf::GgufFile::open(b_path);
    auto model = TransformerModel::load(g);
    model.use_backend(locus::backend::best_backend());
    const auto cpu = run(model, 3);
    model.use_backend(*cuda);
    const auto gpu = run(model, 3);
    REQUIRE(cpu.size() == gpu.size());
    double err = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        err += (double{gpu[i]} - cpu[i]) * (double{gpu[i]} - cpu[i]);
        norm += double{cpu[i]} * cpu[i];
    }
    CHECK(std::sqrt(err) <= 0.05 * std::sqrt(norm));
    CHECK(locus::model::argmax(gpu) == locus::model::argmax(cpu));
}
