#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cppllm/engine/engine.hpp"
#include "cppllm/gguf/gguf.hpp"
#include "cppllm/model/llama.hpp"
#include "cppllm/sys/features.hpp"
#include "cppllm/tok/tokenizer.hpp"

/**
 * Minimal single-sequence greedy generation driver (M2):
 *   cppllm-run <model.gguf> <prompt> [n_tokens]
 */
int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <prompt> [n_tokens]\n",
                     argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string prompt = argv[2];
    const int n_gen = argc > 3 ? std::atoi(argv[3]) : 64;

    try {
        auto g = cppllm::gguf::GgufFile::open(model_path);
        auto model = cppllm::model::LlamaModel::load(g);
        auto tok = cppllm::tok::SpmTokenizer::from_gguf(g);
        const auto& hp = model.hparams();
        std::fprintf(stderr,
                     "%s | layers=%u embd=%u heads=%u/%u vocab=%u\n",
                     cppllm::sys::to_string(cppllm::sys::detect())
                         .c_str(),
                     hp.n_layers, hp.n_embd, hp.n_heads,
                     hp.n_kv_heads, hp.n_vocab);

        auto ids = tok.encode(prompt, true);
        std::printf("%s", tok.decode(ids).c_str());

        cppllm::engine::Engine engine(model, tok.eos_id());
        engine.on_token = [&](const cppllm::engine::Request&,
                              cppllm::tok::TokenId t) {
            if (t != tok.eos_id()) {
                std::printf("%s", tok.decode({&t, 1}).c_str());
                std::fflush(stdout);
            }
        };
        engine.submit(ids,
                      static_cast<std::uint32_t>(n_gen));
        engine.run_to_completion();
        std::printf("\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
