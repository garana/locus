#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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

        auto st = model.make_state();
        std::vector<float> logits(hp.n_vocab);

        auto ids = tok.encode(prompt, true);
        for (auto id : ids) {
            model.forward(id, st, logits);
        }
        std::printf("%s", tok.decode(ids).c_str());

        std::vector<cppllm::tok::TokenId> generated;
        for (int i = 0; i < n_gen; ++i) {
            auto next = cppllm::model::argmax(logits);
            if (next == tok.eos_id()) {
                break;
            }
            generated.push_back(next);
            std::printf("%s",
                        tok.decode({&next, 1}).c_str());
            std::fflush(stdout);
            model.forward(next, st, logits);
        }
        std::printf("\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
