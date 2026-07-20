#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "backend_cli.hpp"
#include "cppllm/engine/engine.hpp"
#include "cppllm/gguf/gguf.hpp"
#include "cppllm/model/llama.hpp"
#include "cppllm/sys/features.hpp"
#include "cppllm/tok/tokenizer.hpp"

/**
 * Minimal single-sequence greedy generation driver (M2):
 *   cppllm-run [--backend NAME] <model.gguf> <prompt> [n_tokens]
 *   cppllm-run --backends
 */
int main(int argc, char** argv) {
    auto args = cppllm_tools::parse_backend_args(argc, argv);
    if (args.list) {
        cppllm_tools::print_backends();
        return 0;
    }
    if (args.list_archs) {
        cppllm_tools::print_archs();
        return 0;
    }
    if (args.help) {
        std::printf(
            "usage: %s [options] <model.gguf> <prompt> "
            "[n_tokens]\n\nGreedy single-prompt generation "
            "(default 64 tokens).\n\n%s",
            argv[0], cppllm_tools::kCommonHelp);
        return 0;
    }
    if (args.positional.size() < 2) {
        std::fprintf(stderr,
                     "usage: %s [--backend NAME] <model.gguf> "
                     "<prompt> [n_tokens]\nsee %s --help\n",
                     argv[0], argv[0]);
        return 2;
    }
    const std::string model_path = args.positional[0];
    const std::string prompt = args.positional[1];
    const int n_gen = args.positional.size() > 2
                          ? std::atoi(args.positional[2].c_str())
                          : 64;

    try {
        auto g = cppllm::gguf::GgufFile::open(model_path);
        auto model = cppllm::model::LlamaModel::load(g);
        model.use_backend(
            cppllm::backend::resolve_backend(args.choice));
        auto tok_ptr = cppllm::tok::tokenizer_from_gguf(g);
        auto& tok = *tok_ptr;
        const auto& hp = model.hparams();
        std::fprintf(stderr,
                     "%s | backend=%s | layers=%u embd=%u "
                     "heads=%u/%u vocab=%u\n",
                     cppllm::sys::to_string(cppllm::sys::detect())
                         .c_str(),
                     std::string(model.active_backend().name)
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
