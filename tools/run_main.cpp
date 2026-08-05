#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "backend_cli.hpp"
#include "locus/engine/engine.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/llama.hpp"
#include "locus/model/moe_stats.hpp"
#include "locus/sys/features.hpp"
#include "locus/tok/tokenizer.hpp"

/**
 * Minimal single-sequence greedy generation driver (M2):
 *   locus-run [--backend NAME] <model.gguf> <prompt> [n_tokens]
 *   locus-run --backends
 */
int main(int argc, char** argv) {
    auto args = locus_tools::parse_backend_args(argc, argv);
    if (args.list) {
        locus_tools::print_backends();
        return 0;
    }
    if (args.list_archs) {
        locus_tools::print_archs();
        return 0;
    }
    if (args.list_quants) {
        locus_tools::print_quants();
        return 0;
    }
    if (args.list_tokenizers) {
        locus_tools::print_tokenizers();
        return 0;
    }
    if (args.list_capabilities) {
        locus_tools::print_capabilities();
        return 0;
    }
    if (args.help) {
        std::printf(
            "usage: %s [options] <model.gguf> <prompt> "
            "[n_tokens]\n\nGreedy single-prompt generation "
            "(default 64 tokens).\n\n%s",
            argv[0], locus_tools::kCommonHelp);
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
        auto g = locus::gguf::GgufFile::open(model_path);
        auto model = locus::model::LlamaModel::load(g);
        model.use_backend(
            locus::backend::resolve_backend(args.choice));
        auto tok_ptr = locus::tok::tokenizer_from_gguf(g);
        auto& tok = *tok_ptr;
        const auto& hp = model.hparams();
        std::fprintf(stderr,
                     "%s | backend=%s | layers=%u embd=%u "
                     "heads=%u/%u vocab=%u\n",
                     locus::sys::to_string(locus::sys::detect())
                         .c_str(),
                     std::string(model.active_backend().name)
                         .c_str(),
                     hp.n_layers, hp.n_embd, hp.n_heads,
                     hp.n_kv_heads, hp.n_vocab);

        auto ids = tok.encode(prompt, true);

        locus::engine::Engine::Config cfg;
        if (args.ctx > 0) {
            cfg.n_blocks = (args.ctx + 15) / 16;
        }
        cfg.batched_decode = args.batch_decode;
        cfg.batched_prefill = !args.no_batch_prefill;
        locus::engine::Engine engine(model, tok.eos_id(), cfg);

        const std::uint32_t conc = std::max(1u, args.concurrent);
        if (conc == 1) {
            std::printf("%s", tok.decode(ids).c_str());
            engine.on_token = [&](const locus::engine::Request&,
                                  locus::tok::TokenId t) {
                if (t != tok.eos_id()) {
                    std::printf("%s", tok.decode({&t, 1}).c_str());
                    std::fflush(stdout);
                }
            };
            engine.submit(ids,
                          static_cast<std::uint32_t>(n_gen));
            engine.run_to_completion();
            std::printf("\n");
        } else {
            // Bench: submit the prompt `conc` times and time the
            // aggregate run (no per-token printing).
            std::vector<std::uint64_t> req;
            for (std::uint32_t i = 0; i < conc; ++i) {
                req.push_back(engine.submit(
                    ids, static_cast<std::uint32_t>(n_gen)));
            }
            const auto t0 = std::chrono::steady_clock::now();
            engine.run_to_completion();
            const auto t1 = std::chrono::steady_clock::now();
            const double sec =
                std::chrono::duration<double>(t1 - t0).count();
            std::uint64_t total = 0;
            for (auto id : req) {
                total += engine.get(id)->generated.size();
            }
            std::fprintf(
                stderr,
                "bench: %u seqs, %llu tokens, %.2fs, %.1f tok/s "
                "(batch_decode=%d batch_prefill=%d)\n",
                conc, static_cast<unsigned long long>(total), sec,
                sec > 0 ? total / sec : 0.0,
                static_cast<int>(cfg.batched_decode),
                static_cast<int>(cfg.batched_prefill));
        }
        locus::model::MoeStats::report();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
