#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backend_cli.hpp"
#include "cli_spec.hpp"
#include "locus/engine/disk_prefix_key.hpp"
#include "locus/pipeline/handshake.hpp"
#include "locus/engine/engine.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "locus/model/moe_stats.hpp"
#include "locus/pipeline/expert_client.hpp"
#include "locus/pipeline/net.hpp"
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
    if (!args.kv_type_ok) {
        std::fprintf(stderr,
                     "%s: unknown --kv-type '%s' (f32, f16, q8, q4)\n",
                     argv[0], args.kv_type_name.c_str());
        return 2;
    }
    if (!args.bad_number_flag.empty()) {
        std::fprintf(stderr,
                     "%s: %s needs a non-negative integer, got '%s'\n",
                     argv[0], args.bad_number_flag.c_str(),
                     args.bad_number_value.c_str());
        return 2;
    }
    if (const std::string e = prefix_cache_error(args); !e.empty()) {
        std::fprintf(stderr, "%s: %s\n", argv[0], e.c_str());
        return 2;
    }
    if (args.positional.size() < 2) {
        std::fprintf(stderr,
                     "usage: %s [--backend NAME] <model.gguf> "
                     "<prompt> [n_tokens]\nsee %s --help\n",
                     argv[0], argv[0]);
        return 2;
    }
    // --remote-experts (expert parallelism, #781): checked before the
    // model loads; connected after, once the expert count is known.
    std::vector<locus_tools::RemoteExpertsSpec> remote;
    for (const std::string& s : args.remote_experts) {
        locus_tools::RemoteExpertsSpec r;
        if (!locus_tools::parse_remote_experts(s, r)) {
            std::fprintf(stderr,
                         "%s: bad --remote-experts '%s' (want "
                         "HOST:PORT=LO:HI)\n",
                         argv[0], s.c_str());
            return 2;
        }
        for (const auto& o : remote) {
            if (r.lo < o.hi && o.lo < r.hi) {
                std::fprintf(stderr,
                             "%s: --remote-experts ranges %u:%u and %u:%u "
                             "overlap\n",
                             argv[0], o.lo, o.hi, r.lo, r.hi);
                return 2;
            }
        }
        remote.push_back(std::move(r));
    }
    // i#850: the expert hosts' shared key, read before the model loads.
    std::string expert_key;
    if (!args.remote_experts_key_file.empty()) {
        if (remote.empty()) {
            std::fprintf(stderr,
                         "%s: --remote-experts-key-file needs "
                         "--remote-experts\n",
                         argv[0]);
            return 2;
        }
        try {
            expert_key = locus::pipeline::read_key_file(
                args.remote_experts_key_file);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: bad --remote-experts-key-file: %s\n",
                         argv[0], e.what());
            return 2;
        }
    }
    const std::string model_path = args.positional[0];
    const std::string prompt = args.positional[1];
    // Guarded parse: atoi wrapped "-1" to 4294967295 tokens and read
    // "abc" as 0 (i#198).
    int n_gen = 64;
    if (args.positional.size() > 2 &&
        !locus_tools::parse_nonneg_int(args.positional[2], n_gen)) {
        std::fprintf(stderr,
                     "%s: n_tokens needs a non-negative integer, got "
                     "'%s'\n",
                     argv[0], args.positional[2].c_str());
        return 2;
    }

    try {
        if (const std::string err = locus_tools::apply_device(args);
            !err.empty()) {
            std::fprintf(stderr, "%s: %s\n", argv[0], err.c_str());
            return 2;
        }
        auto g = locus::gguf::GgufFile::open(model_path);
        auto model = locus::model::TransformerModel::load(g);
        const auto& backend =
            locus::backend::resolve_backend(args.choice);
        model.use_backend(backend);
        locus::backend::vulkan_set_prefill_f16(args.prefill_f16);
        auto tok_ptr = locus::tok::tokenizer_from_gguf(g);
        auto& tok = *tok_ptr;
        const auto& hp = model.hparams();
        const std::string be(model.active_backend().name);
        // Which GPU, when there is a choice of them (i#572).
        const std::string device =
            be == "vulkan" ? " (" + locus::backend::vulkan_device_name() + ")"
                           : "";
        std::fprintf(stderr,
                     "%s | backend=%s%s | layers=%u embd=%u "
                     "heads=%u/%u vocab=%u\n",
                     locus::sys::to_string(locus::sys::detect())
                         .c_str(),
                     be.c_str(), device.c_str(), hp.n_layers, hp.n_embd,
                     hp.n_heads, hp.n_kv_heads, hp.n_vocab);
        const std::string gpu_warn = locus::backend::gpu_idle_warning(
            locus::backend::backend_defaulted(args.choice), backend);
        if (!gpu_warn.empty()) {
            std::fprintf(stderr, "%s\n", gpu_warn.c_str());
        }

        std::vector<locus::pipeline::ExpertClient::Host> hosts;
        for (const auto& r : remote) {
            if (r.hi > hp.n_expert) {
                std::fprintf(stderr,
                             "%s: --remote-experts %u:%u, but the model "
                             "has %u experts\n",
                             argv[0], r.lo, r.hi, hp.n_expert);
                return 2;
            }
            const int fd = locus::pipeline::connect_to(r.host, r.port, 5000);
            if (fd < 0) {
                std::fprintf(stderr, "%s: cannot reach %s:%d\n", argv[0],
                             r.host.c_str(), r.port);
                return 1;
            }
            if (!expert_key.empty() &&
                !locus::pipeline::handshake_connect(
                    fd, expert_key, locus::pipeline::kHandshakeTimeoutMs)) {
                std::fprintf(stderr,
                             "%s: %s:%d failed the key handshake (does it "
                             "run with the same --key-file?)\n",
                             argv[0], r.host.c_str(), r.port);
                return 1;
            }
            hosts.push_back({r.lo, r.hi, fd});
            std::fprintf(stderr, "remote experts [%u,%u) on %s:%d\n", r.lo,
                         r.hi, r.host.c_str(), r.port);
        }
        // Declared before the engine, so it outlives every forward.
        std::unique_ptr<locus::pipeline::ExpertClient> experts;
        if (!hosts.empty()) {
            experts = std::make_unique<locus::pipeline::ExpertClient>(
                std::move(hosts), hp.n_embd);
            model.set_remote_experts(experts.get());
        }

        auto ids = tok.encode(prompt, true);

        locus::engine::Engine::Config cfg;
        if (args.ctx > 0) {
            cfg.n_blocks = (args.ctx + 15) / 16;
        }
        cfg.kv_type = args.kv_type;
        cfg.batched_decode = args.batch_decode;
        cfg.speculative = args.speculative;
        if (args.spec_draft > 0) {
            cfg.spec_draft = args.spec_draft;
        }
        cfg.spec_adaptive = !args.spec_fixed;
        cfg.batched_prefill = !args.no_batch_prefill;
        if (!args.prefix_cache_dir.empty()) {
            cfg.disk_prefix = locus::engine::open_disk_prefix(
                args.prefix_cache_dir,
                std::uint64_t{args.prefix_cache_disk_mib} << 20, g,
                model_path, model, args.kv_type);
        } else if (!args.prefix_cache_helper.empty()) {
            cfg.disk_prefix = locus::engine::open_helper_prefix(
                args.prefix_cache_helper,
                std::uint64_t{args.prefix_cache_disk_mib} << 20, g,
                model_path, model, args.kv_type);
        }
        if (cfg.disk_prefix && !args.prefix_cache_key_file.empty()) {
            cfg.disk_prefix->set_mac_key(
                locus::pipeline::read_key_file(args.prefix_cache_key_file));
        }
        locus::engine::Engine engine(model, tok.eos_id(), cfg);

        using clock = std::chrono::steady_clock;
        if (args.warmup && !ids.empty()) {
            // Throwaway request: first-touch weight uploads and
            // pipeline warm-up land here, not in the timed run.
            engine.submit({ids.front()}, 1);
            engine.run_to_completion();
        }
        const std::uint32_t conc = std::max(1u, args.concurrent);
        if (conc == 1) {
            std::printf("%s", tok.decode(ids).c_str());
            clock::time_point t_first{}, t_last{};
            std::size_t n_out = 0;
            engine.on_token = [&](const locus::engine::Request&,
                                  locus::tok::TokenId t) {
                t_last = clock::now();
                if (n_out++ == 0) {
                    t_first = t_last;
                }
                if (t != tok.eos_id()) {
                    std::printf("%s", tok.decode({&t, 1}).c_str());
                    std::fflush(stdout);
                }
            };
            const clock::time_point t0 = clock::now();
            engine.submit(ids,
                          static_cast<std::uint32_t>(n_gen));
            engine.run_to_completion();
            std::printf("\n");
            if (args.timing && n_out > 0) {
                using ms = std::chrono::duration<double, std::milli>;
                // Prefill: submit to first token (the prompt pass
                // yields its logits); decode: the remaining tokens.
                const double pf = ms(t_first - t0).count();
                const double dec = ms(t_last - t_first).count();
                std::fprintf(
                    stderr,
                    "timing: prefill %zu tok %.1f ms (%.1f tok/s) | "
                    "decode %zu tok %.1f ms (%.1f tok/s)\n",
                    ids.size(), pf,
                    pf > 0 ? 1000.0 * ids.size() / pf : 0.0, n_out - 1,
                    dec, dec > 0 ? 1000.0 * (n_out - 1) / dec : 0.0);
                if (cfg.speculative) {
                    std::fprintf(stderr,
                                 "speculative: %llu verify steps, %llu of "
                                 "%llu drafts accepted\n",
                                 static_cast<unsigned long long>(
                                     engine.spec_steps()),
                                 static_cast<unsigned long long>(
                                     engine.spec_accepted_tokens()),
                                 static_cast<unsigned long long>(
                                     engine.spec_drafted_tokens()));
                    const auto& by_len = engine.spec_draft_counts();
                    std::fprintf(stderr, "verifies by draft length:");
                    for (std::size_t d = 0; d < by_len.size(); ++d) {
                        if (by_len[d] > 0) {
                            std::fprintf(stderr, " %zu:%llu", d,
                                         static_cast<unsigned long long>(
                                             by_len[d]));
                        }
                    }
                    std::fprintf(stderr, "\n");
                }
            }
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
        if (experts) {
            std::fprintf(stderr, "remote expert calls: %llu\n",
                         static_cast<unsigned long long>(experts->calls()));
        }
        locus::model::MoeStats::report();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
