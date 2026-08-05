#include <cstdio>
#include <cstdlib>
#include <string>

#include "backend_cli.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/llama.hpp"
#include "locus/server/server.hpp"
#include "locus/sys/features.hpp"
#include "locus/tok/tokenizer.hpp"

/**
 * OpenAI-compatible inference server (M4):
 *   locus-server [--backend NAME] <model.gguf> [port]
 *   locus-server --backends
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
    if (args.help) {
        std::printf(
            "usage: %s [options] <model.gguf> [port]\n\n"
            "OpenAI + Anthropic compatible inference server "
            "(default port 8080):\n  POST /v1/completions\n  POST "
            "/v1/chat/completions\n  POST /v1/messages  "
            "(Anthropic)\n  GET  /health\n\n%s",
            argv[0], locus_tools::kCommonHelp);
        return 0;
    }
    if (args.positional.empty()) {
        std::fprintf(stderr,
                     "usage: %s [--backend NAME] <model.gguf> "
                     "[port]\nsee %s --help\n",
                     argv[0], argv[0]);
        return 2;
    }
    const std::string model_path = args.positional[0];
    const int port = args.positional.size() > 1
                         ? std::atoi(args.positional[1].c_str())
                         : 8080;

    try {
        auto g = locus::gguf::GgufFile::open(model_path);
        auto model = locus::model::LlamaModel::load(g);
        model.use_backend(
            locus::backend::resolve_backend(args.choice));
        auto tok_ptr = locus::tok::tokenizer_from_gguf(g);
        auto& tok = *tok_ptr;

        locus::server::OpenAiServer::Options opt;
        opt.model_name = model_path;
        if (args.ctx > 0) {
            opt.engine.n_blocks = (args.ctx + 15) / 16;
        }
        opt.chat_template =
            locus::chat::ChatTemplate::from_gguf(g);
        if (!args.metrics_path.empty()) {
            opt.metrics_path = args.metrics_path;
        }
        locus::server::OpenAiServer server(model, tok, opt);
        std::fprintf(
            stderr,
            "%s\nbackend: %s | chat template: %s\n"
            "locus-server listening on :%d\n",
            locus::sys::to_string(locus::sys::detect()).c_str(),
            std::string(model.active_backend().name).c_str(),
            std::string(opt.chat_template.name()).c_str(),
            port);
        if (!server.listen("0.0.0.0", port)) {
            std::fprintf(stderr, "error: cannot bind port %d\n",
                         port);
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
