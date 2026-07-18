#include <cstdio>
#include <cstdlib>
#include <string>

#include "backend_cli.hpp"
#include "cppllm/gguf/gguf.hpp"
#include "cppllm/model/llama.hpp"
#include "cppllm/server/server.hpp"
#include "cppllm/sys/features.hpp"
#include "cppllm/tok/tokenizer.hpp"

/**
 * OpenAI-compatible inference server (M4):
 *   cppllm-server [--backend NAME] <model.gguf> [port]
 *   cppllm-server --backends
 */
int main(int argc, char** argv) {
    auto args = cppllm_tools::parse_backend_args(argc, argv);
    if (args.list) {
        cppllm_tools::print_backends();
        return 0;
    }
    if (args.positional.empty()) {
        std::fprintf(stderr,
                     "usage: %s [--backend NAME] <model.gguf> "
                     "[port]\n       %s --backends\n",
                     argv[0], argv[0]);
        return 2;
    }
    const std::string model_path = args.positional[0];
    const int port = args.positional.size() > 1
                         ? std::atoi(args.positional[1].c_str())
                         : 8080;

    try {
        auto g = cppllm::gguf::GgufFile::open(model_path);
        auto model = cppllm::model::LlamaModel::load(g);
        model.use_backend(
            cppllm::backend::resolve_backend(args.choice));
        auto tok = cppllm::tok::SpmTokenizer::from_gguf(g);

        cppllm::server::OpenAiServer::Options opt;
        opt.model_name = model_path;
        cppllm::server::OpenAiServer server(model, tok, opt);
        std::fprintf(
            stderr,
            "%s\nbackend: %s\ncppllm-server listening on :%d\n",
            cppllm::sys::to_string(cppllm::sys::detect()).c_str(),
            std::string(model.active_backend().name).c_str(),
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
