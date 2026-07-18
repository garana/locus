#include <cstdio>
#include <cstdlib>
#include <string>

#include "cppllm/gguf/gguf.hpp"
#include "cppllm/model/llama.hpp"
#include "cppllm/server/server.hpp"
#include "cppllm/sys/features.hpp"
#include "cppllm/tok/tokenizer.hpp"

/**
 * OpenAI-compatible inference server (M4):
 *   cppllm-server <model.gguf> [port]
 */
int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [port]\n",
                     argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const int port = argc > 2 ? std::atoi(argv[2]) : 8080;

    try {
        auto g = cppllm::gguf::GgufFile::open(model_path);
        auto model = cppllm::model::LlamaModel::load(g);
        auto tok = cppllm::tok::SpmTokenizer::from_gguf(g);

        cppllm::server::OpenAiServer::Options opt;
        opt.model_name = model_path;
        cppllm::server::OpenAiServer server(model, tok, opt);
        std::fprintf(
            stderr, "%s\ncppllm-server listening on :%d\n",
            cppllm::sys::to_string(cppllm::sys::detect()).c_str(),
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
