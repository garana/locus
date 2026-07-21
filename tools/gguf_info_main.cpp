#include <cstdio>
#include <cstring>
#include <string>

#include "locus/gguf/gguf.hpp"
#include "locus/tok/tokenizer.hpp"

/**
 * GGUF inspection tool (stable CLI for probing models):
 *   locus-gguf <model.gguf> info            metadata + counts
 *   locus-gguf <model.gguf> tensor <name>   one tensor's shape
 *   locus-gguf <model.gguf> tokens <text>   token ids for text
 */
int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> info\n"
                     "       %s <model.gguf> tensor <name>\n"
                     "       %s <model.gguf> tokens <text>\n",
                     argv[0], argv[0], argv[0]);
        return 2;
    }
    try {
        auto g = locus::gguf::GgufFile::open(argv[1]);
        const std::string cmd = argv[2];
        if (cmd == "info") {
            for (const auto& [k, v] : g.metadata()) {
                std::printf("%s = u%llu i%lld f%g '%s'\n",
                            k.c_str(),
                            (unsigned long long)v.u,
                            (long long)v.i, v.f, v.s.c_str());
            }
            std::printf("total tensors: %zu\n",
                        g.total_tensor_count());
        } else if (cmd == "tensor" && argc > 3) {
            const auto* t = g.find_tensor(argv[3]);
            if (t == nullptr) {
                std::printf("not found: %s\n", argv[3]);
                return 1;
            }
            std::printf(
                "%s ne=[%llu,%llu,%llu,%llu] type=%u "
                "bytes=%llu first=%02x\n",
                t->name.c_str(), t->ne[0], t->ne[1], t->ne[2],
                t->ne[3], (unsigned)t->type, t->nbytes,
                (unsigned)g.tensor_data(*t)[0]);
        } else if (cmd == "tokens" && argc > 3) {
            auto tok = locus::tok::tokenizer_from_gguf(g);
            for (auto id : tok->encode(argv[3], true)) {
                std::printf("%d ", id);
            }
            std::printf("\n");
        } else {
            std::fprintf(stderr, "unknown command: %s\n",
                         cmd.c_str());
            return 2;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
