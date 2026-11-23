// locus-embed: embeds labeled examples with a locus embedding model,
// for scripts/text/train-linear-head.py (i#1160). The vectors come
// from the same EmbeddingModel::embed the server's /v1/embeddings and
// the linear-head guard use (pooled as the GGUF says, L2-normalized),
// so a head trained on them scores what the server computes.
//
//   locus-embed MODEL.gguf IN.jsonl OUT.jsonl
//
// Each input line is a JSON object with "text" (and any other fields,
// such as "label"); each output line is the same object with
// "embedding" added.

#include <algorithm>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "json.hpp"
#include "locus/backend/registry.hpp"
#include "locus/embed/embedding_model.hpp"

namespace {

/** Inputs per embed() call. */
constexpr std::size_t kBatch = 32;

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: locus-embed MODEL.gguf IN.jsonl OUT.jsonl\n");
        return 2;
    }
    try {
        const auto model = locus::embed::EmbeddingModel::load(argv[1]);
        if (model->reranker()) {
            std::fprintf(stderr, "locus-embed: %s is a reranker, not an "
                                 "embedding model\n", argv[1]);
            return 1;
        }
        std::ifstream in(argv[2]);
        if (!in) {
            std::fprintf(stderr, "locus-embed: cannot read %s\n", argv[2]);
            return 1;
        }
        std::vector<nlohmann::json> rows;
        std::string line;
        for (std::size_t n = 1; std::getline(in, line); ++n) {
            if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
            auto row = nlohmann::json::parse(line);
            if (!row.is_object() || !row.contains("text") ||
                !row["text"].is_string()) {
                std::fprintf(stderr, "locus-embed: line %zu has no \"text\"\n",
                             n);
                return 1;
            }
            rows.push_back(std::move(row));
        }
        std::ofstream out(argv[3]);
        if (!out) {
            std::fprintf(stderr, "locus-embed: cannot write %s\n", argv[3]);
            return 1;
        }
        const auto& op = locus::backend::best_backend().ops;
        for (std::size_t at = 0; at < rows.size(); at += kBatch) {
            const std::size_t end = std::min(rows.size(), at + kBatch);
            std::vector<std::string> texts;
            for (std::size_t i = at; i < end; ++i) {
                texts.push_back(rows[i]["text"].get<std::string>());
            }
            const auto vecs = model->embed(texts, op);
            for (std::size_t i = at; i < end; ++i) {
                rows[i]["embedding"] = vecs[i - at];
                out << rows[i].dump() << '\n';
            }
        }
        std::fprintf(stderr, "locus-embed: %zu examples, width %u, %s\n",
                     rows.size(), model->width(), model->name().c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "locus-embed: %s\n", e.what());
        return 1;
    }
    return 0;
}
