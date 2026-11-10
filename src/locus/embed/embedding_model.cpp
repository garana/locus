#include "locus/embed/embedding_model.hpp"

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace locus::embed {

EmbeddingModel::EmbeddingModel(std::unique_ptr<gguf::GgufFile> g,
                               std::unique_ptr<tok::Tokenizer> tok,
                               BertEncoder enc, std::string name)
    : gguf_(std::move(g)),
      tok_(std::move(tok)),
      enc_(std::move(enc)),
      name_(std::move(name)) {}

std::unique_ptr<EmbeddingModel> EmbeddingModel::load(const std::string& path,
                                                     std::string name) {
    auto g = std::make_unique<gguf::GgufFile>(gguf::GgufFile::open(path));
    if (g->get_string("general.architecture") != "bert") {
        throw gguf::Error(path + ": not an embedding model (arch bert)");
    }
    if (g->get_string("tokenizer.ggml.model") != "bert") {
        throw gguf::Error(path + ": tokenizer is not WordPiece (\"bert\")");
    }
    auto tok = tok::tokenizer_from_gguf(*g);
    auto enc = BertEncoder::load(*g);
    if (tok->vocab_size() == 0) throw gguf::Error(path + ": empty vocab");
    if (name.empty()) name = std::filesystem::path(path).filename().string();
    return std::unique_ptr<EmbeddingModel>(new EmbeddingModel(
        std::move(g), std::move(tok), std::move(enc), std::move(name)));
}

std::vector<std::vector<float>> EmbeddingModel::embed(
    const std::vector<std::string>& inputs, const backend::Ops& op,
    std::size_t* tokens) const {
    std::vector<std::vector<tok::TokenId>> ids;
    ids.reserve(inputs.size());
    std::size_t total = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        ids.push_back(tok_->encode(inputs[i], true));
        if (ids.back().size() > context()) {
            throw std::invalid_argument(
                "input " + std::to_string(i) + " has " +
                std::to_string(ids.back().size()) +
                " tokens; the embedding model takes at most " +
                std::to_string(context()));
        }
        total += ids.back().size();
    }
    if (tokens != nullptr) *tokens = total;
    std::vector<std::vector<float>> out;
    out.reserve(inputs.size());
    for (std::size_t first = 0; first < ids.size();) {
        std::size_t last = first, n = 0;
        while (last < ids.size() &&
               (last == first || n + ids[last].size() <= kPassTokens)) {
            n += ids[last++].size();
        }
        auto part = enc_.embed(
            std::span<const std::vector<tok::TokenId>>(ids).subspan(
                first, last - first),
            op);
        for (auto& v : part) out.push_back(std::move(v));
        first = last;
    }
    return out;
}

}  // namespace locus::embed
