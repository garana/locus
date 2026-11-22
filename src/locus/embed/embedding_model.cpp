#include "locus/embed/embedding_model.hpp"

#include <filesystem>
#include <functional>
#include <stdexcept>
#include <utility>

namespace locus::embed {

namespace {

/**
 * Splits ids into runs of at most EmbeddingModel::kPassTokens tokens
 * (a longer sequence is a run of its own) and calls fn on each run.
 */
void passes(const std::vector<std::vector<tok::TokenId>>& ids,
            const std::function<void(
                std::span<const std::vector<tok::TokenId>>)>& fn) {
    for (std::size_t first = 0; first < ids.size();) {
        std::size_t last = first, n = 0;
        while (last < ids.size() &&
               (last == first ||
                n + ids[last].size() <= EmbeddingModel::kPassTokens)) {
            n += ids[last++].size();
        }
        fn(std::span<const std::vector<tok::TokenId>>(ids).subspan(
            first, last - first));
        first = last;
    }
}

/** @returns gpu for a pass of at least EmbeddingModel::kGpuMinTokens
 *  tokens and kGpuMacs of work when there is one, else op. */
const backend::Ops& pick(std::span<const std::vector<tok::TokenId>> run,
                         const backend::Ops& op, const backend::Ops* gpu,
                         std::uint64_t token_macs) {
    if (gpu == nullptr) return op;
    std::uint64_t n = 0;
    for (const auto& ids : run) n += ids.size();
    return n >= EmbeddingModel::kGpuMinTokens &&
                   n * token_macs >= EmbeddingModel::kGpuMacs
               ? *gpu
               : op;
}

}  // namespace

EmbeddingModel::EmbeddingModel(std::unique_ptr<gguf::GgufFile> g,
                               std::unique_ptr<tok::Tokenizer> tok,
                               BertEncoder enc, std::string name,
                               PairFormat pair)
    : gguf_(std::move(g)),
      tok_(std::move(tok)),
      enc_(std::move(enc)),
      name_(std::move(name)),
      pair_(pair) {}

std::unique_ptr<EmbeddingModel> EmbeddingModel::load(const std::string& path,
                                                     std::string name) {
    auto g = std::make_unique<gguf::GgufFile>(gguf::GgufFile::open(path));
    const auto arch = g->get_string("general.architecture");
    if (arch != "bert" && arch != "nomic-bert") {
        throw gguf::Error(path + ": not an encoder model (arch bert or "
                                 "nomic-bert)");
    }
    // WordPiece (BERT) or SentencePiece unigram (XLM-R).
    const auto tm = g->get_string("tokenizer.ggml.model");
    if (tm != "bert" && tm != "t5") {
        throw gguf::Error(path + ": tokenizer is not WordPiece (\"bert\") "
                                 "or unigram (\"t5\")");
    }
    const bool wpm = tm == "bert";
    auto tok = tok::tokenizer_from_gguf(*g);
    auto enc = BertEncoder::load(*g);
    if (tok->vocab_size() == 0) throw gguf::Error(path + ": empty vocab");
    if (name.empty()) name = std::filesystem::path(path).filename().string();

    // The pair layout of llama-server's rerank, from llama.cpp's
    // defaults per tokenizer (WordPiece: CLS 101 and SEP 102 added, no
    // EOS; unigram: EOS 1 added) and the GGUF's keys. A missing EOS
    // falls back to SEP, as there.
    PairFormat pair;
    if (enc.pooling() == Pooling::kRank) {
        const auto id = [&](std::string_view key, tok::TokenId fallback) {
            const auto v = g->get_uint(key);
            return v ? static_cast<tok::TokenId>(*v) : fallback;
        };
        const auto flag = [&](std::string_view key, bool fallback) {
            return g->get_bool(key).value_or(fallback);
        };
        const tok::TokenId bos = id("tokenizer.ggml.bos_token_id",
                                    wpm ? 101 : -1);
        const tok::TokenId sep = id("tokenizer.ggml.seperator_token_id",
                                    wpm ? 102 : -1);
        tok::TokenId eos = id("tokenizer.ggml.eos_token_id", wpm ? -1 : 1);
        if (eos < 0) eos = sep;
        if (flag("tokenizer.ggml.add_bos_token", wpm)) pair.bos = bos;
        if (flag("tokenizer.ggml.add_eos_token", !wpm)) pair.eos = eos;
        if (flag("tokenizer.ggml.add_sep_token", wpm)) pair.sep = sep;
        const auto n = static_cast<tok::TokenId>(tok->vocab_size());
        for (const tok::TokenId t : {pair.bos, pair.eos, pair.sep}) {
            if (t >= n) throw gguf::Error(path + ": special id past the vocab");
        }
        if (enc.labels() == 0) throw gguf::Error(path + ": no classifier");
    }
    // A classifier's label names (llama.cpp's converter writes id2label
    // as {arch}.classifier.output_labels) and problem type (ours,
    // transformers' value as is), for a guard (i#1157).
    std::vector<std::string> labels;
    std::optional<std::string> problem;
    if (enc.pooling() == Pooling::kRank) {
        const std::string a(*arch);
        if (const auto* v = g->get_array(a + ".classifier.output_labels")) {
            for (const auto& x : *v) {
                if (x.type != gguf::ValueType::kString) {
                    throw gguf::Error(path + ": a classifier label is not a "
                                             "string");
                }
                labels.push_back(x.s);
            }
            if (labels.size() != enc.labels()) {
                throw gguf::Error(path + ": " + std::to_string(labels.size()) +
                                  " label names for " +
                                  std::to_string(enc.labels()) + " outputs");
            }
        } else {
            for (std::uint32_t i = 0; i < enc.labels(); ++i) {
                labels.push_back("LABEL_" + std::to_string(i));
            }
        }
        if (const gguf::Value* pt = g->find(a + ".classifier.problem_type")) {
            if (pt->type != gguf::ValueType::kString) {
                throw gguf::Error(path + ": classifier.problem_type is not a "
                                         "string");
            }
            problem = pt->s;
        }
    }
    auto m = std::unique_ptr<EmbeddingModel>(
        new EmbeddingModel(std::move(g), std::move(tok), std::move(enc),
                           std::move(name), pair));
    m->labels_ = std::move(labels);
    m->problem_type_ = std::move(problem);
    return m;
}

std::vector<tok::TokenId> EmbeddingModel::encode_literal(
    std::string_view text) const {
    const tok::TextRange all{0, text.size()};
    return tok_->encode_literal(text, false, {&all, 1});
}

std::vector<std::vector<float>> EmbeddingModel::logits(
    std::span<const std::vector<tok::TokenId>> inputs, const backend::Ops& op,
    const backend::Ops* gpu) const {
    if (!reranker()) {
        throw std::invalid_argument(name_ + " has no classifier head");
    }
    const std::vector<std::vector<tok::TokenId>> ids(inputs.begin(),
                                                     inputs.end());
    std::vector<std::vector<float>> out;
    out.reserve(ids.size());
    passes(ids, [&](std::span<const std::vector<tok::TokenId>> run) {
        for (auto& l : enc_.rank(run, pick(run, op, gpu, enc_.token_macs()))) {
            out.push_back(std::move(l));
        }
    });
    return out;
}

std::vector<std::vector<float>> EmbeddingModel::embed(
    const std::vector<std::string>& inputs, const backend::Ops& op,
    std::size_t* tokens, const backend::Ops* gpu) const {
    if (reranker()) {
        throw std::invalid_argument(name_ + " is a reranker, not an "
                                            "embedding model");
    }
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
    passes(ids, [&](std::span<const std::vector<tok::TokenId>> run) {
        for (auto& v :
             enc_.embed(run, pick(run, op, gpu, enc_.token_macs()))) {
            out.push_back(std::move(v));
        }
    });
    return out;
}

std::pair<tok::TokenId, tok::TokenId> EmbeddingModel::text_ends() const {
    const auto ids = tok_->encode("", true);
    if (ids.size() != 2) {
        throw gguf::Error(name_ + ": the tokenizer does not frame a text "
                                  "with one first and one last token");
    }
    return {ids[0], ids[1]};
}

std::vector<std::vector<float>> EmbeddingModel::embed_ids(
    std::span<const std::vector<tok::TokenId>> inputs, const backend::Ops& op,
    const backend::Ops* gpu) const {
    if (reranker()) {
        throw std::invalid_argument(name_ + " is a reranker, not an "
                                            "embedding model");
    }
    const std::vector<std::vector<tok::TokenId>> ids(inputs.begin(),
                                                     inputs.end());
    std::vector<std::vector<float>> out;
    out.reserve(ids.size());
    passes(ids, [&](std::span<const std::vector<tok::TokenId>> run) {
        for (auto& v :
             enc_.embed(run, pick(run, op, gpu, enc_.token_macs()))) {
            out.push_back(std::move(v));
        }
    });
    return out;
}

std::vector<float> EmbeddingModel::rank(
    const std::string& query, const std::vector<std::string>& documents,
    const backend::Ops& op, std::size_t* tokens,
    const backend::Ops* gpu) const {
    if (!reranker()) {
        throw std::invalid_argument(name_ + " is not a reranker");
    }
    std::vector<tok::TokenId> head;
    if (pair_.bos >= 0) head.push_back(pair_.bos);
    const auto q = tok_->encode(query, false);
    head.insert(head.end(), q.begin(), q.end());
    if (pair_.eos >= 0) head.push_back(pair_.eos);
    if (pair_.sep >= 0) head.push_back(pair_.sep);
    const std::size_t tail = pair_.eos >= 0 ? 1 : 0;
    if (head.size() + tail >= context()) {
        throw std::invalid_argument(
            "the query has " + std::to_string(q.size()) +
            " tokens, which leaves no room for a document in the "
            "reranker's " + std::to_string(context()));
    }
    const std::size_t room = context() - head.size() - tail;
    std::vector<std::vector<tok::TokenId>> ids;
    ids.reserve(documents.size());
    std::size_t total = 0;
    for (const auto& d : documents) {
        auto doc = tok_->encode(d, false);
        if (doc.size() > room) doc.resize(room);
        std::vector<tok::TokenId> p = head;
        p.insert(p.end(), doc.begin(), doc.end());
        if (pair_.eos >= 0) p.push_back(pair_.eos);
        total += p.size();
        ids.push_back(std::move(p));
    }
    if (tokens != nullptr) *tokens = total;
    std::vector<float> out;
    out.reserve(documents.size());
    passes(ids, [&](std::span<const std::vector<tok::TokenId>> run) {
        for (const auto& logits :
             enc_.rank(run, pick(run, op, gpu, enc_.token_macs()))) {
            out.push_back(logits[0]);
        }
    });
    return out;
}

}  // namespace locus::embed
