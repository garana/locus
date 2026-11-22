#include "locus/server/classifier_guard.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "locus/backend/registry.hpp"
#include "locus/embed/deberta.hpp"
#include "locus/embed/embedding_model.hpp"
#include "locus/gguf/gguf.hpp"

namespace locus::server {

ProblemType parse_problem_type(std::optional<std::string_view> name,
                               std::size_t labels) {
    if (!name) {
        return labels > 1 ? ProblemType::kSingleLabel
                          : ProblemType::kMultiLabel;
    }
    if (*name == "single_label_classification") {
        return ProblemType::kSingleLabel;
    }
    if (*name == "multi_label_classification") return ProblemType::kMultiLabel;
    if (*name == "regression") return ProblemType::kRegression;
    throw std::invalid_argument("unknown classifier problem_type '" +
                                std::string(*name) + "'");
}

std::vector<float> label_scores(std::span<const float> logits,
                                ProblemType type) {
    std::vector<float> out(logits.begin(), logits.end());
    if (type == ProblemType::kMultiLabel) {
        for (float& z : out) z = 1.0f / (1.0f + std::exp(-z));
        return out;
    }
    if (type != ProblemType::kSingleLabel) {
        throw std::invalid_argument("a regression output has no label scores");
    }
    if (out.empty()) return out;
    const float mx = *std::max_element(out.begin(), out.end());
    double sum = 0.0;
    for (float& z : out) {
        z = std::exp(z - mx);
        sum += z;
    }
    for (float& z : out) z = static_cast<float>(z / sum);
    return out;
}

std::vector<std::vector<tok::TokenId>> guard_windows(
    std::span<const tok::TokenId> body, std::uint32_t context,
    std::uint32_t overlap, tok::TokenId cls, tok::TokenId sep) {
    if (context <= 2 + overlap) {
        throw std::invalid_argument(
            "guard windows: the context must exceed 2 + the overlap");
    }
    const std::size_t width = context - 2, stride = width - overlap;
    std::vector<std::vector<tok::TokenId>> out;
    for (std::size_t at = 0; at < body.size(); at += stride) {
        const std::size_t n = std::min(width, body.size() - at);
        std::vector<tok::TokenId> w;
        w.reserve(n + 2);
        w.push_back(cls);
        w.insert(w.end(), body.begin() + static_cast<std::ptrdiff_t>(at),
                 body.begin() + static_cast<std::ptrdiff_t>(at + n));
        w.push_back(sep);
        out.push_back(std::move(w));
        if (at + n == body.size()) break;
    }
    return out;
}

ClassifierGuard::ClassifierGuard(Model model, const GuardSpec& spec)
    : Guard(spec.name, spec.labels),
      model_(std::move(model)),
      max_windows_(spec.max_windows) {
    if (model_.problem == ProblemType::kRegression) {
        throw std::invalid_argument("guard " + spec.name +
                                    ": a regression model has no labels "
                                    "to reject on");
    }
    if (model_.context <= 2 + kOverlap) {
        throw std::invalid_argument("guard " + spec.name +
                                    ": the model's context is too short");
    }
    for (std::size_t i = 0; i < spec.labels.size(); ++i) {
        const auto it = std::find(model_.labels.begin(), model_.labels.end(),
                                  spec.labels[i]);
        if (it == model_.labels.end()) {
            std::string known;
            for (const auto& l : model_.labels) {
                known += (known.empty() ? "" : ", ") + l;
            }
            throw std::invalid_argument("guard " + spec.name + ": label '" +
                                        spec.labels[i] +
                                        "' is not the model's (" + known +
                                        ")");
        }
        const float t = i < spec.thresholds.size() &&
                                !std::isnan(spec.thresholds[i])
                            ? spec.thresholds[i]
                            : spec.threshold;
        rules_.push_back(
            {static_cast<std::size_t>(it - model_.labels.begin()), t});
    }
}

std::string ClassifierGuard::last_error() {
    std::lock_guard<std::mutex> lock(mu_);
    return last_error_;
}

std::vector<GuardVerdict> ClassifierGuard::check_batch(
    std::span<const Turns> requests, const Runner& run) {
    const std::size_t nl = model_.labels.size();
    std::vector<GuardVerdict> out(requests.size());  // kMalformed
    // Every window of the batch, and the request each belongs to.
    std::vector<std::vector<tok::TokenId>> windows;
    std::vector<std::size_t> owner;
    std::vector<bool> ok(requests.size(), true);
    std::lock_guard<std::mutex> lock(mu_);
    try {
        for (std::size_t r = 0; r < requests.size(); ++r) {
            std::size_t mine = 0;
            for (const chat::Message& m : requests[r]) {
                if (m.role != "user") continue;  // tool results included
                auto w = guard_windows(model_.encode(m.content),
                                       model_.context, kOverlap, model_.cls,
                                       model_.sep);
                mine += w.size();
                if (mine > max_windows_) break;
                for (auto& x : w) {
                    windows.push_back(std::move(x));
                    owner.push_back(r);
                }
            }
            if (mine > max_windows_) {
                ok[r] = false;
                out[r].error = "the request needs more than " +
                               std::to_string(max_windows_) +
                               " windows of the guard's context";
                // Drop the windows already queued for it.
                while (!owner.empty() && owner.back() == r) {
                    owner.pop_back();
                    windows.pop_back();
                }
            }
        }
        std::vector<std::vector<float>> logits;
        logits.reserve(windows.size());
        const auto work = [&] {
            for (std::size_t at = 0; at < windows.size();
                 at += kWindowsPerCall) {
                const std::size_t n =
                    std::min(kWindowsPerCall, windows.size() - at);
                auto part = model_.classify(
                    std::span<const std::vector<tok::TokenId>>(
                        windows.data() + at, n));
                if (part.size() != n) {
                    throw std::runtime_error(
                        "the classifier gave the wrong number of results");
                }
                for (auto& z : part) logits.push_back(std::move(z));
            }
        };
        if (!windows.empty()) {
            if (run) {
                run(work);
            } else {
                work();
            }
        }
        // Each label's highest score over each request's windows.
        std::vector<std::vector<float>> top(requests.size(),
                                            std::vector<float>(nl, 0.0f));
        for (std::size_t i = 0; i < logits.size(); ++i) {
            if (logits[i].size() != nl) {
                throw std::runtime_error(
                    "the classifier gave the wrong number of logits");
            }
            const auto s = label_scores(logits[i], model_.problem);
            auto& t = top[owner[i]];
            for (std::size_t l = 0; l < nl; ++l) t[l] = std::max(t[l], s[l]);
        }
        last_error_.clear();
        for (std::size_t r = 0; r < requests.size(); ++r) {
            if (!ok[r]) {
                last_error_ = out[r].error;
                continue;
            }
            GuardVerdict v;
            v.kind = GuardVerdict::Kind::kSafe;
            for (const Rule& rule : rules_) {
                if (top[r][rule.index] >= rule.threshold) {
                    v.kind = GuardVerdict::Kind::kUnsafe;
                    v.categories.push_back(model_.labels[rule.index]);
                }
            }
            out[r] = std::move(v);
        }
    } catch (const std::exception& e) {
        // The whole batch fails closed, each with the reason.
        last_error_ = e.what();
        for (GuardVerdict& v : out) {
            v = GuardVerdict{};  // kMalformed
            v.error = last_error_;
        }
    }
    return out;
}

std::unique_ptr<ClassifierGuard> load_prompt_guard(const GuardSpec& spec) {
    // The GGUF, encoder and tokenizer live together for the guard.
    struct Files {
        gguf::GgufFile file;
        embed::DebertaEncoder encoder;
        std::unique_ptr<tok::Tokenizer> tok;
        explicit Files(const std::string& path)
            : file(gguf::GgufFile::open(path)),
              encoder(embed::DebertaEncoder::load(file)),
              tok(tok::tokenizer_from_gguf(file)) {}
    };
    auto files = std::make_shared<Files>(spec.model);
    const gguf::Value* pt =
        files->file.find("deberta-v2.classifier.problem_type");
    if (pt != nullptr && pt->type != gguf::ValueType::kString) {
        throw gguf::Error("prompt guard: classifier.problem_type is not a "
                          "string");
    }
    const tok::TokenId cls = files->tok->bos_id(), sep = files->tok->eos_id();
    if (cls < 0 || sep < 0) {
        throw gguf::Error("prompt guard: the tokenizer has no [CLS] or "
                          "[SEP] (bos and eos)");
    }
    ClassifierGuard::Model m;
    m.kind = "prompt-guard";
    m.labels = files->encoder.labels();
    m.problem = parse_problem_type(
        pt != nullptr ? std::optional<std::string_view>(pt->s) : std::nullopt,
        m.labels.size());
    m.context = files->encoder.context();
    Files* raw = files.get();
    m.encode = [raw](std::string_view text) {
        return raw->tok->encode(text, false);
    };
    m.cls = cls;
    m.sep = sep;
    m.classify = [raw](std::span<const std::vector<tok::TokenId>> batch) {
        return raw->encoder.classify(batch, backend::best_backend().ops);
    };
    m.owner = files;
    return std::make_unique<ClassifierGuard>(std::move(m), spec);
}

std::unique_ptr<ClassifierGuard> load_encoder_classifier(
    const GuardSpec& spec, const backend::Ops* gpu) {
    std::shared_ptr<embed::EmbeddingModel> em =
        embed::EmbeddingModel::load(spec.model);
    if (!em->reranker()) {
        throw gguf::Error(spec.model + ": no classifier head (a sequence "
                                       "classifier has pooling rank)");
    }
    if (!em->has_pooler()) {
        // llama.cpp's converter drops BertForSequenceClassification's
        // pooler.dense, so cls.output then reads the raw CLS state and
        // the scores are not the model's. A guard with wrong scores is
        // worse than none.
        throw gguf::Error(
            spec.model + ": the classifier head has no pooler (cls); "
                         "convert the model with "
                         "scripts/text/convert-bert-classifier.py");
    }
    const tok::TokenId cls = em->cls_token(), sep = em->end_token();
    if (cls < 0 || sep < 0) {
        throw gguf::Error(spec.model + ": the tokenizer has no first or "
                                       "last token for one sequence");
    }
    ClassifierGuard::Model m;
    m.kind = "classifier";
    m.labels = em->label_names();
    const auto& pt = em->problem_type();
    m.problem = parse_problem_type(
        pt ? std::optional<std::string_view>(*pt) : std::nullopt,
        m.labels.size());
    m.context = em->context();
    embed::EmbeddingModel* raw = em.get();
    // Literal: this tokenizer's plain encode() parses typed special
    // tokens (SentencePiece since #1151), which a guard must not.
    m.encode = [raw](std::string_view text) {
        return raw->encode_literal(text);
    };
    m.cls = cls;
    m.sep = sep;
    m.classify = [raw, gpu](std::span<const std::vector<tok::TokenId>> batch) {
        return raw->logits(batch, backend::best_backend().ops, gpu);
    };
    m.owner = em;
    return std::make_unique<ClassifierGuard>(std::move(m), spec);
}

}  // namespace locus::server
