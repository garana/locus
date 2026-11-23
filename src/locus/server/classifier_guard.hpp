#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "locus/server/guard.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::server {

/** How a classifier's logits become label scores, as transformers'
 *  config.json problem_type names it. */
enum class ProblemType {
    kSingleLabel,  ///< "single_label_classification": softmax
    kMultiLabel,   ///< "multi_label_classification": sigmoid per label
    kRegression,   ///< "regression": the raw value (no guard takes it)
};

/**
 * @param name A GGUF's classifier.problem_type, or nullopt when the
 *     key is absent (llama.cpp writes none): then what transformers'
 *     text-classification pipeline applies, softmax over several
 *     labels and a sigmoid on one.
 * @param labels The classifier's label count.
 * @returns The problem type.
 * @throws std::invalid_argument on any other name.
 */
ProblemType parse_problem_type(std::optional<std::string_view> name,
                               std::size_t labels);

/**
 * @param logits One input's logits.
 * @param type Softmax (single label) or sigmoid (multi-label).
 * @returns One score in [0, 1] per label.
 * @throws std::invalid_argument for kRegression.
 */
std::vector<float> label_scores(std::span<const float> logits,
                                ProblemType type);

/**
 * Cuts one text's tokens into an encoder's windows: each holds at most
 * context - 2 of them between `cls` and `sep`, and each starts
 * `overlap` tokens before the previous one ends, so a phrase cut by
 * one window's end lies whole in the next.
 *
 * @param body The text's tokens, without specials.
 * @param context The encoder's tokens per input, specials included
 *     (more than 2 + overlap).
 * @param overlap Tokens two neighbouring windows share.
 * @param cls Prepended to each window.
 * @param sep Appended to each window.
 * @returns The windows; none for an empty body.
 * @throws std::invalid_argument when context <= 2 + overlap.
 */
std::vector<std::vector<tok::TokenId>> guard_windows(
    std::span<const tok::TokenId> body, std::uint32_t context,
    std::uint32_t overlap, tok::TokenId cls, tok::TokenId sep);

/**
 * A guard that scores the untrusted text of a request with a sequence
 * classifier (i#1159; the encoder classifiers of i#1157 fit too):
 * every user turn, which includes the tool results (they arrive as
 * user turns), is cut into the encoder's windows (guard_windows), every
 * window of the batch's requests is classified, and each label's
 * score is its highest over the request's windows. A label rejects
 * when that score reaches its threshold; the verdict is kUnsafe with
 * those labels, else kSafe. System and assistant turns are not judged.
 */
class ClassifierGuard : public Guard {
  public:
    /** Logits per window, one vector per input, labels.size() each. */
    using Classify = std::function<std::vector<std::vector<float>>(
        std::span<const std::vector<tok::TokenId>>)>;

    /** The classifier a guard runs. */
    struct Model {
        std::string kind;  ///< --guard's kind= ("prompt-guard")
        std::vector<std::string> labels;  ///< in logit order
        ProblemType problem = ProblemType::kSingleLabel;
        /** Tokens per input, specials included (512). */
        std::uint32_t context = 0;
        /** Encodes a turn's text without specials (encode(text,
         *  false)): the model's own tokenizer, which keeps typed
         *  special-token text as text. */
        std::function<std::vector<tok::TokenId>(std::string_view)> encode;
        tok::TokenId cls = 0, sep = 0;  ///< around each window
        Classify classify;
        /** Keeps the model's files alive as long as the guard. */
        std::shared_ptr<void> owner;
    };

    /** The default of GuardSpec::max_windows: windows over one request,
     *  at most; a longer one fails closed rather than stall the chat
     *  engine (the guard runs on its worker). 64 windows hold about 29k
     *  tokens and take about 15 s of a DeBERTa v3 base on an M4 Max's
     *  CPU (0.24 s a window). */
    static constexpr std::size_t kMaxWindows = kGuardMaxWindows;
    /** Tokens two neighbouring windows share. */
    static constexpr std::uint32_t kOverlap = 64;
    /** Windows per classify() call. */
    static constexpr std::size_t kWindowsPerCall = 16;

    /**
     * @param model The classifier.
     * @param spec The guard's --guard spec: its name, the labels that
     *     reject and their thresholds.
     * @throws std::invalid_argument on a label the model does not have
     *     (the error lists the model's), a regression model, or a
     *     context of at most 2 + kOverlap.
     */
    ClassifierGuard(Model model, const GuardSpec& spec);

    std::string_view kind() const override { return model_.kind; }

    std::vector<GuardVerdict> check_batch(std::span<const Turns> requests,
                                          const Runner& run = {}) override;

    std::string last_error() override;

    /** @returns The model's labels, in logit order. */
    const std::vector<std::string>& model_labels() const {
        return model_.labels;
    }

  private:
    /** A label that rejects: its logit index and threshold. */
    struct Rule {
        std::size_t index;
        float threshold;
    };
    Model model_;
    std::vector<Rule> rules_;
    std::size_t max_windows_;  ///< GuardSpec::max_windows
    std::string last_error_;
    std::mutex mu_;
};

/**
 * Loads a Prompt Guard GGUF (kind "prompt-guard"): a DeBERTa v3
 * sequence classifier of arch deberta-v2 (scripts/text/convert-deberta.py),
 * with its labels and problem type from the GGUF. Its encoder runs on
 * the best CPU backend.
 *
 * @param spec The guard's spec (model path, name, labels).
 * @returns The guard.
 * @throws gguf::Error when the GGUF cannot load; std::invalid_argument
 *     as ClassifierGuard's constructor does.
 */
std::unique_ptr<ClassifierGuard> load_prompt_guard(const GuardSpec& spec);

/**
 * Loads an encoder sequence classifier (kind "classifier", i#1157): a
 * BERT, XLM-R or nomic-bert GGUF with a classifier head (pooling
 * rank), as scripts/text/convert-bert-classifier.py writes it, or a
 * llama.cpp reranker. Labels come from {arch}.classifier.output_labels
 * (LABEL_0, ... when absent), the problem type from
 * {arch}.classifier.problem_type. A head without its pooler ("cls":
 * llama.cpp's converter drops BertForSequenceClassification's
 * pooler.dense) is refused, since its scores are not the model's.
 * Text is encoded literally (typed special tokens stay text).
 *
 * @param spec The guard's spec (model path, name, labels).
 * @param gpu GPU ops for large encoder passes, or null.
 * @returns The guard.
 * @throws gguf::Error when the GGUF is not such a model or lacks the
 *     pooler; std::invalid_argument as ClassifierGuard's constructor.
 */
std::unique_ptr<ClassifierGuard> load_encoder_classifier(
    const GuardSpec& spec, const backend::Ops* gpu);

}  // namespace locus::server
