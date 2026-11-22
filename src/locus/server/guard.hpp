#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "locus/chat/template.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::backend {
struct Backend;
struct Ops;
}

namespace locus::server {

/**
 * One hazard category of a guard model's taxonomy (i#1106), as its
 * prompt lists it: a code ("S1") and a name ("Violent Crimes").
 */
struct GuardCategory {
    std::string code;
    std::string name;
};

/** A guard model's verdict on a conversation, parsed from its reply. */
struct GuardVerdict {
    enum class Kind {
        kSafe,       ///< first line "safe"
        kUnsafe,     ///< first line "unsafe", codes on the second
        kMalformed,  ///< anything else: a guard error, fails closed
    };
    Kind kind = Kind::kMalformed;
    /** The violated category codes ("S1", "S10"), for kUnsafe. */
    std::vector<std::string> categories;
    /** Why the guard failed on this request, for kMalformed (logs);
     *  may be empty. */
    std::string error;
};

/**
 * Parses a Llama Guard 3 reply: "safe", or "unsafe" then a line of
 * comma-separated category codes ("S1,S10"). Leading and trailing
 * whitespace is ignored, and so are spaces around the codes.
 *
 * @param reply The guard's decoded output.
 * @returns The verdict; kMalformed for any other text, including
 *     "unsafe" with no valid code.
 */
GuardVerdict parse_guard_verdict(std::string_view reply);

/**
 * Whether a verdict rejects the request: an unsafe verdict naming at
 * least one enabled category, or a malformed one (fail closed).
 *
 * @param v The verdict.
 * @param enabled The category codes that reject; empty means all.
 * @returns True to reject.
 */
bool guard_rejects(const GuardVerdict& v,
                   std::span<const std::string> enabled);

/**
 * The Llama Guard 3 prompt that asks whether the last user message of
 * `turns` is unsafe: the task line, the categories, the conversation
 * (its user and assistant turns as "User:" and "Agent:", system turns
 * left out), the instruction, then the assistant header. Text only
 * (no image token). Starts with <|begin_of_text|>, so encode it
 * without adding BOS, and with encode_literal over `content`: a
 * client that types `<|eot_id|>` in a message must not close the
 * guard's turn and write its own verdict.
 *
 * @param turns The chat request's messages.
 * @param categories The taxonomy to list, in order.
 * @returns The prompt text and the byte ranges copied from messages.
 */
chat::ChatTemplate::Prompt llama_guard3_prompt(
    std::span<const chat::Message> turns,
    std::span<const GuardCategory> categories);

/**
 * @returns Llama Guard 3 1B's taxonomy, S1 to S13, as its model card
 *     names them (the 1B model has no S14).
 */
std::vector<GuardCategory> llama_guard3_1b_categories();

/**
 * The text of a guard's reply up to its first `eos` or `eot` token.
 *
 * @param generated The tokens the guard generated.
 * @param eos The model's EOS id.
 * @param eot Its end-of-turn id (<|eot_id|>), which may differ.
 * @param tok The guard's tokenizer.
 * @returns The decoded text before the stop token.
 * @throws std::runtime_error when neither token is there: the reply
 *     reached the token cap, and a cut list of codes would read as a
 *     shorter one.
 */
std::string guard_reply_text(std::span<const tok::TokenId> generated,
                             tok::TokenId eos, tok::TokenId eot,
                             const tok::Tokenizer& tok);

/**
 * A guard: something that judges a chat request before the chat model
 * sees it (i#1106, i#1156). Each kind (Llama Guard 3 now; the
 * classifiers of i#1155 next) implements check(); the labels that
 * reject and the fail-closed rule are shared. A server runs several.
 */
class Guard {
  public:
    /** Runs a piece of work and returns when it is done (for example
     *  EngineLoop::run_on_worker); its exception is rethrown. */
    using Runner = std::function<void(std::function<void()>)>;

    virtual ~Guard() = default;

    /** @returns The kind, as --guard's kind= names it. */
    virtual std::string_view kind() const = 0;

    /** One request's messages. */
    using Turns = std::span<const chat::Message>;

    /**
     * Judges several requests as one batch, one verdict each, in
     * order. A request the guard fails on (it throws, or gives no
     * verdict) gets kMalformed, which rejects.
     *
     * @param requests The requests' messages.
     * @param run Where the model runs: the chat engine's worker thread
     *     in the server, so no backend runs on two threads at once;
     *     empty runs it on the calling thread.
     * @returns requests.size() verdicts; for kUnsafe, the labels found.
     */
    virtual std::vector<GuardVerdict> check_batch(
        std::span<const Turns> requests, const Runner& run = {}) = 0;

    /** check_batch() of one request. */
    GuardVerdict check(Turns turns, const Runner& run = {}) {
        return check_batch({&turns, 1}, run).at(0);
    }

    /** @returns Why the last check failed (it threw), or "". */
    virtual std::string last_error() { return {}; }

    /** @returns The name /metrics and the error body use. */
    const std::string& name() const { return name_; }

    /** @returns Whether `v` rejects under this guard's labels. */
    bool rejects(const GuardVerdict& v) const {
        return guard_rejects(v, enabled_);
    }

  protected:
    /**
     * @param name The guard's name.
     * @param enabled The labels that reject; empty means all.
     */
    Guard(std::string name, std::vector<std::string> enabled)
        : name_(std::move(name)), enabled_(std::move(enabled)) {}

  private:
    std::string name_;
    std::vector<std::string> enabled_;
};

/**
 * Llama Guard 3 (kind "llama-guard"): builds its prompt from a
 * request's turns, runs the model greedily for a few tokens and parses
 * its written verdict. One check runs at a time.
 */
class LlamaGuard final : public Guard {
  public:
    /** Maps a guard prompt to the guard's reply text (the test seam;
     *  load() wires the real model in). */
    using Classifier =
        std::function<std::string(const chat::ChatTemplate::Prompt&)>;

    /**
     * @param classify Produces the guard's reply for a prompt.
     * @param enabled Category codes that reject; empty means all.
     * @param name The guard's name.
     */
    LlamaGuard(Classifier classify, std::vector<std::string> enabled,
               std::string name = "llama-guard");

    /**
     * Loads a guard GGUF (a llama-arch Llama Guard 3) on `backend`,
     * with its own KV pool of `n_blocks` 16-token blocks.
     *
     * @param path The guard GGUF.
     * @param backend The backend the chat model runs on.
     * @param n_blocks KV blocks; a prompt past them fails the check.
     * @param enabled Category codes that reject; empty means all.
     * @param name The guard's name.
     * @throws std::runtime_error (or gguf::Error) when it cannot load.
     */
    static std::unique_ptr<LlamaGuard> load(
        const std::string& path, const backend::Backend& backend,
        std::uint32_t n_blocks, std::vector<std::string> enabled,
        std::string name = "llama-guard");

    ~LlamaGuard() override;

    std::string_view kind() const override { return "llama-guard"; }

    /** Judges each request's last user turn; the batch's prompts go
     *  to the guard's engine together. */
    std::vector<GuardVerdict> check_batch(std::span<const Turns> requests,
                                          const Runner& run = {}) override;

    /**
     * @returns The last check's prompt tokens and the positions it
     *     prefilled (the guard's prefix cache served the rest); {0, 0}
     *     for a stand-in guard.
     */
    std::pair<std::uint32_t, std::uint32_t> last_prefill();

    std::string last_error() override;

  private:
    struct Loaded;  // the GGUF, model, tokenizer and engine
    std::unique_ptr<Loaded> loaded_;
    Classifier classify_;
    std::vector<GuardCategory> categories_;
    std::string last_error_;
    std::mutex mu_;
};

/**
 * Batches one guard's checks across threads (i#1156): a check joins a
 * queue; when no batch is running it runs every queued check as one
 * check_batch() at once (a lone check runs at once, with no wait
 * window), and while a batch runs, new checks wait and form the next
 * one.
 */
class GuardBatcher {
  public:
    explicit GuardBatcher(Guard& guard) : guard_(guard) {}

    /**
     * @param turns The request's messages.
     * @param run Where the guard runs (see Guard::check_batch).
     * @returns The request's verdict.
     */
    GuardVerdict check(Guard::Turns turns, const Guard::Runner& run);

    /** @returns The batches run so far (tests and /metrics). */
    std::uint64_t batches() const { return batches_.load(); }

    /** @returns The checks waiting for the next batch (tests). */
    std::size_t queued() {
        std::lock_guard<std::mutex> lk(mu_);
        return queue_.size();
    }

  private:
    struct Pending {
        Guard::Turns turns;
        GuardVerdict verdict;
        bool done = false;
    };
    Guard& guard_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<Pending*> queue_;
    bool busy_ = false;
    std::atomic<std::uint64_t> batches_{0};
};

/** One --guard flag: which guard to load and how it rejects. */
struct GuardSpec {
    std::string kind;   ///< "llama-guard" or "prompt-guard"
    std::string model;  ///< the GGUF path
    /** /metrics and error-body name; default: the GGUF's file name
     *  without ".gguf". */
    std::string name;
    /** The labels that reject; empty means all (llama-guard only: a
     *  classifier kind needs them). */
    std::vector<std::string> labels;
    /** Per label, the score from which it rejects (classifier kinds,
     *  NAME@T); NaN takes `threshold`. */
    std::vector<float> thresholds;
    /** threshold=: the default for labels without their own. */
    float threshold = 0.5f;
    /** max_windows= (classifier kinds): the most encoder windows one
     *  request may take; past it the check fails closed (the guard
     *  runs on the chat engine's worker, so this bounds its stall). */
    std::size_t max_windows = 64;
};

/**
 * Parses a --guard value: comma-separated key=value pairs, with
 * `kind` and `model` required and `name`, `labels` (colon-separated,
 * e.g. labels=S1:S9) and `threshold` optional. For a classifier kind
 * (prompt-guard) labels are required and each may carry its own
 * threshold, labels=INJECTION@0.9:JAILBREAK; threshold= sets the
 * default (0.5). A label rejects when its score reaches its threshold.
 *
 * @param text The flag's value, e.g.
 *     "kind=llama-guard,model=/m/lg.gguf,labels=S1:S9".
 * @returns The spec, with a default name filled in.
 * @throws std::invalid_argument on an unknown kind or key, a missing
 *     kind or model, a pair without '=', a classifier kind without
 *     labels, a threshold on llama-guard, or a threshold that is not a
 *     number in (0, 1].
 */
GuardSpec parse_guard_spec(std::string_view text);

/**
 * Loads the guard a spec names.
 *
 * @param spec The guard to load.
 * @param backend The backend the chat model runs on.
 * @param n_blocks KV blocks for a guard that generates.
 * @param encoder_gpu GPU ops for an encoder classifier's large passes
 *     (kind=classifier), or null.
 * @returns The guard.
 * @throws std::invalid_argument on an unknown kind; what the kind's
 *     loader throws when the model cannot load.
 */
std::unique_ptr<Guard> make_guard(const GuardSpec& spec,
                                  const backend::Backend& backend,
                                  std::uint32_t n_blocks,
                                  const backend::Ops* encoder_gpu = nullptr);

}  // namespace locus::server
