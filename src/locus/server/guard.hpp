#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "locus/chat/template.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::backend {
struct Backend;
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
 * A guard model that judges chat requests (i#1106): it builds the
 * guard's prompt from a request's turns, runs the guard greedily for
 * a few tokens and parses its verdict. One check runs at a time.
 */
class Guard {
  public:
    /** Maps a guard prompt to the guard's reply text (the test seam;
     *  load() wires the real model in). */
    using Classifier =
        std::function<std::string(const chat::ChatTemplate::Prompt&)>;

    /**
     * @param classify Produces the guard's reply for a prompt.
     * @param enabled Category codes that reject; empty means all.
     */
    Guard(Classifier classify, std::vector<std::string> enabled);

    /**
     * Loads a guard GGUF (a llama-arch Llama Guard 3) on `backend`,
     * with its own KV pool of `n_blocks` 16-token blocks.
     *
     * @param path The guard GGUF.
     * @param backend The backend the chat model runs on.
     * @param n_blocks KV blocks; a prompt past them fails the check.
     * @param enabled Category codes that reject; empty means all.
     * @throws std::runtime_error (or gguf::Error) when it cannot load.
     */
    static std::unique_ptr<Guard> load(const std::string& path,
                                       const backend::Backend& backend,
                                       std::uint32_t n_blocks,
                                       std::vector<std::string> enabled);

    ~Guard();

    /**
     * Judges the last user turn of `turns`. A guard that throws or
     * replies with anything but a verdict gives kMalformed.
     *
     * @param turns The request's messages.
     * @returns The verdict.
     */
    GuardVerdict check(std::span<const chat::Message> turns);

    /** @returns Whether `v` rejects under this guard's categories. */
    bool rejects(const GuardVerdict& v) const {
        return guard_rejects(v, enabled_);
    }

  private:
    struct Loaded;  // the GGUF, model, tokenizer and engine
    std::unique_ptr<Loaded> loaded_;
    Classifier classify_;
    std::vector<std::string> enabled_;
    std::vector<GuardCategory> categories_;
    std::mutex mu_;
};

}  // namespace locus::server
