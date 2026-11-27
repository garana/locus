#include "locus/server/guard.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>

#include "locus/engine/engine.hpp"
#include "locus/server/classifier_guard.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::server {

namespace {

/** s without leading and trailing whitespace. */
std::string_view trim(std::string_view s) {
    const auto ws = [](char c) {
        return std::isspace(static_cast<unsigned char>(c)) != 0;
    };
    while (!s.empty() && ws(s.front())) s.remove_prefix(1);
    while (!s.empty() && ws(s.back())) s.remove_suffix(1);
    return s;
}

/** Whether s is a category code: "S" then one or more digits. */
bool is_code(std::string_view s) {
    if (s.size() < 2 || s[0] != 'S') return false;
    return std::all_of(s.begin() + 1, s.end(), [](char c) {
        return std::isdigit(static_cast<unsigned char>(c)) != 0;
    });
}

}  // namespace

GuardVerdict parse_guard_verdict(std::string_view reply) {
    GuardVerdict v;
    const std::string_view text = trim(reply);
    const std::size_t nl = text.find('\n');
    const std::string_view first = trim(text.substr(0, nl));
    if (first == "safe" && nl == std::string_view::npos) {
        v.kind = GuardVerdict::Kind::kSafe;
        return v;
    }
    if (first != "unsafe" || nl == std::string_view::npos) {
        return v;  // kMalformed
    }
    const std::string_view rest = trim(text.substr(nl + 1));
    if (rest.find('\n') != std::string_view::npos) {
        return v;  // more than two lines
    }
    std::size_t at = 0;
    while (at <= rest.size()) {
        const std::size_t comma = rest.find(',', at);
        const std::string_view code = trim(rest.substr(
            at, comma == std::string_view::npos ? std::string_view::npos
                                                : comma - at));
        if (!is_code(code)) {
            v.categories.clear();
            return v;  // kMalformed
        }
        v.categories.emplace_back(code);
        if (comma == std::string_view::npos) break;
        at = comma + 1;
    }
    v.kind = GuardVerdict::Kind::kUnsafe;
    return v;
}

bool guard_rejects(const GuardVerdict& v,
                   std::span<const std::string> enabled) {
    switch (v.kind) {
        case GuardVerdict::Kind::kSafe: return false;
        case GuardVerdict::Kind::kMalformed: return true;
        case GuardVerdict::Kind::kUnsafe: break;
    }
    if (enabled.empty() || v.rejects_any) return true;
    return std::any_of(v.categories.begin(), v.categories.end(),
                       [&](const std::string& c) {
                           return std::find(enabled.begin(), enabled.end(),
                                            c) != enabled.end();
                       });
}

chat::ChatTemplate::Prompt llama_guard3_prompt(
    std::span<const chat::Message> turns,
    std::span<const GuardCategory> categories) {
    // Meta's Llama Guard 3 layout for the user role. The blank lines
    // follow its model card; they are checked against the guard
    // GGUF's own chat template once one is in the model store (i#1106).
    chat::ChatTemplate::Prompt out;
    std::string& p = out.text;
    const auto turn = [&](std::string_view who, const std::string& text) {
        p += who;
        out.content.push_back({p.size(), p.size() + text.size()});
        p += text;
        p += "\n\n";
    };
    p =
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "Task: Check if there is unsafe content in 'User' messages in "
        "conversations according our safety policy with the below "
        "categories.\n\n"
        "<BEGIN UNSAFE CONTENT CATEGORIES>\n";
    for (const GuardCategory& c : categories) {
        p += c.code + ": " + c.name + ".\n";
    }
    p += "<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\n\n";
    for (const chat::Message& m : turns) {
        if (m.role == "user") {
            turn("User: ", m.content);
        } else if (m.role == "assistant") {
            turn("Agent: ", m.content);
        }
    }
    p +=
        "<END CONVERSATION>\n\n"
        "Provide your safety assessment for ONLY THE LAST User message "
        "in the above conversation:\n"
        " - First line must read 'safe' or 'unsafe'.\n"
        " - If unsafe, a second line must include a comma-separated list "
        "of violated categories.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return out;
}

std::vector<GuardCategory> llama_guard3_1b_categories() {
    return {
        {"S1", "Violent Crimes"},
        {"S2", "Non-Violent Crimes"},
        {"S3", "Sex-Related Crimes"},
        {"S4", "Child Sexual Exploitation"},
        {"S5", "Defamation"},
        {"S6", "Specialized Advice"},
        {"S7", "Privacy"},
        {"S8", "Intellectual Property"},
        {"S9", "Indiscriminate Weapons"},
        {"S10", "Hate"},
        {"S11", "Suicide & Self-Harm"},
        {"S12", "Sexual Content"},
        {"S13", "Elections"},
    };
}

namespace {

// Qwen3Guard-Gen's tokenizer.chat_template, for a conversation whose last
// turn is the user's (i#1106): the text before the conversation and the
// text after it, as its Jinja literals hold them.
constexpr std::string_view kQwen3GuardHead =
    "<|im_start|>user\n"
    "# Task:\n"
    "1) Evaluate whether THE LAST USER's query belongs to safe or unsafe "
    "content based on our safety policy.\n"
    "2) Identify the category of unsafe content found.\n"
    "\n"
    "<BEGIN SAFETY POLICY>\n"
    "**Safety: Safe**: The USER's query does not contain any prohibited "
    "content or content that could lead to prohibited content.\n"
    "**Safety: Unsafe**: The USER's query contains prohibited content.\n"
    "**Safety: Controversial**: The USER's query does not explicitly "
    "contain prohibited content, but its intent, context, or potential "
    "responses could be misinterpreted, misused, or exploited to generate "
    "unsafe content under certain conditions.\n"
    "<END SAFETY POLICY>\n"
    "\n"
    "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
    "Violent.\n"
    "Non-violent Illegal Acts.\n"
    "Sexual Content or Sexual Acts.\n"
    "PII.\n"
    "Suicide & Self-Harm.\n"
    "Unethical Acts.\n"
    "Politically Sensitive Topics.\n"
    "Copyright Violation.\n"
    "Jailbreak.\n"
    "<END UNSAFE CONTENT CATEGORIES>\n"
    "\n"
    "<BEGIN CONVERSATION>\n"
    "\n";
constexpr std::string_view kQwen3GuardTail =
    "\n"
    "\n"
    "<END CONVERSATION>\n"
    "\n"
    "Provide your safety assessment for ONLY THE LAST **USER's query** in "
    "the above conversation:\n"
    " - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', "
    "'Safety: Controversial'.\n"
    " - The second line should start with 'Categories:' followed by a list "
    "of any unsafe content categories, separated by commas. If the content "
    "is safe, use 'Categories: None'.<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n"
    "\n"
    "</think>\n"
    "\n";

}  // namespace

chat::ChatTemplate::Prompt qwen3_guard_prompt(
    std::span<const chat::Message> turns) {
    chat::ChatTemplate::Prompt out;
    std::string& p = out.text;
    p = kQwen3GuardHead;
    const auto put = [&](const std::string& text) {
        out.content.push_back({p.size(), p.size() + text.size()});
        p += text;
    };
    // The template's loop: the first turn (system or user) as "USER: ";
    // a user turn right after a system turn joined by a blank line with
    // no prefix; then assistant and user turns on their own, each after a
    // blank line. Other roles write nothing.
    for (std::size_t i = 0; i < turns.size(); ++i) {
        const chat::Message& m = turns[i];
        if (i == 0) {
            if (m.role == "system" || m.role == "user") {
                p += "USER: ";
                put(m.content);
            }
        } else if (turns[i - 1].role == "system" && m.role == "user") {
            p += "\n\n";
            put(m.content);
        } else if (m.role == "assistant") {
            p += "\n\nASSISTANT: ";
            put(m.content);
        } else if (m.role == "user") {
            p += "\n\nUSER: ";
            put(m.content);
        }
    }
    p += kQwen3GuardTail;
    return out;
}

std::vector<GuardCategory> qwen3_guard_categories() {
    std::vector<GuardCategory> out;
    for (const char* n :
         {"Violent", "Non-violent Illegal Acts",
          "Sexual Content or Sexual Acts", "PII", "Suicide & Self-Harm",
          "Unethical Acts", "Politically Sensitive Topics",
          "Copyright Violation", "Jailbreak"}) {
        out.push_back({n, n});
    }
    return out;
}

GuardVerdict parse_qwen3_guard_verdict(std::string_view reply,
                                       bool controversial_rejects) {
    GuardVerdict v;
    const std::string_view text = trim(reply);
    const std::size_t nl = text.find('\n');
    if (nl == std::string_view::npos) return v;  // kMalformed
    const std::string_view first = trim(text.substr(0, nl));
    const std::string_view second = trim(text.substr(nl + 1));
    if (second.find('\n') != std::string_view::npos) return v;
    constexpr std::string_view kSafety = "Safety: ";
    constexpr std::string_view kCats = "Categories:";
    if (first.substr(0, kSafety.size()) != kSafety ||
        second.substr(0, kCats.size()) != kCats) {
        return v;
    }
    const std::string_view label = first.substr(kSafety.size());
    const std::string_view list = trim(second.substr(kCats.size()));
    std::vector<std::string> cats;
    if (list != "None") {
        const auto known = qwen3_guard_categories();
        std::size_t at = 0;
        while (at <= list.size()) {
            const std::size_t comma = list.find(',', at);
            const std::string_view name = trim(list.substr(
                at, comma == std::string_view::npos ? std::string_view::npos
                                                    : comma - at));
            if (std::none_of(known.begin(), known.end(),
                             [&](const GuardCategory& c) {
                                 return c.name == name;
                             })) {
                return v;  // kMalformed: not one of the nine
            }
            cats.emplace_back(name);
            if (comma == std::string_view::npos) break;
            at = comma + 1;
        }
    }
    if (label == "Safe" && cats.empty()) {
        v.kind = GuardVerdict::Kind::kSafe;
    } else if (label == "Unsafe" && !cats.empty()) {
        v.kind = GuardVerdict::Kind::kUnsafe;
        v.categories = std::move(cats);
    } else if (label == "Controversial") {
        v.kind = controversial_rejects ? GuardVerdict::Kind::kUnsafe
                                       : GuardVerdict::Kind::kSafe;
        if (controversial_rejects) {
            v.categories = std::move(cats);
            v.rejects_any = true;  // whatever labels= enables
        }
    }
    return v;
}

const GuardFormat& llama_guard3_format() {
    static const GuardFormat f{
        "llama-guard",
        [](std::span<const chat::Message> turns) {
            return llama_guard3_prompt(turns, llama_guard3_1b_categories());
        },
        [](std::string_view reply, bool) {
            return parse_guard_verdict(reply);
        },
        "<|eot_id|>"};
    return f;
}

const GuardFormat& qwen3_guard_format() {
    static const GuardFormat f{"qwen3-guard", &qwen3_guard_prompt,
                               &parse_qwen3_guard_verdict, "<|im_end|>"};
    return f;
}

std::string guard_reply_text(std::span<const tok::TokenId> generated,
                             tok::TokenId eos, tok::TokenId eot,
                             const tok::Tokenizer& tok) {
    const auto stop = std::find_if(
        generated.begin(), generated.end(),
        [&](tok::TokenId t) { return t == eos || t == eot; });
    if (stop == generated.end()) {
        throw std::runtime_error("guard: reply cut before its end");
    }
    return tok.decode(generated.first(
        static_cast<std::size_t>(stop - generated.begin())));
}

struct LlamaGuard::Loaded {
    gguf::GgufFile file;
    model::TransformerModel model;
    std::unique_ptr<tok::Tokenizer> tok;
    engine::Engine engine;

    /** The format's end-of-turn marker's id when the vocabulary has it
     *  as one token, else EOS: Llama Guard 3 ends its verdict at
     *  <|eot_id|> (a GGUF may name <|end_of_text|> as EOS instead),
     *  Qwen3Guard at <|im_end|>. */
    tok::TokenId eot;

    Loaded(const std::string& path, const backend::Backend& backend,
           std::uint32_t n_blocks, std::string_view eot_marker)
        : file(gguf::GgufFile::open(path)),
          model(model::TransformerModel::load(file)),
          tok(tok::tokenizer_from_gguf(file)),
          engine(use(model, backend), tok->eos_id(), config(n_blocks)),
          eot(end_of_turn(*tok, eot_marker)) {}

    static tok::TokenId end_of_turn(const tok::Tokenizer& t,
                                    std::string_view marker) {
        const auto ids = t.encode(std::string(marker), false);
        return ids.size() == 1 ? ids[0] : t.eos_id();
    }

    static const model::TransformerModel& use(model::TransformerModel& m,
                                              const backend::Backend& b) {
        m.use_backend(b);
        return m;
    }

    static engine::Engine::Config config(std::uint32_t n_blocks) {
        engine::Engine::Config c;
        c.n_blocks = n_blocks;
        // Turn N's guard prompt starts with turn N-1's up to the end
        // of its last turn, so a chat's later checks prefill only the
        // new turn and the closing lines.
        c.prefix_cache = true;
        return c;
    }

    /** The last batch's prompt tokens, and the positions it prefilled
     *  (the rest came from the prefix cache). */
    std::uint32_t last_prompt = 0;
    std::uint32_t last_prefilled = 0;

    /** The most tokens a verdict may take: "unsafe" and all 14 codes
     *  fit well inside it. */
    static constexpr std::uint32_t kMaxReply = 64;

    /** One prompt's reply text, or why it has none. */
    struct Reply {
        std::string text;
        std::string error;  ///< non-empty: no reply
    };

    /**
     * The guard's greedy replies to a batch of prompts, each up to its
     * first EOS or <|eot_id|>, run as one engine batch. A reply with
     * neither within kMaxReply tokens was cut, and a cut list of codes
     * would parse as a shorter one, so it gets an error (and its check
     * fails closed).
     *
     * The prefix cache adopts an entry only when the whole cached
     * prompt starts the new one, and a guard prompt ends with the
     * instruction after the conversation. So each prompt's
     * conversation part (up to the end of its last turn, cut where its
     * tokens agree with the full prompt's) runs first, the whole
     * batch's together: the full prompts then adopt them, and the next
     * turn's conversation part adopts them too, prefilling only the
     * new turn.
     */
    std::vector<Reply> replies(
        std::span<const chat::ChatTemplate::Prompt> prompts) {
        const std::size_t n = prompts.size();
        std::vector<Reply> out(n);
        std::vector<std::vector<tok::TokenId>> ids(n), heads(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto& p = prompts[i];
            ids[i] = tok->encode_literal(p.text, false, p.content);
            if (p.content.empty()) continue;
            const std::size_t conv_end = p.content.back().end + 2;
            const auto head = tok->encode_literal(
                std::string_view(p.text).substr(
                    0, std::min(conv_end, p.text.size())),
                false, p.content);
            std::size_t k = 0;
            while (k < head.size() && k < ids[i].size() &&
                   head[k] == ids[i][k]) {
                ++k;
            }
            if (k >= 2 && k < ids[i].size()) {
                heads[i].assign(ids[i].begin(), ids[i].begin() + k);
            }
        }
        std::uint32_t prefilled = 0, prompt = 0;
        // The conversation parts, then the full prompts.
        run(heads, 1, prefilled, out);
        for (std::size_t i = 0; i < n; ++i) {
            if (!out[i].error.empty()) ids[i].clear();
            prompt += static_cast<std::uint32_t>(ids[i].size());
        }
        const auto gens = run(ids, kMaxReply, prefilled, out);
        for (std::size_t i = 0; i < n; ++i) {
            if (!out[i].error.empty() || ids[i].empty()) continue;
            try {
                out[i].text =
                    guard_reply_text(gens[i], tok->eos_id(), eot, *tok);
            } catch (const std::exception& e) {
                out[i].error = e.what();
            }
        }
        last_prompt = prompt;
        last_prefilled = prefilled;
        return out;
    }

    /**
     * Submits every non-empty prompt of `batch` and runs the engine
     * until all are done. Adds their prefilled positions (prompt minus
     * what the prefix cache served) to `prefilled`; a request that does
     * not finish sets its item's error.
     *
     * @returns Each item's generated tokens (empty for a skipped one).
     */
    std::vector<std::vector<tok::TokenId>> run(
        const std::vector<std::vector<tok::TokenId>>& batch,
        std::uint32_t max_new, std::uint32_t& prefilled,
        std::vector<Reply>& out) {
        std::vector<std::optional<std::uint64_t>> rid(batch.size());
        for (std::size_t i = 0; i < batch.size(); ++i) {
            if (!batch[i].empty() && out[i].error.empty()) {
                rid[i] = engine.submit(batch[i], max_new);
            }
        }
        engine.run_to_completion();
        std::vector<std::vector<tok::TokenId>> gens(batch.size());
        for (std::size_t i = 0; i < batch.size(); ++i) {
            if (!rid[i]) continue;
            const engine::Request* r = engine.get(*rid[i]);
            if (r == nullptr || r->status != engine::Status::kDone) {
                out[i].error = "guard: " + (r != nullptr
                                                ? r->error
                                                : std::string("lost request"));
            } else {
                const auto np = static_cast<std::uint32_t>(batch[i].size());
                prefilled += np - std::min(np, r->reused_prefix_tokens);
                gens[i] = r->generated;
            }
            engine.release(*rid[i]);
        }
        return gens;
    }
};

LlamaGuard::LlamaGuard(Classifier classify, std::vector<std::string> enabled,
                       std::string name, const GuardFormat& format,
                       bool controversial_rejects)
    : Guard(std::move(name), std::move(enabled)),
      classify_(std::move(classify)),
      format_(format),
      controversial_rejects_(controversial_rejects) {}

LlamaGuard::~LlamaGuard() = default;

std::unique_ptr<LlamaGuard> LlamaGuard::load(
    const std::string& path, const backend::Backend& backend,
    std::uint32_t n_blocks, std::vector<std::string> enabled,
    std::string name, const GuardFormat& format,
    bool controversial_rejects) {
    auto loaded =
        std::make_unique<Loaded>(path, backend, n_blocks, format.eot);
    Loaded* raw = loaded.get();
    auto g = std::make_unique<LlamaGuard>(
        [raw](const chat::ChatTemplate::Prompt& p) {
            auto r = raw->replies({&p, 1});
            if (!r[0].error.empty()) throw std::runtime_error(r[0].error);
            return r[0].text;
        },
        std::move(enabled), std::move(name), format, controversial_rejects);
    g->loaded_ = std::move(loaded);
    return g;
}

std::string LlamaGuard::last_error() {
    std::lock_guard<std::mutex> lock(mu_);
    return last_error_;
}

std::pair<std::uint32_t, std::uint32_t> LlamaGuard::last_prefill() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!loaded_) return {0, 0};
    return {loaded_->last_prompt, loaded_->last_prefilled};
}

namespace {

/** A threshold for a label without its own: GuardSpec::threshold. */
constexpr float kNoThreshold = std::numeric_limits<float>::quiet_NaN();

/** @returns text as a probability threshold in (0, 1].
 *  @throws std::invalid_argument when it is not one. */
float parse_threshold(const std::string& text) {
    char* end = nullptr;
    const float t = std::strtof(text.c_str(), &end);
    if (text.empty() || end != text.c_str() + text.size() || !(t > 0.0f) ||
        t > 1.0f) {
        throw std::invalid_argument("--guard: threshold '" + text +
                                    "' is not a number in (0, 1]");
    }
    return t;
}

}  // namespace

GuardSpec parse_guard_spec(std::string_view text) {
    GuardSpec s;
    bool have_labels = false, have_threshold = false, have_max_windows = false;
    bool have_controversial = false;
    std::size_t at = 0;
    while (at <= text.size()) {
        const std::size_t comma = text.find(',', at);
        const std::string_view pair = text.substr(
            at, comma == std::string_view::npos ? std::string_view::npos
                                                : comma - at);
        const std::size_t eq = pair.find('=');
        if (eq == std::string_view::npos) {
            throw std::invalid_argument("--guard: '" + std::string(pair) +
                                        "' is not key=value");
        }
        const std::string_view key = pair.substr(0, eq);
        const std::string value(pair.substr(eq + 1));
        if (key == "kind") {
            s.kind = value;
        } else if (key == "model") {
            s.model = value;
        } else if (key == "name") {
            s.name = value;
        } else if (key == "labels") {
            have_labels = true;
            std::size_t b = 0;
            while (b <= value.size()) {
                const std::size_t c = value.find(':', b);
                const std::string l = value.substr(
                    b, c == std::string::npos ? std::string::npos : c - b);
                if (!l.empty()) {
                    // NAME or NAME@threshold (classifier kinds).
                    const std::size_t at_sign = l.find('@');
                    if (at_sign == std::string::npos) {
                        s.labels.push_back(l);
                        s.thresholds.push_back(kNoThreshold);
                    } else {
                        s.labels.push_back(l.substr(0, at_sign));
                        s.thresholds.push_back(
                            parse_threshold(l.substr(at_sign + 1)));
                        if (s.labels.back().empty()) {
                            throw std::invalid_argument(
                                "--guard: '" + l + "' names no label");
                        }
                    }
                }
                if (c == std::string::npos) break;
                b = c + 1;
            }
        } else if (key == "threshold") {
            s.threshold = parse_threshold(value);
            have_threshold = true;
        } else if (key == "max_windows") {
            // A whole positive number: a request past it fails closed.
            std::size_t w = 0;
            const auto [end, ec] =
                std::from_chars(value.data(), value.data() + value.size(), w);
            if (ec != std::errc{} || end != value.data() + value.size() ||
                w == 0) {
                throw std::invalid_argument("--guard: max_windows '" + value +
                                            "' is not a positive whole number");
            }
            s.max_windows = w;
            have_max_windows = true;
        } else if (key == "embedding") {
            s.embedding = value;
        } else if (key == "controversial") {
            if (value != "reject" && value != "allow") {
                throw std::invalid_argument("--guard: controversial '" +
                                            value +
                                            "' is not reject or allow");
            }
            s.controversial_rejects = value == "reject";
            have_controversial = true;
        } else {
            throw std::invalid_argument("--guard: unknown key '" +
                                        std::string(key) + "'");
        }
        if (comma == std::string_view::npos) break;
        at = comma + 1;
    }
    if (s.kind.empty() || s.model.empty()) {
        throw std::invalid_argument("--guard needs kind= and model=");
    }
    if (s.kind != "llama-guard" && s.kind != "qwen3-guard" &&
        s.kind != "prompt-guard" && s.kind != "classifier" &&
        s.kind != "linear-head") {
        throw std::invalid_argument(
            "--guard: unknown kind '" + s.kind +
            "' (known: llama-guard, qwen3-guard, prompt-guard, classifier, "
            "linear-head)");
    }
    // Only Qwen3Guard writes a Controversial verdict.
    if (have_controversial && s.kind != "qwen3-guard") {
        throw std::invalid_argument(
            "--guard: only kind=qwen3-guard takes controversial=");
    }
    if (s.kind == "qwen3-guard") {
        const auto known = qwen3_guard_categories();
        for (const std::string& l : s.labels) {
            if (std::none_of(known.begin(), known.end(),
                             [&](const GuardCategory& c) {
                                 return c.name == l;
                             })) {
                throw std::invalid_argument(
                    "--guard: '" + l + "' is not a Qwen3Guard category "
                    "(Violent, Non-violent Illegal Acts, Sexual Content or "
                    "Sexual Acts, PII, Suicide & Self-Harm, Unethical Acts, "
                    "Politically Sensitive Topics, Copyright Violation, "
                    "Jailbreak)");
            }
        }
    }
    // A linear head scores the vectors of the embedding model it was
    // trained on; no other kind takes one.
    if ((s.kind == "linear-head") != !s.embedding.empty()) {
        throw std::invalid_argument(
            s.kind == "linear-head"
                ? "--guard: kind=linear-head needs embedding= (the "
                  "embedding GGUF its head was trained on)"
                : "--guard: only kind=linear-head takes embedding=");
    }
    if (have_labels && s.labels.empty()) {
        throw std::invalid_argument("--guard: labels= names no label");
    }
    const bool scores = s.kind != "llama-guard" && s.kind != "qwen3-guard";
    if (!scores) {
        // A generating guard writes its categories: no scores to
        // compare with a threshold.
        if (have_threshold ||
            std::any_of(s.thresholds.begin(), s.thresholds.end(),
                        [](float t) { return !std::isnan(t); })) {
            throw std::invalid_argument("--guard: kind=" + s.kind +
                                        " takes no threshold");
        }
        if (have_max_windows) {
            throw std::invalid_argument("--guard: kind=" + s.kind +
                                        " takes no max_windows");
        }
    } else if (s.labels.empty()) {
        // "All labels" would reject on the benign label too.
        throw std::invalid_argument(
            "--guard: kind=" + s.kind + " needs labels= (the labels that "
            "reject, e.g. labels=INJECTION@0.9:JAILBREAK)");
    }
    if (s.name.empty()) {
        const std::size_t slash = s.model.find_last_of('/');
        s.name = s.model.substr(slash == std::string::npos ? 0 : slash + 1);
        if (s.name.size() > 5 &&
            s.name.compare(s.name.size() - 5, 5, ".gguf") == 0) {
            s.name.resize(s.name.size() - 5);
        }
    }
    return s;
}

std::unique_ptr<Guard> make_guard(const GuardSpec& spec,
                                  const backend::Backend& backend,
                                  std::uint32_t n_blocks,
                                  const backend::Ops* encoder_gpu) {
    if (spec.kind == "llama-guard") {
        return LlamaGuard::load(spec.model, backend, n_blocks, spec.labels,
                                spec.name);
    }
    if (spec.kind == "qwen3-guard") {
        return LlamaGuard::load(spec.model, backend, n_blocks, spec.labels,
                                spec.name, qwen3_guard_format(),
                                spec.controversial_rejects);
    }
    if (spec.kind == "prompt-guard") return load_prompt_guard(spec);
    if (spec.kind == "classifier") {
        return load_encoder_classifier(spec, encoder_gpu);
    }
    if (spec.kind == "linear-head") return load_linear_head(spec, encoder_gpu);
    throw std::invalid_argument("unknown guard kind '" + spec.kind + "'");
}

std::vector<GuardVerdict> LlamaGuard::check_batch(
    std::span<const Turns> requests, const Runner& run) {
    std::vector<chat::ChatTemplate::Prompt> prompts;
    prompts.reserve(requests.size());
    for (const Turns& t : requests) {
        prompts.push_back(format_.prompt(t));
    }
    // A loaded guard runs the batch as one engine batch; the test
    // seam's classifier, prompt by prompt.
    std::vector<Loaded::Reply> got(prompts.size());
    const auto work = [&] {
        if (loaded_) {
            got = loaded_->replies(prompts);
            return;
        }
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            try {
                got[i].text = classify_(prompts[i]);
            } catch (const std::exception& e) {
                got[i].error = e.what();
            }
        }
    };
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<GuardVerdict> out(prompts.size());  // kMalformed
    try {
        if (run) {
            run(work);
        } else {
            work();
        }
    } catch (const std::exception& e) {
        last_error_ = e.what();
        for (GuardVerdict& v : out) v.error = last_error_;
        return out;  // the whole batch fails closed
    }
    last_error_.clear();
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i].error.empty()) {
            out[i] = format_.parse(got[i].text, controversial_rejects_);
            if (out[i].kind == GuardVerdict::Kind::kMalformed) {
                out[i].error = "no verdict in the guard's reply";
            }
        } else {
            last_error_ = got[i].error;
            out[i].error = got[i].error;
        }
    }
    return out;
}

GuardVerdict GuardBatcher::check(Guard::Turns turns,
                                 const Guard::Runner& run) {
    Pending mine{turns, {}, false};
    std::unique_lock<std::mutex> lk(mu_);
    queue_.push_back(&mine);
    while (!mine.done) {
        if (busy_) {
            cv_.wait(lk);
            continue;
        }
        // No batch running: this thread runs every queued check.
        busy_ = true;
        std::vector<Pending*> batch;
        batch.swap(queue_);
        lk.unlock();
        std::vector<Guard::Turns> requests;
        requests.reserve(batch.size());
        for (const Pending* p : batch) requests.push_back(p->turns);
        std::vector<GuardVerdict> verdicts;
        try {
            verdicts = guard_.check_batch(requests, run);
        } catch (...) {
            // Anything thrown fails every check closed; busy_ must be
            // cleared below, or later checks of this guard would wait
            // for ever.
            verdicts.clear();
        }
        batches_.fetch_add(1, std::memory_order_relaxed);
        lk.lock();
        for (std::size_t i = 0; i < batch.size(); ++i) {
            if (i < verdicts.size()) {
                batch[i]->verdict = verdicts[i];
            } else {
                batch[i]->verdict = GuardVerdict{};
                batch[i]->verdict.error = "the guard gave no verdict";
            }
            batch[i]->done = true;
        }
        busy_ = false;
        cv_.notify_all();
    }
    return mine.verdict;
}

}  // namespace locus::server
