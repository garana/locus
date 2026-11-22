#include "locus/server/guard.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "locus/engine/engine.hpp"
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
    if (enabled.empty()) return true;
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

struct Guard::Loaded {
    gguf::GgufFile file;
    model::TransformerModel model;
    std::unique_ptr<tok::Tokenizer> tok;
    engine::Engine engine;

    /** <|eot_id|>'s id when the vocabulary has it as one token, else
     *  EOS: Llama Guard 3 ends its verdict there, and a GGUF may name
     *  <|end_of_text|> as EOS instead. */
    tok::TokenId eot;

    Loaded(const std::string& path, const backend::Backend& backend,
           std::uint32_t n_blocks)
        : file(gguf::GgufFile::open(path)),
          model(model::TransformerModel::load(file)),
          tok(tok::tokenizer_from_gguf(file)),
          engine(use(model, backend), tok->eos_id(), config(n_blocks)),
          eot(end_of_turn(*tok)) {}

    static tok::TokenId end_of_turn(const tok::Tokenizer& t) {
        const auto ids = t.encode("<|eot_id|>", false);
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

    /** The last check's prompt tokens, and the positions it prefilled
     *  (the rest came from the prefix cache). */
    std::uint32_t last_prompt = 0;
    std::uint32_t last_prefilled = 0;

    /** The most tokens a verdict may take: "unsafe" and all 14 codes
     *  fit well inside it. */
    static constexpr std::uint32_t kMaxReply = 64;

    /**
     * The guard's greedy reply, up to its first EOS or <|eot_id|>. A
     * reply with neither within kMaxReply tokens was cut, and a cut
     * list of codes would parse as a shorter one, so it throws (and
     * the check fails closed).
     */
    std::string reply(const chat::ChatTemplate::Prompt& p) {
        const auto ids = tok->encode_literal(p.text, false, p.content);
        // The prefix cache adopts an entry only when the whole cached
        // prompt starts the new one, and a guard prompt ends with the
        // instruction after the conversation. So the conversation part
        // (up to the end of the last turn) runs first as a request of
        // its own: the full prompt then adopts it, and the next turn's
        // conversation part adopts it too, prefilling only the new
        // turn. Its tokens are cut where the two encodings agree.
        std::uint32_t prefilled = 0;
        if (!p.content.empty()) {
            const std::size_t conv_end = p.content.back().end + 2;
            const auto head_text = std::string_view(p.text).substr(
                0, std::min(conv_end, p.text.size()));
            const auto head =
                tok->encode_literal(head_text, false, p.content);
            std::size_t k = 0;
            while (k < head.size() && k < ids.size() && head[k] == ids[k]) {
                ++k;
            }
            if (k >= 2 && k < ids.size()) {
                run(std::vector<tok::TokenId>(ids.begin(), ids.begin() + k),
                    1, prefilled);
            }
        }
        const auto gen = run(ids, kMaxReply, prefilled);
        last_prompt = static_cast<std::uint32_t>(ids.size());
        last_prefilled = prefilled;
        return guard_reply_text(gen, tok->eos_id(), eot, *tok);
    }

    /** Runs one request to its end; adds its prefilled positions
     *  (prompt minus what the prefix cache served) to `prefilled`. */
    std::vector<tok::TokenId> run(std::vector<tok::TokenId> ids,
                                  std::uint32_t max_new,
                                  std::uint32_t& prefilled) {
        const auto n = static_cast<std::uint32_t>(ids.size());
        const std::uint64_t id = engine.submit(std::move(ids), max_new);
        engine.run_to_completion();
        const engine::Request* r = engine.get(id);
        if (r == nullptr || r->status != engine::Status::kDone) {
            const std::string why =
                r != nullptr ? r->error : "unknown request";
            engine.release(id);
            throw std::runtime_error("guard: " + why);
        }
        prefilled += n - std::min(n, r->reused_prefix_tokens);
        std::vector<tok::TokenId> gen = r->generated;
        engine.release(id);
        return gen;
    }
};

Guard::Guard(Classifier classify, std::vector<std::string> enabled)
    : classify_(std::move(classify)),
      enabled_(std::move(enabled)),
      categories_(llama_guard3_1b_categories()) {}

Guard::~Guard() = default;

std::unique_ptr<Guard> Guard::load(const std::string& path,
                                   const backend::Backend& backend,
                                   std::uint32_t n_blocks,
                                   std::vector<std::string> enabled) {
    auto loaded = std::make_unique<Loaded>(path, backend, n_blocks);
    Loaded* raw = loaded.get();
    auto g = std::make_unique<Guard>(
        [raw](const chat::ChatTemplate::Prompt& p) { return raw->reply(p); },
        std::move(enabled));
    g->loaded_ = std::move(loaded);
    return g;
}

std::string Guard::last_error() {
    std::lock_guard<std::mutex> lock(mu_);
    return last_error_;
}

std::pair<std::uint32_t, std::uint32_t> Guard::last_prefill() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!loaded_) return {0, 0};
    return {loaded_->last_prompt, loaded_->last_prefilled};
}

GuardVerdict Guard::check(std::span<const chat::Message> turns,
                          const Runner& run) {
    const auto prompt = llama_guard3_prompt(turns, categories_);
    std::lock_guard<std::mutex> lock(mu_);
    try {
        std::string reply;
        const auto work = [&] { reply = classify_(prompt); };
        if (run) {
            run(work);
        } else {
            work();
        }
        last_error_.clear();
        return parse_guard_verdict(reply);
    } catch (const std::exception& e) {
        last_error_ = e.what();
        return {};  // kMalformed: fails closed
    }
}

}  // namespace locus::server
