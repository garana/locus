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

struct Guard::Loaded {
    gguf::GgufFile file;
    model::TransformerModel model;
    std::unique_ptr<tok::Tokenizer> tok;
    engine::Engine engine;

    Loaded(const std::string& path, const backend::Backend& backend,
           std::uint32_t n_blocks)
        : file(gguf::GgufFile::open(path)),
          model(model::TransformerModel::load(file)),
          tok(tok::tokenizer_from_gguf(file)),
          engine(use(model, backend), tok->eos_id(), config(n_blocks)) {}

    static const model::TransformerModel& use(model::TransformerModel& m,
                                              const backend::Backend& b) {
        m.use_backend(b);
        return m;
    }

    static engine::Engine::Config config(std::uint32_t n_blocks) {
        engine::Engine::Config c;
        c.n_blocks = n_blocks;
        return c;
    }

    /** The guard's greedy reply: at most 16 tokens, ending at EOS. */
    std::string reply(const chat::ChatTemplate::Prompt& p) {
        auto ids = tok->encode_literal(p.text, false, p.content);
        const std::uint64_t id = engine.submit(std::move(ids), 16);
        engine.run_to_completion();
        const engine::Request* r = engine.get(id);
        if (r == nullptr || r->status != engine::Status::kDone) {
            const std::string why =
                r != nullptr ? r->error : "unknown request";
            engine.release(id);
            throw std::runtime_error("guard: " + why);
        }
        std::string text = tok->decode(r->generated);
        engine.release(id);
        return text;
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

GuardVerdict Guard::check(std::span<const chat::Message> turns) {
    const auto prompt = llama_guard3_prompt(turns, categories_);
    std::lock_guard<std::mutex> lock(mu_);
    try {
        return parse_guard_verdict(classify_(prompt));
    } catch (const std::exception&) {
        return {};  // kMalformed: fails closed
    }
}

}  // namespace locus::server
