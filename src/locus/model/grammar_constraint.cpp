#include "locus/model/grammar_constraint.hpp"

#include <algorithm>

namespace locus::model {

namespace {

/** Masks kept per request (one byte per vocab token each). */
constexpr std::size_t kCacheMasks = 16;

}  // namespace

TokenTrie::TokenTrie(std::shared_ptr<const std::vector<std::string>> pieces)
    : pieces_(std::move(pieces)), nodes_(1) {
    for (std::size_t t = 0; t < pieces_->size(); ++t) {
        std::uint32_t n = 0;
        for (const char ch : (*pieces_)[t]) {
            const auto b = static_cast<std::uint8_t>(ch);
            auto& kids = nodes_[n].kids;
            const auto it = std::find_if(
                kids.begin(), kids.end(),
                [b](const auto& k) { return k.first == b; });
            if (it != kids.end()) {
                n = it->second;
            } else {
                const auto child = static_cast<std::uint32_t>(nodes_.size());
                nodes_[n].kids.push_back({b, child});
                nodes_.emplace_back();
                n = child;
            }
        }
        if (n != 0) {
            nodes_[n].tokens.push_back(static_cast<tok::TokenId>(t));
        }
        max_len_ = std::max(max_len_, (*pieces_)[t].size());
    }
}

GrammarConstraint::GrammarConstraint(std::shared_ptr<const Gbnf> g,
                                     std::shared_ptr<const TokenTrie> trie,
                                     tok::TokenId eos)
    : m_(std::move(g)), trie_(std::move(trie)), eos_(eos) {}

void GrammarConstraint::walk(std::uint32_t node, const GbnfMatcher& m,
                             std::size_t depth) const {
    GbnfMatcher& next = scratch_[depth];
    for (const auto& [b, child] : trie_->nodes()[node].kids) {
        if (!m.step(b, next)) continue;
        for (const tok::TokenId t : trie_->nodes()[child].tokens) {
            mask_[t] = 1;
        }
        walk(child, next, depth + 1);
    }
}

bool GrammarConstraint::more_than_space() const {
    const auto& pieces = trie_->pieces();
    for (std::size_t t = 0; t < mask_.size(); ++t) {
        if (mask_[t] == 0) continue;
        for (const char ch : pieces[t]) {
            if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') {
                return true;
            }
        }
    }
    return false;
}

bool GrammarConstraint::allows(tok::TokenId t) const {
    if (!fresh_) {
        std::string k = m_.key();
        const auto it = cache_.find(k);
        if (it != cache_.end()) {
            mask_ = it->second;
        } else {
            if (scratch_.empty()) {
                // Sized once: walk() holds references into it.
                scratch_.assign(trie_->max_len() + 1, m_);
            }
            mask_.assign(trie_->pieces().size(), 0);
            walk(0, m_, 0);
            if (m_.accepted() && !more_than_space()) {
                // Complete, and only whitespace may follow (the JSON
                // grammars end every value with optional space): stop
                // here instead of letting the model pad the reply.
                std::fill(mask_.begin(), mask_.end(), 0);
            }
            if (static_cast<std::size_t>(eos_) < mask_.size()) {
                mask_[eos_] = m_.accepted() ? 1 : 0;
            }
            if (cache_.size() >= kCacheMasks) cache_.clear();
            cache_.emplace(std::move(k), mask_);
        }
        fresh_ = true;
    }
    return static_cast<std::size_t>(t) < mask_.size() && mask_[t] != 0;
}

void GrammarConstraint::commit(tok::TokenId t) {
    fresh_ = false;
    if (t == eos_ || static_cast<std::size_t>(t) >= trie_->pieces().size()) {
        return;
    }
    for (const char ch : trie_->pieces()[t]) {
        m_.feed(static_cast<std::uint8_t>(ch));
    }
}

}  // namespace locus::model
