#include "locus/model/grammar_constraint.hpp"

#include <algorithm>

namespace locus::model {

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
    }
}

GrammarConstraint::GrammarConstraint(std::shared_ptr<const Gbnf> g,
                                     std::shared_ptr<const TokenTrie> trie,
                                     tok::TokenId eos)
    : m_(std::move(g)), trie_(std::move(trie)), eos_(eos) {}

void GrammarConstraint::walk(std::uint32_t node, const GbnfMatcher& m) const {
    for (const auto& [b, child] : trie_->nodes()[node].kids) {
        GbnfMatcher next = m;
        if (!next.feed(b)) continue;
        for (const tok::TokenId t : trie_->nodes()[child].tokens) {
            mask_[t] = 1;
        }
        walk(child, next);
    }
}

bool GrammarConstraint::allows(tok::TokenId t) const {
    if (!fresh_) {
        mask_.assign(trie_->pieces().size(), 0);
        walk(0, m_);
        if (static_cast<std::size_t>(eos_) < mask_.size()) {
            mask_[eos_] = m_.accepted() ? 1 : 0;
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
