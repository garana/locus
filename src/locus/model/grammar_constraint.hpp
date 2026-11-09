#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "locus/model/gbnf.hpp"
#include "locus/model/sampling.hpp"

namespace locus::model {

/**
 * The vocabulary's byte strings as a trie, built once per model and
 * shared by every constrained request: tokens with a common prefix
 * share the work of checking it against the grammar.
 */
class TokenTrie {
  public:
    /** @param pieces Each token's bytes; empty ones are left out. */
    explicit TokenTrie(std::shared_ptr<const std::vector<std::string>> pieces);

    struct Node {
        std::vector<std::pair<std::uint8_t, std::uint32_t>> kids;
        std::vector<tok::TokenId> tokens;  ///< tokens that end here
    };

    const std::vector<Node>& nodes() const { return nodes_; }
    const std::vector<std::string>& pieces() const { return *pieces_; }

  private:
    std::shared_ptr<const std::vector<std::string>> pieces_;
    std::vector<Node> nodes_;  // nodes_[0] is the root
};

/**
 * Constrains decoding to the strings of a GBNF grammar (i#402). A
 * token is allowed when its bytes continue a string of the grammar;
 * EOS is allowed only when the output so far is a complete string, and
 * is the only choice once nothing more can follow, so the request ends
 * with a full match. The allowed set is computed once per step by a
 * walk of the vocabulary trie.
 */
class GrammarConstraint : public TokenConstraint {
  public:
    GrammarConstraint(std::shared_ptr<const Gbnf> g,
                      std::shared_ptr<const TokenTrie> trie, tok::TokenId eos);

    bool allows(tok::TokenId t) const override;
    void commit(tok::TokenId t) override;

    /** @returns Whether the output so far is a complete string. */
    bool accepted() const { return m_.accepted(); }

  private:
    void walk(std::uint32_t node, const GbnfMatcher& m) const;

    GbnfMatcher m_;
    std::shared_ptr<const TokenTrie> trie_;
    tok::TokenId eos_;
    mutable std::vector<std::uint8_t> mask_;
    mutable bool fresh_ = false;
};

}  // namespace locus::model
