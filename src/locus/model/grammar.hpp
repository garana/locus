#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace locus::model {

/**
 * Incremental JSON validator for constrained decoding: feed the
 * output bytes one at a time; feed() rejects a byte that cannot
 * continue a valid JSON value, and complete() reports whether a full
 * top-level value has been consumed (so decoding may stop).
 *
 * It is copyable and cheap, so a sampler can trial-feed a candidate
 * token's bytes on a copy to decide whether the token is allowed,
 * then feed() the chosen token into the live instance to advance.
 *
 * The structure (objects, arrays, strings, keywords) is validated
 * exactly; number internals are checked leniently (digit/./eE/+-),
 * which is enough to keep output parseable for the tool-call /
 * json-mode use.
 */
class JsonGrammar {
  public:
    /** Advances on byte b; @returns false (state unchanged) if b is
     * not a legal continuation here. */
    bool feed(std::uint8_t b);

    /** @returns true when a complete top-level JSON value is done
     * (trailing whitespace stays complete). */
    bool complete() const;

  private:
    enum class S {
        Val,       // expecting a value (top / after ':' / after ',')
        ArrStart,  // just after '[' -- value or ']'
        ObjStart,  // just after '{' -- '"' key or '}'
        ObjKey,    // after ',' in an object -- '"' key
        Colon,     // after a key string -- ':'
        InStr,     // inside a string
        StrEsc,    // after '\\'
        StrU,      // inside a \uXXXX escape
        InNum,     // inside a number
        Kw,        // matching true / false / null
        AfterVal,  // a value finished -- ',', close, or ws
    };
    bool begin_value(std::uint8_t b);
    void finish_value();

    std::vector<char> stack_;  // '{' or '[' per open container
    S state_ = S::Val;
    bool in_key_ = false;
    bool num_has_digit_ = false;
    int u_left_ = 0;
    std::string kw_rest_;
};

}  // namespace locus::model
