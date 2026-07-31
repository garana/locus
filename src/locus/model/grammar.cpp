#include "locus/model/grammar.hpp"

namespace locus::model {

namespace {
bool is_ws(std::uint8_t b) {
    return b == ' ' || b == '\t' || b == '\n' || b == '\r';
}
bool is_digit(std::uint8_t b) { return b >= '0' && b <= '9'; }
bool is_hex(std::uint8_t b) {
    return is_digit(b) || (b >= 'a' && b <= 'f') ||
           (b >= 'A' && b <= 'F');
}
}  // namespace

bool JsonGrammar::begin_value(std::uint8_t b) {
    if (b == '{') {
        stack_.push_back('{');
        state_ = S::ObjStart;
        return true;
    }
    if (b == '[') {
        stack_.push_back('[');
        state_ = S::ArrStart;
        return true;
    }
    if (b == '"') {
        state_ = S::InStr;
        in_key_ = false;
        return true;
    }
    if (b == '-' || is_digit(b)) {
        state_ = S::InNum;
        num_has_digit_ = is_digit(b);
        return true;
    }
    if (b == 't') {
        state_ = S::Kw;
        kw_rest_ = "rue";
        return true;
    }
    if (b == 'f') {
        state_ = S::Kw;
        kw_rest_ = "alse";
        return true;
    }
    if (b == 'n') {
        state_ = S::Kw;
        kw_rest_ = "ull";
        return true;
    }
    return false;
}

void JsonGrammar::finish_value() { state_ = S::AfterVal; }

bool JsonGrammar::feed(std::uint8_t b) {
    for (;;) {  // loops only to reprocess a delimiter after a number
        switch (state_) {
            case S::Val:
            case S::ArrStart:
                if (is_ws(b)) {
                    return true;
                }
                if (state_ == S::ArrStart && b == ']') {
                    stack_.pop_back();
                    finish_value();
                    return true;
                }
                return begin_value(b);
            case S::ObjStart:
                if (is_ws(b)) {
                    return true;
                }
                if (b == '}') {
                    stack_.pop_back();
                    finish_value();
                    return true;
                }
                if (b == '"') {
                    state_ = S::InStr;
                    in_key_ = true;
                    return true;
                }
                return false;
            case S::ObjKey:
                if (is_ws(b)) {
                    return true;
                }
                if (b == '"') {
                    state_ = S::InStr;
                    in_key_ = true;
                    return true;
                }
                return false;
            case S::Colon:
                if (is_ws(b)) {
                    return true;
                }
                if (b == ':') {
                    state_ = S::Val;
                    return true;
                }
                return false;
            case S::InStr:
                if (b == '"') {
                    if (in_key_) {
                        in_key_ = false;
                        state_ = S::Colon;
                    } else {
                        finish_value();
                    }
                    return true;
                }
                if (b == '\\') {
                    state_ = S::StrEsc;
                    return true;
                }
                return b >= 0x20;  // reject bare control chars
            case S::StrEsc:
                if (b == '"' || b == '\\' || b == '/' || b == 'b' ||
                    b == 'f' || b == 'n' || b == 'r' || b == 't') {
                    state_ = S::InStr;
                    return true;
                }
                if (b == 'u') {
                    state_ = S::StrU;
                    u_left_ = 4;
                    return true;
                }
                return false;
            case S::StrU:
                if (is_hex(b)) {
                    if (--u_left_ == 0) {
                        state_ = S::InStr;
                    }
                    return true;
                }
                return false;
            case S::InNum:
                if (is_digit(b)) {
                    num_has_digit_ = true;
                    return true;
                }
                if (b == '.' || b == 'e' || b == 'E' || b == '+' ||
                    b == '-') {
                    return true;
                }
                if (!num_has_digit_) {
                    return false;
                }
                finish_value();
                continue;  // reprocess b as a delimiter
            case S::Kw:
                if (!kw_rest_.empty() &&
                    b == static_cast<std::uint8_t>(kw_rest_.front())) {
                    kw_rest_.erase(kw_rest_.begin());
                    if (kw_rest_.empty()) {
                        finish_value();
                    }
                    return true;
                }
                return false;
            case S::AfterVal:
                if (is_ws(b)) {
                    return true;
                }
                if (stack_.empty()) {
                    return false;  // top-level done: only ws/eos
                }
                if (stack_.back() == '[') {
                    if (b == ',') {
                        state_ = S::Val;
                        return true;
                    }
                    if (b == ']') {
                        stack_.pop_back();
                        finish_value();
                        return true;
                    }
                    return false;
                }
                if (b == ',') {
                    state_ = S::ObjKey;
                    return true;
                }
                if (b == '}') {
                    stack_.pop_back();
                    finish_value();
                    return true;
                }
                return false;
        }
    }
}

bool JsonGrammar::complete() const {
    return stack_.empty() &&
           (state_ == S::AfterVal ||
            (state_ == S::InNum && num_has_digit_));
}

}  // namespace locus::model
