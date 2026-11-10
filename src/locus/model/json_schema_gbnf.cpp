#include "locus/model/json_schema_gbnf.hpp"

#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include "json.hpp"
#include "locus/model/gbnf.hpp"

namespace locus::model {

namespace {

using ojson = nlohmann::ordered_json;

/** The shared rules, and the ones each depends on. */
struct Prim {
    const char* body;
    std::vector<const char*> deps;
};

const std::map<std::string, Prim>& prims() {
    static const std::map<std::string, Prim> p = {
        {"space", {R"(| " " | "\n" [ \t]{0,20})", {}}},
        {"boolean", {R"(("true" | "false") space)", {"space"}}},
        {"null", {R"("null" space)", {"space"}}},
        {"char",
         {R"([^"\\\x7F\x00-\x1F] | [\\] (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))",
          {}}},
        {"string", {R"("\"" char* "\"" space)", {"char", "space"}}},
        {"integral-part", {R"([0] | [1-9] [0-9]{0,15})", {}}},
        {"decimal-part", {R"([0-9]{1,16})", {}}},
        {"integer", {R"("-"? integral-part space)", {"integral-part", "space"}}},
        {"number",
         {R"("-"? integral-part ("." decimal-part)? ([eE] [-+]? integral-part)? space)",
          {"integral-part", "decimal-part", "space"}}},
        {"value",
         {"object | array | string | number | boolean | null",
          {"object", "array", "string", "number", "boolean", "null"}}},
        {"object",
         {R"("{" space (string ":" space value ("," space string ":" space value)*)? "}" space)",
          {"space", "string", "value"}}},
        {"array",
         {R"("[" space (value ("," space value)*)? "]" space)",
          {"space", "value"}}},
    };
    return p;
}

/** Text as a GBNF string literal. */
std::string gbnf_literal(std::string_view s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c < 0x20 || c == 0x7F) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\x%02X", c);
            out += buf;
        } else {
            out += static_cast<char>(c);  // UTF-8 bytes pass through
        }
    }
    return out + "\"";
}

/** Converts one schema (see json_schema_to_gbnf). */
class Converter {
  public:
    explicit Converter(const ojson& root) : root_(root) {}

    std::string run() {
        rules_["root"] = "";
        rules_["root"] = expr(root_, "root");
        std::string out;
        for (const auto& [name, body] : rules_) {
            out += name + " ::= " + body + "\n";
        }
        return out;
    }

    void use_prim(const std::string& name) {
        if (rules_.count(name)) return;
        const Prim& p = prims().at(name);
        rules_[name] = p.body;
        for (const char* d : p.deps) use_prim(d);
    }

  private:
    [[noreturn]] static void fail(const std::string& why) {
        throw std::invalid_argument("json schema: " + why);
    }

    /** A new rule name from a hint, unique, with only [a-z0-9-]. */
    std::string fresh_name(const std::string& hint) {
        std::string n;
        for (char c : hint) {
            const bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                            c == '-';
            if (c >= 'A' && c <= 'Z') {
                n += static_cast<char>(c - 'A' + 'a');
            } else {
                n += ok ? c : '-';
            }
        }
        if (n.empty()) n = "r";
        std::string name = n;
        for (int i = 2; rules_.count(name) || prims().count(name); ++i) {
            name = n + "-" + std::to_string(i);
        }
        return name;
    }

    /** Defines a rule for `body` and returns its name. */
    std::string rule(const std::string& hint, const std::string& body) {
        const std::string n = fresh_name(hint);
        rules_[n] = body;
        return n;
    }

    /** A GBNF element (a rule name or a group) for schema `s`. */
    std::string expr(const ojson& s, const std::string& hint) {
        if (s.is_boolean()) {
            if (!s.get<bool>()) fail("schema false accepts nothing");
            use_prim("value");
            return "value";
        }
        if (!s.is_object()) fail("a schema must be an object or a boolean");
        if (s.contains("$ref")) return ref(s["$ref"]);
        if (s.contains("const")) {
            use_prim("space");
            return "(" + gbnf_literal(s["const"].dump()) + " space)";
        }
        if (s.contains("enum")) {
            const ojson& e = s["enum"];
            if (!e.is_array() || e.empty()) fail("enum must be a non-empty array");
            use_prim("space");
            std::string out = "((";
            for (std::size_t i = 0; i < e.size(); ++i) {
                if (i) out += " | ";
                out += gbnf_literal(e[i].dump());
            }
            return out + ") space)";
        }
        for (const char* k : {"anyOf", "oneOf"}) {
            if (s.contains(k)) {
                const ojson& a = s[k];
                if (!a.is_array() || a.empty()) {
                    fail(std::string(k) + " must be a non-empty array");
                }
                std::string out = "(";
                for (std::size_t i = 0; i < a.size(); ++i) {
                    if (i) out += " | ";
                    out += expr(a[i], hint + "-" + std::to_string(i));
                }
                return out + ")";
            }
        }
        if (s.contains("allOf")) {
            const ojson& a = s["allOf"];
            if (!a.is_array() || a.size() != 1) {
                fail("allOf is supported with one member only");
            }
            return expr(a[0], hint);
        }
        if (s.contains("type")) {
            const ojson& t = s["type"];
            if (t.is_array()) {
                if (t.empty()) fail("type must not be an empty list");
                std::string out = "(";
                for (std::size_t i = 0; i < t.size(); ++i) {
                    if (!t[i].is_string()) fail("type must be a string");
                    if (i) out += " | ";
                    out += typed(t[i].get<std::string>(), s, hint);
                }
                return out + ")";
            }
            if (!t.is_string()) fail("type must be a string or a list");
            return typed(t.get<std::string>(), s, hint);
        }
        if (s.contains("properties") || s.contains("additionalProperties")) {
            return typed("object", s, hint);
        }
        if (s.contains("items") || s.contains("prefixItems")) {
            return typed("array", s, hint);
        }
        use_prim("value");
        return "value";
    }

    std::string ref(const ojson& r) {
        if (!r.is_string()) fail("$ref must be a string");
        const std::string path = r.get<std::string>();
        if (path == "#") return "root";
        std::string key;
        const ojson* defs = nullptr;
        for (const char* prefix : {"#/$defs/", "#/definitions/"}) {
            const std::string p = prefix;
            if (path.rfind(p, 0) == 0) {
                key = path.substr(p.size());
                const std::string section = p.substr(2, p.size() - 3);
                if (root_.contains(section)) defs = &root_[section];
            }
        }
        if (defs == nullptr || !defs->contains(key)) {
            fail("unresolved $ref " + path);
        }
        const auto it = refs_.find(path);
        if (it != refs_.end()) return it->second;
        const std::string name = fresh_name("def-" + key);
        refs_[path] = name;
        rules_[name] = "";  // reserved: a recursive $ref finds the name
        rules_[name] = expr((*defs)[key], name);
        return name;
    }

    std::string typed(const std::string& t, const ojson& s,
                      const std::string& hint) {
        if (t == "string") return string_expr(s, hint);
        if (t == "number" || t == "integer" || t == "boolean" ||
            t == "null") {
            use_prim(t);
            return t;
        }
        if (t == "object") return object_expr(s, hint);
        if (t == "array") return array_expr(s, hint);
        fail("unknown type " + t);
    }

    static std::uint64_t count(const ojson& s, const char* k,
                               std::uint64_t def) {
        if (!s.contains(k)) return def;
        if (!s[k].is_number_unsigned()) {
            fail(std::string(k) + " must be a non-negative integer");
        }
        return s[k].get<std::uint64_t>();
    }

    std::string string_expr(const ojson& s, const std::string& hint) {
        use_prim("space");
        if (s.contains("pattern")) {
            if (!s["pattern"].is_string()) fail("pattern must be a string");
            std::string re = s["pattern"].get<std::string>();
            if (!re.empty() && re.back() == '$') re.pop_back();
            return rule(hint, "\"\\\"\" (" + regex_to_gbnf(re, true) +
                                  ") \"\\\"\" space");
        }
        if (s.contains("format") && s["format"].is_string()) {
            static const std::map<std::string, std::string> formats = {
                {"date",
                 "[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"},
                {"time",
                 "([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\\.[0-9]{1,6})?"
                 "(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])"},
                {"date-time",
                 "[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])T"
                 "([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\\.[0-9]{1,6})?"
                 "(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])"},
                {"uuid",
                 "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                 "[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"},
            };
            const auto it = formats.find(s["format"].get<std::string>());
            if (it != formats.end()) {
                return rule(hint, "\"\\\"\" (" + regex_to_gbnf(it->second) +
                                      ") \"\\\"\" space");
            }
        }
        const bool has_min = s.contains("minLength");
        const bool has_max = s.contains("maxLength");
        if (has_min || has_max) {
            const std::uint64_t lo = count(s, "minLength", 0);
            use_prim("char");
            std::string rep = "{" + std::to_string(lo) + ",";
            if (has_max) {
                const std::uint64_t hi = count(s, "maxLength", 0);
                if (hi < lo) fail("maxLength < minLength");
                rep += std::to_string(hi);
            }
            return rule(hint, "\"\\\"\" char" + rep + "} \"\\\"\" space");
        }
        use_prim("string");
        return "string";
    }

    struct Prop {
        std::string kv;  // the key, the colon and the value
        bool required = false;
    };

    std::string object_expr(const ojson& s, const std::string& hint) {
        use_prim("space");
        std::vector<Prop> props;
        std::set<std::string> required;
        if (s.contains("required")) {
            if (!s["required"].is_array()) fail("required must be an array");
            for (const auto& r : s["required"]) {
                if (!r.is_string()) fail("required must list strings");
                required.insert(r.get<std::string>());
            }
        }
        const bool has_props = s.contains("properties");
        if (has_props) {
            if (!s["properties"].is_object()) {
                fail("properties must be an object");
            }
            for (const auto& [key, sub] : s["properties"].items()) {
                const std::string v = expr(sub, hint + "-" + key);
                props.push_back(
                    {gbnf_literal(ojson(key).dump()) +
                         " space \":\" space " + v,
                     required.count(key) > 0});
            }
        }
        // Extra keys: the schema's additionalProperties, else allowed
        // only when no properties are listed.
        std::string extra;
        const bool has_add = s.contains("additionalProperties");
        const ojson add = has_add ? s["additionalProperties"] : ojson(!has_props);
        if (!(add.is_boolean() && !add.get<bool>())) {
            use_prim("string");
            extra = "string \":\" space " +
                    (add.is_boolean() ? (use_prim("value"), std::string("value"))
                                      : expr(add, hint + "-additional"));
        }
        const std::string base = fresh_name(hint);
        rules_[base] = "";
        memo_.clear();
        const std::string body = members(props, extra, 0, true, base);
        rules_[base] = "\"{\" space " + body + " \"}\" space";
        return base;
    }

    /**
     * The members from index i on; `first` when no member came before
     * (so no comma). Each (i, first) gets one rule, so optional
     * members do not multiply the grammar.
     */
    std::string members(const std::vector<Prop>& props,
                        const std::string& extra, std::size_t i, bool first,
                        const std::string& base) {
        const std::string comma = first ? "" : "\",\" space ";
        if (i == props.size()) {
            if (extra.empty()) return "";
            return "(" + comma + extra + " (\",\" space " + extra + ")*)?";
        }
        const auto key = std::make_pair(i, first);
        const auto it = memo_.find(key);
        if (it != memo_.end()) return it->second;
        const std::string name =
            fresh_name(base + "-m" + std::to_string(i) + (first ? "f" : "n"));
        memo_[key] = name;
        rules_[name] = "";
        const std::string rest = members(props, extra, i + 1, false, base);
        std::string body = comma + props[i].kv + (rest.empty() ? "" : " " + rest);
        if (!props[i].required) {
            const std::string skip = members(props, extra, i + 1, first, base);
            body = "(" + body + ") | " + (skip.empty() ? "\"\"" : skip);
        }
        rules_[name] = body;
        return name;
    }

    std::string array_expr(const ojson& s, const std::string& hint) {
        use_prim("space");
        if (s.contains("prefixItems")) {
            const ojson& p = s["prefixItems"];
            if (!p.is_array()) fail("prefixItems must be an array");
            std::string body = "\"[\" space";
            for (std::size_t i = 0; i < p.size(); ++i) {
                if (i) body += " \",\" space";
                body += " " + expr(p[i], hint + "-" + std::to_string(i));
            }
            return rule(hint, body + " \"]\" space");
        }
        std::string item;
        if (s.contains("items")) {
            item = expr(s["items"], hint + "-item");
        } else {
            use_prim("value");
            item = "value";
        }
        const std::uint64_t lo = count(s, "minItems", 0);
        const bool bounded = s.contains("maxItems");
        const std::uint64_t hi = count(s, "maxItems", 0);
        if (bounded && hi < lo) fail("maxItems < minItems");
        if (bounded && hi == 0) return rule(hint, "\"[\" space \"]\" space");
        const std::uint64_t more_lo = lo > 0 ? lo - 1 : 0;
        std::string more = "(\",\" space " + item + "){" +
                           std::to_string(more_lo) + "," +
                           (bounded ? std::to_string(hi - 1) : "") + "}";
        std::string list = item + " " + more;
        if (lo == 0) list = "(" + list + ")?";
        return rule(hint, "\"[\" space " + list + " \"]\" space");
    }

    const ojson& root_;
    std::map<std::string, std::string> rules_;
    std::map<std::string, std::string> refs_;
    std::map<std::pair<std::size_t, bool>, std::string> memo_;
};

}  // namespace

std::string json_schema_to_gbnf(std::string_view schema_json) {
    ojson s;
    try {
        s = ojson::parse(schema_json);
    } catch (const std::exception& e) {
        throw std::invalid_argument(std::string("json schema: ") + e.what());
    }
    std::string g = Converter(s).run();
    Gbnf::parse(g);  // a converter bug surfaces here, not mid-request
    return g;
}

std::string json_value_gbnf() {
    static const std::string g = [] {
        std::map<std::string, std::string> rules{{"root", "value"}};
        std::vector<std::string> todo{"value"};
        while (!todo.empty()) {
            const std::string n = todo.back();
            todo.pop_back();
            if (rules.count(n)) continue;
            const Prim& p = prims().at(n);
            rules[n] = p.body;
            for (const char* d : p.deps) {
                if (!rules.count(d)) todo.push_back(d);
            }
        }
        std::string out;
        for (const auto& [name, body] : rules) {
            out += name + " ::= " + body + "\n";
        }
        return out;
    }();
    return g;
}

}  // namespace locus::model
