#pragma once

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "locus/config/config_file.hpp"

namespace locus_tools {

/**
 * Parses `s` as a non-negative value in [0, INT_MAX]. Returns true and
 * sets `out` on success; false on empty input, trailing non-digit junk,
 * a negative value, or a value out of range (ERANGE from strtol, or
 * above INT_MAX before the narrowing cast).
 *
 * This is the ONE guarded integer parse behind every CLI integer flag
 * (Spec kInt, --device, --layers). The guards exist because strtol
 * returns a long: on LP64 a value above INT_MAX passes a `< 0` test and
 * then truncates on the cast to int/uint32_t (2^31 -> INT_MIN, 2^32 ->
 * 0), which silently yields a DIFFERENT value than the user typed. The
 * three callers had drifted apart (one hardened, two not), so they now
 * share this -- a fix here is a fix in all of them.
 *
 * @param s The candidate string (a whole token; no surrounding space).
 * @param out Set to the parsed value only when the function returns true.
 * @returns true iff `s` is a valid non-negative integer in [0, INT_MAX].
 */
inline bool parse_nonneg_int(const std::string& s, int& out) {
    if (s.empty()) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (*end != '\0' || errno == ERANGE || v < 0 ||
        v > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

/**
 * Whether a directive takes a parameter, and of what type. One enum
 * both the CLI parser and the config-file loader read, so a flag and
 * its config key can never disagree about arity or type.
 */
enum class ArgType {
    kBool,    /**< No parameter; presence sets the bound bool true. */
    kString,  /**< Requires a string parameter. */
    kInt,     /**< Requires a non-negative integer parameter. */
    kList,    /**< Requires a parameter; repeatable, appended. */
};

template <class Opt>
class Spec;

/**
 * One CLI flag / config-file directive, bound to a field of `Opt`. The
 * CLI flag is "--" + name and the config-file key is name verbatim, so
 * the two are the same directive by construction (i#19).
 *
 * Construct only via the per-type factories (string_/integer/list_/
 * boolean): they pair the ArgType with the one matching member pointer,
 * so a type/binding mismatch (e.g. kInt bound to a string field, which
 * would dereference a null member pointer in apply()) cannot be built.
 * The bindings are private; Spec reads them.
 */
template <class Opt>
class Directive {
  public:
    static Directive string_(std::string name, std::string Opt::*m,
                             std::string help, bool required = false) {
        Directive d(std::move(name), ArgType::kString, std::move(help),
                    required);
        d.str_ = m;
        return d;
    }
    static Directive integer(std::string name, int Opt::*m,
                             std::string help, bool required = false) {
        Directive d(std::move(name), ArgType::kInt, std::move(help),
                    required);
        d.num_ = m;
        return d;
    }
    static Directive string_list(std::string name,
                                 std::vector<std::string> Opt::*m,
                                 std::string help,
                                 bool required = false) {
        Directive d(std::move(name), ArgType::kList, std::move(help),
                    required);
        d.list_ = m;
        return d;
    }
    static Directive boolean(std::string name, bool Opt::*m,
                             std::string help, bool required = false) {
        Directive d(std::move(name), ArgType::kBool, std::move(help),
                    required);
        d.flag_ = m;
        return d;
    }

    const std::string& name() const { return name_; }
    bool required() const { return required_; }
    /** @returns true if this directive takes a parameter (not kBool). */
    bool takes_value() const { return type_ != ArgType::kBool; }

  private:
    friend class Spec<Opt>;
    Directive(std::string name, ArgType type, std::string help,
              bool required)
        : name_(std::move(name)),
          type_(type),
          required_(required),
          help_(std::move(help)) {}

    std::string name_;
    ArgType type_;
    bool required_ = false;
    std::string help_;
    bool Opt::*flag_ = nullptr;
    std::string Opt::*str_ = nullptr;
    int Opt::*num_ = nullptr;
    std::vector<std::string> Opt::*list_ = nullptr;
};

/**
 * A table of directives bound to an options struct `Opt`. Drives CLI
 * parsing, config-file overlay (CLI-then-file, the file wins), the
 * required-field check, and --help from one source of truth.
 */
template <class Opt>
class Spec {
  public:
    explicit Spec(std::vector<Directive<Opt>> d) : d_(std::move(d)) {}

    const std::vector<Directive<Opt>>& directives() const { return d_; }

    /** @returns the directive named `name`, or nullptr. */
    const Directive<Opt>* find(const std::string& name) const {
        for (const auto& d : d_) {
            if (d.name_ == name) {
                return &d;
            }
        }
        return nullptr;
    }

    /**
     * Applies directive `d` to `opt` from `value` (value ignored for a
     * kBool, which is set true). kList appends. @throws
     * std::runtime_error if a kInt value is not a non-negative integer.
     */
    void apply(const Directive<Opt>& d, Opt& opt,
               const std::string& value) const {
        switch (d.type_) {
            case ArgType::kBool:
                opt.*(d.flag_) = true;
                break;
            case ArgType::kString:
                opt.*(d.str_) = value;
                break;
            case ArgType::kInt:
                opt.*(d.num_) = parse_int(value, d.name_);
                break;
            case ArgType::kList:
                (opt.*(d.list_)).push_back(value);
                break;
        }
    }

    /**
     * Overlays a parsed config file onto `opt` (the file wins over
     * whatever the CLI left there). A kList key's values replace the
     * bound list only when the file gives it at least once. Each key
     * touched is inserted into `seen` (for the required check).
     * @throws std::runtime_error on an unknown key or a bad kInt value.
     */
    void apply_config(const locus::config::ConfigFile& cfg, Opt& opt,
                      std::set<std::string>& seen) const {
        for (const auto& key : cfg.keys()) {
            const Directive<Opt>* d = find(key);
            if (d == nullptr) {
                throw std::runtime_error("unknown config key: " + key);
            }
            if (d->type_ == ArgType::kList) {
                opt.*(d->list_) = cfg.values(key);  // file replaces list
            } else {
                apply(*d, opt, cfg.get(key).value_or(""));
            }
            seen.insert(key);
        }
    }

    /** @returns the name of the first required directive absent from
     * `seen`, or "" if all required directives were given. */
    std::string first_missing_required(
        const std::set<std::string>& seen) const {
        for (const auto& d : d_) {
            if (d.required_ && seen.find(d.name_) == seen.end()) {
                return d.name_;
            }
        }
        return "";
    }

    /** @returns a rendered flag list for --help, one directive a line. */
    std::string help() const {
        std::string out;
        for (const auto& d : d_) {
            out += "  --" + d.name_;
            if (d.type_ == ArgType::kInt) {
                out += " N";
            } else if (d.type_ != ArgType::kBool) {
                out += " VALUE";
            }
            out += "\n      " + d.help_;
            if (d.required_) {
                out += " (required)";
            }
            out += "\n";
        }
        return out;
    }

  private:
    static int parse_int(const std::string& v, const std::string& name) {
        int out = 0;
        if (!parse_nonneg_int(v, out)) {
            throw std::runtime_error(
                "'" + name +
                "' must be a non-negative integer, got '" + v + "'");
        }
        return out;
    }

    std::vector<Directive<Opt>> d_;
};

}  // namespace locus_tools
