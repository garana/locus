#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace locus::config {

/**
 * A parsed key=value config file (DESIGN.md R15+). Every CLI flag has a
 * config-file equivalent: one key per flag, named like the flag without
 * its leading dashes (e.g. --connect-timeout -> connect-timeout). The
 * tool decides which keys it recognizes; this parser is generic.
 *
 * Format:
 *   - One `key = value` per line (`key=value` and surrounding
 *     whitespace around key and value are both fine).
 *   - A line whose first non-space character is `#` is a comment;
 *     blank lines are ignored. There are no trailing comments, so a
 *     value may contain `#` (e.g. a URL fragment).
 *   - A key may repeat: values are kept in file order, which models a
 *     repeatable flag (e.g. several `downstream` lines = a pool).
 *
 * The file is UTF-8 bytes; values are taken verbatim after the first
 * `=`, trimmed of surrounding whitespace.
 */
class ConfigFile {
  public:
    /**
     * Thrown on an unreadable file or a malformed line. Derives from
     * std::runtime_error so a single `catch (const std::exception&)`
     * (as the reload path and future callers use) catches it; `message`
     * is kept for call sites that read it directly.
     */
    struct Error : std::runtime_error {
        explicit Error(std::string msg)
            : std::runtime_error(msg), message(std::move(msg)) {}
        std::string message;
    };

    /**
     * Parses `path`. @throws Error if the file cannot be opened or a
     * non-blank, non-comment line has no `=`.
     */
    static ConfigFile parse(const std::string& path);

    /** Parses config text directly (for tests / embedding). @throws
     * Error on a malformed line. `source` names the input in errors. */
    static ConfigFile parse_string(const std::string& text,
                                   const std::string& source = "<string>");

    /** @returns true if `key` appears at least once. */
    bool has(const std::string& key) const;

    /**
     * @returns the value of `key`, or nullopt if absent. If `key`
     * repeats, the LAST occurrence wins (so a later line overrides an
     * earlier one, matching the file-wins precedence model).
     */
    std::optional<std::string> get(const std::string& key) const;

    /** @returns every value given for `key`, in file order (empty if
     * absent). Use for repeatable keys like downstream / allow. */
    std::vector<std::string> values(const std::string& key) const;

    /** @returns the distinct keys present, in first-seen order (so a
     * tool can reject unknown keys). */
    const std::vector<std::string>& keys() const { return keys_; }

  private:
    struct Entry {
        std::string key;
        std::string value;
    };
    std::vector<Entry> entries_;  // in file order, repeats kept
    std::vector<std::string> keys_;  // distinct, first-seen order
};

}  // namespace locus::config
