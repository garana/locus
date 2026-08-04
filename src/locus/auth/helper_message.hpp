#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace locus::auth {

/**
 * A wire-agnostic message exchanged over the locus <-> helper socket
 * (ported from codicis ipc/helper_message). A correlation id, a type
 * name, and an ordered set of string key/value fields. The same
 * struct carries both auth resolution (request/response) and locus's
 * request-lifecycle event pushes, so the helper logic is independent
 * of the message kind.
 */
struct HelperMessage {
    /** Correlation id (echoed in a response; 0 for fire-and-forget
     * event notifications that expect no reply). */
    std::uint64_t req_id = 0;
    /** Message type, e.g. "auth" or "request". */
    std::string type;
    std::vector<std::pair<std::string, std::string>> fields;

    /** @returns The first value for `key`, or nullptr if absent. */
    const std::string* get(std::string_view key) const {
        for (const auto& kv : fields) {
            if (kv.first == key) {
                return &kv.second;
            }
        }
        return nullptr;
    }

    /** Appends a field. */
    void set(std::string key, std::string value) {
        fields.emplace_back(std::move(key), std::move(value));
    }
};

/** Outcome of a codec decode attempt. */
enum class HelperDecode {
    kIncomplete,  /**< Need more bytes; nothing consumed. */
    kComplete,    /**< A message was decoded and consumed. */
    kError,       /**< Malformed input; the connection should close. */
};

/**
 * Encodes `msg` as a newline-delimited key=value record terminated by
 * a blank line, appended to `out`:
 *
 *     req_id=<n>\ntype=<t>\n<key>=<value>\n...\n\n
 *
 * Human-readable and pipelining-friendly (records concatenate).
 * Values may not contain a newline; encode_text returns false and
 * appends nothing if any key/value would break the framing.
 */
bool encode_text(const HelperMessage& msg, std::string& out);

/**
 * Decodes one message from the front of `buf`. On kComplete the
 * consumed bytes (through the blank-line terminator) are erased from
 * `buf` and `out` is filled; on kIncomplete `buf` is untouched; on
 * kError `err` describes the fault and the caller should close.
 */
HelperDecode decode_text(std::string& buf, HelperMessage& out,
                         std::string& err);

}  // namespace locus::auth
