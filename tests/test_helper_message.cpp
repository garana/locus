#include <string>

#include "catch_amalgamated.hpp"
#include "locus/auth/helper_message.hpp"

using locus::auth::decode_text;
using locus::auth::encode_text;
using locus::auth::HelperDecode;
using locus::auth::HelperMessage;

namespace {
HelperMessage make(std::uint64_t id, const std::string& type) {
    HelperMessage m;
    m.req_id = id;
    m.type = type;
    return m;
}
}  // namespace

TEST_CASE("helper message text round-trips", "[auth]") {
    HelperMessage m = make(42, "auth");
    m.set("credential", "sk-abc123");
    m.set("kind", "chat");
    std::string wire;
    REQUIRE(encode_text(m, wire));

    HelperMessage out;
    std::string err;
    REQUIRE(decode_text(wire, out, err) == HelperDecode::kComplete);
    REQUIRE(out.req_id == 42);
    REQUIRE(out.type == "auth");
    REQUIRE(*out.get("credential") == "sk-abc123");
    REQUIRE(*out.get("kind") == "chat");
    REQUIRE(out.get("missing") == nullptr);
    REQUIRE(wire.empty());  // fully consumed
}

TEST_CASE("helper codec pipelines and correlates out of order",
          "[auth]") {
    // Three responses queued back-to-back in one buffer, and -- as
    // the helper may answer out of order -- their req_ids need not
    // ascend. The client correlates by req_id, so decoding must
    // surface each id independently and in arrival order.
    std::string wire;
    REQUIRE(encode_text(make(7, "auth"), wire));
    REQUIRE(encode_text(make(3, "auth"), wire));
    REQUIRE(encode_text(make(9, "auth"), wire));

    std::string err;
    HelperMessage out;
    REQUIRE(decode_text(wire, out, err) == HelperDecode::kComplete);
    REQUIRE(out.req_id == 7);
    REQUIRE(decode_text(wire, out, err) == HelperDecode::kComplete);
    REQUIRE(out.req_id == 3);
    REQUIRE(decode_text(wire, out, err) == HelperDecode::kComplete);
    REQUIRE(out.req_id == 9);
    REQUIRE(decode_text(wire, out, err) == HelperDecode::kIncomplete);
}

TEST_CASE("helper codec decodes incrementally", "[auth]") {
    HelperMessage m = make(1, "request");
    m.set("event", "create");
    std::string full;
    REQUIRE(encode_text(m, full));

    // Feed all but the final terminator byte: incomplete, untouched.
    std::string buf = full.substr(0, full.size() - 1);
    HelperMessage out;
    std::string err;
    REQUIRE(decode_text(buf, out, err) == HelperDecode::kIncomplete);
    REQUIRE(buf.size() == full.size() - 1);
    // Deliver the rest.
    buf += full.substr(full.size() - 1);
    REQUIRE(decode_text(buf, out, err) == HelperDecode::kComplete);
    REQUIRE(*out.get("event") == "create");
}

TEST_CASE("helper codec rejects malformed input", "[auth]") {
    std::string err;
    HelperMessage out;
    // Missing req_id.
    std::string a = "type=auth\n\n";
    REQUIRE(decode_text(a, out, err) == HelperDecode::kError);
    // Line without '='.
    std::string b = "req_id=1\ngarbage\n\n";
    REQUIRE(decode_text(b, out, err) == HelperDecode::kError);
    // Non-numeric req_id.
    std::string c = "req_id=NaN\ntype=auth\n\n";
    REQUIRE(decode_text(c, out, err) == HelperDecode::kError);
}

TEST_CASE("helper encode rejects newline in a value", "[auth]") {
    HelperMessage m = make(1, "auth");
    m.set("credential", "line1\nline2");  // would break framing
    std::string wire;
    REQUIRE_FALSE(encode_text(m, wire));
    REQUIRE(wire.empty());
}
