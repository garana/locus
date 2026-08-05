// Reference auth helper for locus (see docs/CLIENTS.md, DESIGN R15).
//
// Speaks the framed text protocol on stdin/stdout: for each `auth`
// request it answers ok=1 + identity + not_after when the credential
// is in its allow-list, else ok=0. Request-lifecycle event pushes
// (type "request") are read and ignored. The allow-list is the union
// of argv tokens and the comma-separated LOCUS_AUTH_TOKENS env var.
//
// REFERENCE ONLY: operators replace this with real validation (an
// API-key DB, JWT check, OIDC introspection, ...). It is deliberately
// tiny and dependency-free beyond the shared codec.
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>

#include "locus/auth/helper_message.hpp"

namespace {

using locus::auth::decode_text;
using locus::auth::encode_text;
using locus::auth::HelperDecode;
using locus::auth::HelperMessage;

std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void write_all(const std::string& s) {
    std::size_t off = 0;
    while (off < s.size()) {
        const ssize_t n =
            ::write(STDOUT_FILENO, s.data() + off, s.size() - off);
        if (n <= 0) {
            return;
        }
        off += static_cast<std::size_t>(n);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::set<std::string> allowed;
    for (int i = 1; i < argc; ++i) {
        allowed.insert(argv[i]);
    }
    if (const char* env = std::getenv("LOCUS_AUTH_TOKENS")) {
        std::string s(env), tok;
        for (char c : s) {
            if (c == ',') {
                if (!tok.empty()) {
                    allowed.insert(tok);
                }
                tok.clear();
            } else {
                tok.push_back(c);
            }
        }
        if (!tok.empty()) {
            allowed.insert(tok);
        }
    }

    std::string buf;
    char chunk[4096];
    for (;;) {
        const ssize_t n = ::read(STDIN_FILENO, chunk, sizeof chunk);
        if (n <= 0) {
            break;  // EOF: locus closed the pipe -> exit
        }
        buf.append(chunk, static_cast<std::size_t>(n));
        HelperMessage m;
        std::string err;
        HelperDecode d;
        while ((d = decode_text(buf, m, err)) ==
               HelperDecode::kComplete) {
            if (m.type == "auth") {
                const std::string* cred = m.get("credential");
                HelperMessage r;
                r.req_id = m.req_id;
                r.type = "auth";
                if (cred != nullptr && allowed.count(*cred) != 0) {
                    r.set("ok", "1");
                    r.set("identity", "user:" + *cred);
                    r.set("not_after",
                          std::to_string(now_ns() + 60'000'000'000));
                } else {
                    r.set("ok", "0");
                }
                std::string wire;
                if (encode_text(r, wire)) {
                    write_all(wire);
                }
            }
            // type "request" (lifecycle event) and anything else are
            // acknowledged by being consumed; no reply.
            m = HelperMessage{};
        }
        if (d == HelperDecode::kError) {
            break;
        }
    }
    return 0;
}
