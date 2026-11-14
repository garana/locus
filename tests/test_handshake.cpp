// i#848: SHA-256 and HMAC-SHA-256, and the stage protocol's shared-key
// handshake (i#847) built on them.

#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>

#include "catch_amalgamated.hpp"
#include "locus/pipeline/handshake.hpp"
#include "locus/sys/sha256.hpp"

using locus::pipeline::HandshakeClient;
using locus::pipeline::HandshakeServer;
using locus::pipeline::handshake_accept;
using locus::pipeline::handshake_connect;
using locus::pipeline::read_key_file;
using locus::sys::hmac_sha256;
using locus::sys::sha256;
using locus::sys::to_hex;
namespace fs = std::filesystem;

TEST_CASE("sha256 matches the FIPS 180-4 examples", "[handshake]") {
    CHECK(to_hex(sha256("")) ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(to_hex(sha256("abc")) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(to_hex(sha256("abcdbcdecdefdefgefghfghighijhijk"
                        "ijkljklmklmnlmnomnopnopq")) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(to_hex(sha256(std::string(1000000, 'a'))) ==
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST_CASE("hmac_sha256 matches RFC 4231", "[handshake]") {
    // Test cases 1, 2 and 6 (a key longer than the block).
    CHECK(to_hex(hmac_sha256(std::string(20, '\x0b'), "Hi There")) ==
          "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    CHECK(to_hex(hmac_sha256("Jefe", "what do ya want for nothing?")) ==
          "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    CHECK(to_hex(hmac_sha256(
              std::string(131, '\xaa'),
              "Test Using Larger Than Block-Size Key - Hash Key First")) ==
          "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
}

namespace {

/** Both ends of the handshake over a socketpair, the accepting one on
 *  a thread. @returns {accept's result, connect's result}. */
std::pair<bool, bool> over_socketpair(const std::string& server_key,
                                      const std::string& client_key) {
    int sv[2];
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    auto server = std::async(std::launch::async, [&] {
        const bool ok = handshake_accept(sv[0], server_key, 2000);
        ::close(sv[0]);
        return ok;
    });
    const bool c = handshake_connect(sv[1], client_key, 2000);
    ::close(sv[1]);
    return {server.get(), c};
}

}  // namespace

TEST_CASE("the handshake passes with one key and fails with two",
          "[handshake]") {
    const std::string k = "0123456789abcdef0123456789abcdef";
    CHECK(over_socketpair(k, k) == std::pair{true, true});
    const auto [s, c] = over_socketpair(k, "fedcba9876543210fedcba9876543210");
    CHECK_FALSE(s);  // the server refuses a wrong answer
    CHECK_FALSE(c);  // and the client gets no proof
}

TEST_CASE("the handshake steps refuse what is not the protocol",
          "[handshake]") {
    const std::string k = "0123456789abcdef";
    HandshakeServer srv(k);
    HandshakeClient cli(k);
    CHECK_FALSE(cli.on_hello("GET / HTTP/1.1\r\nHost: x\r\n\r"));
    const auto answer = cli.on_hello(srv.hello());
    REQUIRE(answer);
    std::string bad = *answer;
    bad.back() = static_cast<char>(bad.back() ^ 1);  // one bit of the tag
    CHECK_FALSE(srv.on_answer(bad));
    const auto proof = srv.on_answer(*answer);
    REQUIRE(proof);
    CHECK(cli.on_proof(*proof));
    std::string forged = *proof;
    forged[0] = static_cast<char>(forged[0] ^ 1);
    CHECK_FALSE(cli.on_proof(forged));
    // A replayed answer fails against a new greeting (a new nonce).
    HandshakeServer srv2(k);
    CHECK_FALSE(srv2.on_answer(*answer));
}

TEST_CASE("the handshake gives up on a silent peer", "[handshake]") {
    int sv[2];
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    const auto t0 = std::chrono::steady_clock::now();
    CHECK_FALSE(handshake_accept(sv[0], "0123456789abcdef", 200));
    // sv[1] has the hello, answers it, and no proof ever comes.
    CHECK_FALSE(handshake_connect(sv[1], "0123456789abcdef", 200));
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    CHECK(ms < 2000);
    ::close(sv[0]);
    ::close(sv[1]);
}

TEST_CASE("read_key_file takes only a private file of 16 to 4096 bytes",
          "[handshake]") {
    const fs::path d = fs::temp_directory_path() /
                       ("locus-key-" + std::to_string(::getpid()));
    fs::create_directories(d);
    const auto write = [&](const char* name, const std::string& body,
                           fs::perms mode) {
        const fs::path p = d / name;
        std::ofstream(p, std::ios::binary) << body;
        fs::permissions(p, mode, fs::perm_options::replace);
        return p.string();
    };
    const auto rw = fs::perms::owner_read | fs::perms::owner_write;
    CHECK(read_key_file(write("good", std::string(32, 'k'), rw)) ==
          std::string(32, 'k'));
    CHECK_THROWS_AS(read_key_file(write("short", "abc", rw)),
                    std::runtime_error);
    CHECK_THROWS_AS(
        read_key_file(write("open", std::string(32, 'k'),
                            rw | fs::perms::group_read)),
        std::runtime_error);
    CHECK_THROWS_AS(read_key_file((d / "missing").string()),
                    std::runtime_error);
    fs::create_symlink(d / "good", d / "link");
    CHECK_THROWS_AS(read_key_file((d / "link").string()), std::runtime_error);
    fs::remove_all(d);
}
