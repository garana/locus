#include "locus/engine/helper_prefix_store.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "locus/auth/helper_spawn.hpp"

namespace locus::engine {

namespace {

using auth::HelperMessage;

/** A POSIX shm object of this process, unlinked on scope exit. */
class Shm {
  public:
    Shm() = default;
    Shm(const Shm&) = delete;
    Shm& operator=(const Shm&) = delete;
    ~Shm() {
        if (p_ != MAP_FAILED) {
            ::munmap(p_, n_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
        if (!name_.empty()) {
            ::shm_unlink(name_.c_str());
        }
    }

    /** Creates `name` with `n` bytes reserved (posix_fallocate, so a
     *  full /dev/shm fails here, not as a SIGBUS on a write).
     *  @returns false on any error. */
    bool create(const std::string& name, std::size_t n) {
        fd_ = ::shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC,
                         0600);
        if (fd_ < 0) {
            return false;
        }
        name_ = name;
        if (::posix_fallocate(fd_, 0, static_cast<off_t>(n)) != 0) {
            return false;
        }
        p_ = ::mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        n_ = n;
        return p_ != MAP_FAILED;
    }

    std::uint8_t* data() const { return static_cast<std::uint8_t*>(p_); }
    std::size_t size() const { return n_; }

  private:
    std::string name_;
    int fd_ = -1;
    void* p_ = MAP_FAILED;
    std::size_t n_ = 0;
};

std::string hex(std::uint64_t v) {
    char s[17];
    std::snprintf(s, sizeof s, "%016llx", static_cast<unsigned long long>(v));
    return s;
}

/** @returns true when `s` is 16 lower-case hex digits, into `v`. */
bool parse_hex(std::string_view s, std::uint64_t& v) {
    if (s.size() != 16) {
        return false;
    }
    v = 0;
    for (const char c : s) {
        int d;
        if (c >= '0' && c <= '9') {
            d = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            d = c - 'a' + 10;
        } else {
            return false;
        }
        v = (v << 4) | static_cast<std::uint64_t>(d);
    }
    return true;
}

/** @returns the names of a list reply's space-separated `names`. */
std::unordered_set<std::uint64_t> names_of(
    const std::optional<HelperMessage>& r) {
    std::unordered_set<std::uint64_t> names;
    if (const std::string* list = r ? r->get("names") : nullptr) {
        std::string_view s = *list;
        while (!s.empty()) {
            const std::size_t sp = s.find(' ');
            std::uint64_t v;
            if (parse_hex(s.substr(0, sp), v)) {
                names.insert(v);
            }
            s = sp == std::string_view::npos ? "" : s.substr(sp + 1);
        }
    }
    return names;
}

bool says_ok(const std::optional<HelperMessage>& r) {
    const std::string* ok = r ? r->get("ok") : nullptr;
    return ok != nullptr && *ok == "1";
}

}  // namespace

HelperPrefixStore::HelperPrefixStore(std::vector<std::string> argv,
                                     DiskKey key, std::size_t block_bytes,
                                     std::uint64_t cap_bytes,
                                     std::chrono::milliseconds timeout,
                                     std::chrono::milliseconds refresh)
    : DiskPrefixStore(std::move(key), block_bytes, cap_bytes, nullptr),
      argv_(std::move(argv)),
      timeout_(timeout) {
    if (argv_.empty()) {
        throw std::invalid_argument("HelperPrefixStore: empty argv");
    }
    {
        std::lock_guard s(start_mu_);
        start();
    }
    if (refresh.count() > 0) {
        refresher_ = std::thread([this, refresh] {
            std::unique_lock lk(stop_mu_);
            while (!stop_cv_.wait_for(lk, refresh, [this] { return stop_; })) {
                lk.unlock();
                HelperMessage req;
                req.type = "list";
                const std::optional<HelperMessage> r = call(std::move(req));
                if (r) {
                    // Added, not replaced: a name the helper dropped is
                    // forgotten when its read misses.
                    std::unordered_set<std::uint64_t> got = names_of(r);
                    std::lock_guard lock(mu_);
                    names_.merge(got);
                }
                lk.lock();
            }
        });
    }
}

HelperPrefixStore::~HelperPrefixStore() {
    {
        std::lock_guard lk(stop_mu_);
        stop_ = true;
    }
    stop_cv_.notify_all();
    if (refresher_.joinable()) {
        refresher_.join();
    }
}

bool HelperPrefixStore::start() {
    if (starts_ >= 2) {
        if (!off_.exchange(true)) {
            std::fprintf(stderr,
                         "locus: the prefix cache helper died twice; "
                         "the helper tier is off\n");
        }
        std::lock_guard lock(mu_);
        conn_.reset();
        names_.clear();
        return false;
    }
    ++starts_;
    std::string err;
    std::shared_ptr<auth::HelperConnection> c =
        auth::spawn_helper(argv_, &err);
    if (!c) {
        std::fprintf(stderr, "locus: prefix cache helper: %s\n", err.c_str());
        return false;
    }
    HelperMessage req;
    req.type = "list";
    const std::optional<HelperMessage> r = c->request(req, timeout_);
    std::unordered_set<std::uint64_t> names = names_of(r);
    std::lock_guard lock(mu_);
    conn_ = std::move(c);
    names_ = std::move(names);
    return r.has_value();
}

std::optional<HelperMessage> HelperPrefixStore::call(HelperMessage req) {
    for (int attempt = 0; attempt < 2 && !off_.load(); ++attempt) {
        std::shared_ptr<auth::HelperConnection> c;
        {
            std::lock_guard lock(mu_);
            c = conn_;
        }
        if (c && !c->broken()) {
            std::optional<HelperMessage> r = c->request(req, timeout_);
            if (r || !c->broken()) {
                return r;  // an answer, or a slow helper: a miss
            }
        }
        std::lock_guard s(start_mu_);
        {
            std::lock_guard lock(mu_);
            if (conn_ != c) {
                continue;  // another call started it again meanwhile
            }
        }
        if (!start()) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::uint32_t HelperPrefixStore::find(
    std::span<const tok::TokenId> prompt) const {
    if (off_.load()) {
        return 0;
    }
    const std::vector<std::uint64_t> d = digests(prompt);
    std::lock_guard lock(mu_);
    for (std::size_t j = d.size() - 1; j >= 1; --j) {
        if (names_.contains(d[j])) {
            return static_cast<std::uint32_t>(j);
        }
    }
    return 0;
}

std::size_t HelperPrefixStore::known() const {
    std::lock_guard lock(mu_);
    return names_.size();
}

bool HelperPrefixStore::read(std::span<const tok::TokenId> prompt,
                             std::uint32_t n_blocks,
                             std::span<std::uint8_t* const> dst) {
    if (off_.load() || n_blocks == 0 || dst.size() < n_blocks ||
        prompt.size() < std::size_t{n_blocks} * key().block_tokens) {
        return false;
    }
    const std::uint64_t digest = digest_of(prompt, n_blocks);
    {
        std::lock_guard lock(mu_);
        if (!names_.contains(digest)) {
            return false;
        }
    }
    const std::uint64_t bytes =
        file_bytes(header(prompt, n_blocks).size(), n_blocks);
    Shm shm;
    const std::string name = "/locus-pfx-" + std::to_string(::getpid()) +
                             "-" + std::to_string(++shm_seq_);
    if (!shm.create(name, bytes)) {
        return false;
    }
    HelperMessage req;
    req.type = "read";
    req.set("name", hex(digest));
    req.set("shm", name);
    req.set("bytes", std::to_string(bytes));
    const std::optional<HelperMessage> r = call(std::move(req));
    if (!r) {
        return false;  // no answer: the name may still be good
    }
    if (says_ok(r) &&
        decode({shm.data(), shm.size()}, prompt, n_blocks, dst)) {
        return true;
    }
    {
        std::lock_guard lock(mu_);
        names_.erase(digest);
    }
    if (says_ok(r)) {
        // The helper has a blob that does not check: have it dropped.
        HelperMessage ev;
        ev.type = "evict";
        ev.set("name", hex(digest));
        std::shared_ptr<auth::HelperConnection> c;
        {
            std::lock_guard lock(mu_);
            c = conn_;
        }
        if (c) {
            c->notify(std::move(ev));
        }
    }
    return false;
}

bool HelperPrefixStore::write(std::span<const tok::TokenId> prompt,
                              std::uint32_t n_blocks,
                              std::span<const std::uint8_t* const> src) {
    if (off_.load() || n_blocks == 0 || src.size() < n_blocks ||
        prompt.size() < std::size_t{n_blocks} * key().block_tokens) {
        return false;
    }
    const std::uint64_t digest = digest_of(prompt, n_blocks);
    {
        std::lock_guard lock(mu_);
        if (names_.contains(digest)) {
            return true;
        }
    }
    const std::vector<std::uint8_t> m = meta(prompt, n_blocks, src);
    const std::size_t bb = block_bytes();
    const std::uint64_t bytes =
        m.size() + static_cast<std::uint64_t>(bb) * n_blocks;
    if (bytes > cap_bytes()) {
        return false;
    }
    Shm shm;
    const std::string name = "/locus-pfx-" + std::to_string(::getpid()) +
                             "-" + std::to_string(++shm_seq_);
    if (!shm.create(name, bytes)) {
        return false;
    }
    std::memcpy(shm.data(), m.data(), m.size());
    for (std::uint32_t i = 0; i < n_blocks; ++i) {
        std::memcpy(shm.data() + m.size() + bb * i, src[i], bb);
    }
    HelperMessage req;
    req.type = "write";
    req.set("name", hex(digest));
    req.set("shm", name);
    req.set("bytes", std::to_string(bytes));
    if (!says_ok(call(std::move(req)))) {
        return false;
    }
    std::lock_guard lock(mu_);
    names_.insert(digest);
    return true;
}

}  // namespace locus::engine
