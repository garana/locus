#include "locus/auth/token_cache.hpp"

#include <chrono>

namespace locus::auth {

Nanos WallClock::now() const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

namespace {
/** Payload bytes charged for an entry: key + value sizes. */
std::size_t entry_bytes(const std::string& key,
                        const std::string& value) {
    return key.size() + value.size();
}
}  // namespace

TokenCache::TokenCache(const Clock& clock, std::size_t capacity,
                       std::size_t max_bytes, std::size_t max_purge)
    : clock_(clock),
      capacity_(capacity),
      max_bytes_(max_bytes),
      max_purge_(max_purge) {}

TokenCache::~TokenCache() = default;

void TokenCache::unlink(Node* n) {
    if (n->prev != nullptr) {
        n->prev->next = n->next;
    } else {
        head_ = n->next;
    }
    if (n->next != nullptr) {
        n->next->prev = n->prev;
    } else {
        tail_ = n->prev;
    }
    n->prev = nullptr;
    n->next = nullptr;
}

void TokenCache::push_tail(Node* n) {
    n->prev = tail_;
    n->next = nullptr;
    if (tail_ != nullptr) {
        tail_->next = n;
    } else {
        head_ = n;
    }
    tail_ = n;
}

void TokenCache::erase(Node* n) {
    bytes_ -= entry_bytes(n->key, n->value);
    unlink(n);
    index_.erase(n->key);  // frees the Node (owned by the index)
}

void TokenCache::purge_expired_tail() {
    const Nanos now = clock_.now();
    std::size_t purged = 0;
    while (tail_ != nullptr && now >= tail_->expiry) {
        erase(tail_);  // stops at the first non-expired tail entry...
        if (max_purge_ != 0 && ++purged >= max_purge_) {
            break;  // ...and at the per-insert cap, bounding work.
        }
    }
}

void TokenCache::insert(const std::string& key,
                        const std::string& value, Nanos expiry_ns) {
    if (capacity_ == 0 || expiry_ns <= clock_.now()) {
        return;  // disabled, or the entry is already expired
    }
    // Reclaim dead entries at the cold end before evicting a live one.
    purge_expired_tail();
    const std::size_t need = entry_bytes(key, value);

    // An entry larger than the whole byte budget can never fit; drop
    // any stale copy and give up rather than evicting everything.
    if (max_bytes_ != 0 && need > max_bytes_) {
        if (const auto it = index_.find(key); it != index_.end()) {
            erase(it->second.get());
        }
        return;
    }
    // Replacing an existing key: remove the old entry, then re-admit
    // at the tail (re-probates a refreshed credential).
    if (const auto it = index_.find(key); it != index_.end()) {
        erase(it->second.get());
    }
    // Evict from the tail until the new entry fits both budgets.
    while (tail_ != nullptr && index_.size() >= capacity_) {
        erase(tail_);
    }
    while (tail_ != nullptr && max_bytes_ != 0 &&
           bytes_ + need > max_bytes_) {
        erase(tail_);
    }

    auto node = std::make_unique<Node>();
    node->key = key;
    node->value = value;
    node->expiry = expiry_ns;
    Node* raw = node.get();
    index_.emplace(key, std::move(node));
    bytes_ += need;
    push_tail(raw);
}

const std::string* TokenCache::lookup(const std::string& key) {
    const auto it = index_.find(key);
    if (it == index_.end()) {
        return nullptr;
    }
    Node* n = it->second.get();
    if (clock_.now() >= n->expiry) {
        erase(n);  // expired
        return nullptr;
    }
    // Promote one step toward the head: swap with the predecessor.
    if (n->prev != nullptr) {
        Node* p = n->prev;
        unlink(n);
        n->next = p;
        n->prev = p->prev;
        if (p->prev != nullptr) {
            p->prev->next = n;
        } else {
            head_ = n;
        }
        p->prev = n;
    }
    return &n->value;
}

}  // namespace locus::auth
