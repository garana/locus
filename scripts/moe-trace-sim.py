#!/usr/bin/env python3
"""Replays a LOCUS_MOE_TRACE routing trace against expert residency
policies and prints how skewed the routing is.

Each routed expert is one cache entry (its gate, up and down tensors
travel together; every expert of a model has the same size). The
budget is a fraction of all layer-expert slots of the MoE layers. A
hit means the picked expert was resident on the GPU.

Two kinds of policy:
- on-demand (lru, lfu, belady): a miss uploads the expert, as the GPU
  weight pool does today, so uploads equal misses;
- hybrid (wlru-step, wlru-hl*, static-top): a miss runs the expert on
  the CPU; only a promotion uploads.

  moe-trace-sim.py TRACE --k K [--skip-tokens P] [--budgets ...]

--k is the experts each token picks per layer (expert_used_count).
--skip-tokens drops the first P tokens' picks (the prompt: batched
prefill records a chunk layer by layer, not token by token).
"""

import argparse
import collections
import heapq
import math
import sys


def load(path, skip):
    picks = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            layer, expert = line.split()
            picks.append((int(layer), int(expert)))
    return picks[skip:]


def skew(picks):
    """Mean share of a layer's picks covered by its hottest experts."""
    per_layer = collections.defaultdict(collections.Counter)
    for layer, expert in picks:
        per_layer[layer][expert] += 1
    n_exp = max(e for _, e in picks) + 1
    out = {}
    for frac in (0.1, 0.25, 0.5):
        top = max(1, round(frac * n_exp))
        shares = []
        for counts in per_layer.values():
            total = sum(counts.values())
            hot = sum(c for _, c in counts.most_common(top))
            shares.append(hot / total)
        out[frac] = sum(shares) / len(shares)
    return out, len(per_layer), n_exp


def lru(picks, cap):
    """On-demand LRU (today's pool): (hits, uploads)."""
    res = collections.OrderedDict()
    hits = 0
    for p in picks:
        if p in res:
            hits += 1
            res.move_to_end(p)
        else:
            if len(res) >= cap:
                res.popitem(last=False)
            res[p] = True
    return hits, len(picks) - hits


class Weights:
    """Decayed hit counts: a pick adds 1, and every pick ages every
    weight by 2^(-1/half_life); half_life=None never decays (LFU).
    Kept in boosted units (weight / decay^t) so aging is implicit."""

    def __init__(self, half_life):
        self.decay = (1.0 if half_life is None
                      else 2.0 ** (-1.0 / half_life))
        self.score = collections.defaultdict(float)
        self.boost = 1.0

    def pick(self, p):
        if self.decay != 1.0:
            self.boost /= self.decay
            if self.boost > 1e100:  # renormalize to avoid overflow
                for k in self.score:
                    self.score[k] /= self.boost
                self.boost = 1.0
        self.score[p] += self.boost

    def __getitem__(self, p):
        return self.score[p]


def lfu(picks, cap):
    """On-demand LFU: a miss always uploads, evicting the lowest
    all-time count. (hits, uploads)."""
    w = Weights(None)
    res = set()
    hits = 0
    for p in picks:
        w.pick(p)
        if p in res:
            hits += 1
            continue
        if len(res) >= cap:
            res.discard(min(res, key=w.__getitem__))
        res.add(p)
    return hits, len(picks) - hits


def wlru_decay(picks, cap, half_life):
    """Hybrid decayed WLRU: weight = decayed hit count (frequency
    weighted by recency). A non-resident pick runs on the CPU, and is
    promoted (uploaded, evicting the lowest-weight resident) only when
    its weight beats that resident's. (hits, uploads)."""
    w = Weights(half_life)
    res = set()
    hits = uploads = 0
    for p in picks:
        w.pick(p)
        if p in res:
            hits += 1
            continue
        if len(res) < cap:
            res.add(p)
            uploads += 1
            continue
        coldest = min(res, key=w.__getitem__)
        if w[p] > w[coldest]:
            res.discard(coldest)
            res.add(p)
            uploads += 1
    return hits, uploads


def wlru_step(picks, cap):
    """Step WLRU: one ranked list of every expert seen, the first cap
    positions are the GPU-resident hot zone. Each pick swaps the
    expert one step toward the head; when that swap moves it into
    the hot zone it is uploaded and the expert it swapped with drops
    out of the zone and is purged. A new expert joins at the tail
    (inside the zone while the list is shorter than cap). A pick is a
    hit when the expert is in the zone before it moves. (hits,
    uploads); a pick outside the zone runs on the CPU."""
    order = []  # rank -> expert
    pos = {}  # expert -> rank
    hits = uploads = 0
    for p in picks:
        i = pos.get(p)
        if i is None:
            i = len(order)
            order.append(p)
            pos[p] = i
            if i < cap:
                uploads += 1  # fills the pool
            continue
        if i < cap:
            hits += 1
        if i > 0:
            q = order[i - 1]
            order[i - 1], order[i] = p, q
            pos[p], pos[q] = i - 1, i
            if i == cap:  # crossed into the zone: swap residency
                uploads += 1
    return hits, uploads


def static_top(picks, cap):
    """Best fixed resident set: the cap most picked experts overall,
    chosen with hindsight (an upper bound for any placement that
    never changes). (hits, uploads)."""
    counts = collections.Counter(picks)
    top = counts.most_common(cap)
    return sum(c for _, c in top), len(top)


def belady(picks, cap):
    """Optimal on-demand cache: evict the resident expert used
    furthest in the future. (hits, uploads)."""
    nxt = [0] * len(picks)
    last = {}
    for i in range(len(picks) - 1, -1, -1):
        nxt[i] = last.get(picks[i], math.inf)
        last[picks[i]] = i
    res = {}
    heap = []
    hits = 0
    for i, p in enumerate(picks):
        if p in res:
            hits += 1
        elif len(res) >= cap:
            while True:
                neg, q = heapq.heappop(heap)
                if res.get(q) == -neg:
                    del res[q]
                    break
        res[p] = nxt[i]
        heapq.heappush(heap, (-nxt[i], p))
    return hits, len(picks) - hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--skip-tokens", type=int, default=0)
    ap.add_argument("--budgets", default="0.1,0.25,0.5,0.75")
    a = ap.parse_args()
    all_picks = load(a.trace, 0)
    n_moe = len({l for l, _ in all_picks})
    per_token = n_moe * a.k
    picks = all_picks[a.skip_tokens * per_token:]
    if not picks:
        sys.exit("no picks")
    sk, n_layers, n_exp = skew(picks)
    slots = n_layers * n_exp
    tokens = len(picks) / per_token
    print(f"{len(picks)} picks ({tokens:.0f} tokens at {per_token} picks"
          f"/token), {n_layers} moe layers x {n_exp} experts = {slots}"
          " slots")
    print("hottest experts' share of a layer's picks (mean over layers):"
          + "".join(f" top {int(f * 100)}% {s * 100:.1f}%;"
                    for f, s in sk.items()))
    policies = [
        ("lru", lambda c: lru(picks, c)),
        ("lfu", lambda c: lfu(picks, c)),
        ("belady", lambda c: belady(picks, c)),
        ("wlru-step", lambda c: wlru_step(picks, c)),
        ("wlru-hl1t", lambda c: wlru_decay(picks, c, per_token)),
        ("wlru-hl10t", lambda c: wlru_decay(picks, c, 10 * per_token)),
        ("wlru-hl100t", lambda c: wlru_decay(picks, c, 100 * per_token)),
        ("static-top", lambda c: static_top(picks, c)),
    ]
    budgets = [float(b) for b in a.budgets.split(",")]
    print("GPU hit rate / uploads per token, by budget (share of slots"
          " resident); hl<N>t = half-life of N tokens:")
    print(f"{'policy':<12}"
          + "".join(f"{str(int(b * 100)) + '%':>16}" for b in budgets))
    for name, fn in policies:
        row = []
        for b in budgets:
            cap = max(1, int(b * slots))
            hits, uploads = fn(cap)
            row.append(f"{100.0 * hits / len(picks):7.1f}% "
                       f"{uploads / tokens:7.2f}")
        print(f"{name:<12}" + "".join(f"{r:>16}" for r in row))


if __name__ == "__main__":
    main()
