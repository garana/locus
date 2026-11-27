#!/usr/bin/env bash
#
# Prefill and decode speed of locus against llama.cpp on this host, for
# docs/HOSTS.md. Refuses to run while pbw_miner is running (not
# stopped), so the numbers carry no background load.
#
# For each model: `locus-run --warmup --timing` on the first N bytes of
# docs/DESIGN.md (about 512 and 2048 tokens) then 128 decoded tokens,
# ROUNDS times (the median is kept), and `llama-bench -p 512,2048
# -n 128 -r ROUNDS`. Prints one markdown table: tok/s for both and
# locus's share. locus decodes after its 512-token prompt; llama-bench's
# tg128 starts from an empty context, so decode is close, not equal.
#
# Usage:
#   scripts/bench/vs-llama.sh [--backend NAME] [--rounds N]
#       [--llama-args "ARGS"] MODEL.gguf...
#
#   --backend NAME     locus's --backend (default: its own choice)
#   --rounds N         rounds a run (default 3)
#   --llama-args ARGS  extra llama-bench arguments, e.g. "-ngl 0" for
#                      its CPU path
#   LOCUS_RUN, LLAMA_BENCH  the binaries (default build/locus-run and
#                      llama-bench on PATH)
#
# Needs: bash, awk, and both binaries.

set -euo pipefail

backend=""
rounds=3
llama_args=""
models=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) backend="$2"; shift ;;
        --rounds) rounds="$2"; shift ;;
        --llama-args) llama_args="$2"; shift ;;
        -h | --help) sed -n '2,25p' "$0"; exit 0 ;;
        *) models+=("$1") ;;
    esac
    shift
done
if [[ ${#models[@]} -eq 0 ]]; then
    echo "usage: $0 [--backend NAME] [--rounds N] MODEL.gguf..." >&2
    exit 2
fi

here="$(cd "$(dirname "$0")/../.." && pwd)"
locus="${LOCUS_RUN:-$here/build/locus-run}"
llama="${LLAMA_BENCH:-llama-bench}"
design="$here/docs/DESIGN.md"

# pbw_miner must be stopped (absent, or every process in state T).
miner_running() {
    local p s
    for p in $(pgrep -x pbw_miner 2>/dev/null || true); do
        s="$(ps -o state= -p "$p" 2>/dev/null | tr -d ' ')"
        [[ -n "$s" && "$s" != T* ]] && return 0
    done
    return 1
}
if miner_running; then
    echo "pbw_miner is running: stop it (or hold its gate) first" >&2
    exit 3
fi

median() { sort -n | awk '{v[NR] = $1} END {print v[int((NR + 1) / 2)]}'; }

# tok/s of one locus phase ("prefill" or "decode") from --timing.
locus_rate() {
    local model="$1" bytes="$2" phase="$3" out
    local args=(--warmup --timing)
    [[ -n "$backend" ]] && args+=(--backend "$backend")
    out="$("$locus" "${args[@]}" "$model" "$(head -c "$bytes" "$design")" \
        128 2>&1 | grep -o 'timing:.*')"
    echo "$out" |
        sed -E "s/.*$phase [0-9]+ tok [0-9.]+ ms \(([0-9.]+) tok\/s\).*/\1/"
}

echo "| model | phase | locus tok/s | llama.cpp tok/s | locus share |"
echo "|-------|-------|-------------|-----------------|-------------|"
for m in "${models[@]}"; do
    name="$(basename "$m" .gguf)"
    # shellcheck disable=SC2086
    csv="$("$llama" -m "$m" -p 512,2048 -n 128 -r "$rounds" -o csv \
        $llama_args 2>/dev/null)"
    ll_rate() {  # avg_ts of the row with n_prompt $1 and n_gen $2
        echo "$csv" | awk -F, -v p="$1" -v g="$2" '
            NR == 1 { for (i = 1; i <= NF; ++i) { gsub(/"/, "", $i);
                      c[$i] = i } next }
            { gsub(/"/, "") }
            $c["n_prompt"] == p && $c["n_gen"] == g { print $c["avg_ts"] }'
    }
    for spec in "prefill 512 2200 512 0" "prefill 2048 8600 2048 0" \
                "decode 128 2200 0 128"; do
        read -r phase size bytes lp lg <<<"$spec"
        lr="$(for ((r = 0; r < rounds; ++r)); do
                  locus_rate "$m" "$bytes" "$phase"
              done | median)"
        cr="$(ll_rate "$lp" "$lg")"
        share="$(awk -v a="$lr" -v b="$cr" \
            'BEGIN { if (b > 0) printf "%.2fx", a / b; else print "-" }')"
        printf '| %s | %s %s | %.1f | %.1f | %s |\n' "$name" "$phase" \
            "$size" "$lr" "${cr:-0}" "$share"
    done
done
