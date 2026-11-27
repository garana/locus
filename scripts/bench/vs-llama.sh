#!/usr/bin/env bash
#
# Prefill and decode speed of locus against llama.cpp on this host, for
# docs/HOSTS.md. Refuses to run while pbw_miner or pbw_gpu_worker is
# running (not stopped), so the numbers carry no background load.
#
# For each model: `locus-run --warmup --timing` on the first 2200 and
# 8600 bytes of docs/DESIGN.md (about 512 and 2048 tokens; the table
# gives the real count) then 128 decoded tokens, ROUNDS times (the
# median), and `llama-bench -p 512,2048 -n 128 -r ROUNDS` (its mean).
# Prints one markdown table: tok/s for both and locus's share. locus
# decodes after its 512-token prompt; llama-bench's tg128 starts from an
# empty context, so decode is close, not equal. A run that fails gives
# "-" in its cells.
#
# Usage:
#   scripts/bench/vs-llama.sh [--backend NAME] [--rounds N]
#       [--llama-args "ARGS"] MODEL.gguf...
#
#   --backend NAME     locus's --backend (default: its own choice)
#   --rounds N         rounds a run (default 3)
#   --llama-args ARGS  extra llama-bench arguments, e.g. "-ngl 0" for
#                      its CPU path (on Apple that path still uses
#                      Accelerate for prefill)
#   LOCUS_RUN, LLAMA_BENCH  the binaries (default build/locus-run and
#                      llama-bench on PATH)
#
# Needs: bash, awk, grep, sed, and both binaries.

set -uo pipefail
export LC_ALL=C  # a dot decimal for printf, sort and awk

backend=""
rounds=3
llama_args=""
models=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) backend="$2"; shift ;;
        --rounds) rounds="$2"; shift ;;
        --llama-args) llama_args="$2"; shift ;;
        -h | --help) sed -n '2,28p' "$0"; exit 0 ;;
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

# The miner must be stopped: no process of either name, or each in T.
miner_running() {
    local p s
    for p in $(pgrep -x pbw_miner 2>/dev/null) \
             $(pgrep -x pbw_gpu_worker 2>/dev/null); do
        s="$(ps -o state= -p "$p" 2>/dev/null | tr -d ' ')"
        [[ -n "$s" && "$s" != T* ]] && return 0
    done
    return 1
}
if miner_running; then
    echo "pbw_miner is running: stop it (or hold its gate) first" >&2
    exit 3
fi

# The median of the numbers on stdin, or "-" for none.
median() {
    sort -n | awk '{v[NR] = $1} END {print NR ? v[int((NR + 1) / 2)] : "-"}'
}

# One locus run on the first $2 bytes: "prompt_tokens prefill decode".
locus_run() {
    local args=(--warmup --timing)
    [[ -n "$backend" ]] && args+=(--backend "$backend")
    local rate='[0-9.]+ ms \(([0-9.]+) tok\/s\)'
    local re=".*prefill ([0-9]+) tok $rate.*decode [0-9]+ tok $rate.*"
    "$locus" "${args[@]}" "$1" "$(head -c "$2" "$design")" 128 2>&1 |
        sed -nE "s/$re/\1 \2 \3/p" | head -1
}

# The cell text for a rate ("-" stays).
cell() { [[ "$1" == "-" ]] && echo "-" || printf '%.1f' "$1"; }

echo "| model | phase | locus tok/s | llama.cpp tok/s | locus share |"
echo "|-------|-------|-------------|-----------------|-------------|"
for m in "${models[@]}"; do
    name="$(basename "$m" .gguf)"
    # shellcheck disable=SC2086
    jl="$("$llama" -m "$m" -p 512,2048 -n 128 -r "$rounds" -o jsonl \
        $llama_args 2>/dev/null)"
    # llama.cpp's mean tok/s for (n_prompt, n_gen), or "-".
    ll_rate() {
        echo "$jl" | awk -v p="$1" -v g="$2" '
            function num(k,   s) {
                if (!match($0, "\"" k "\": ?[0-9.]+")) return "";
                s = substr($0, RSTART, RLENGTH); sub(/.*: ?/, "", s);
                return s
            }
            num("n_prompt") == p && num("n_gen") == g { r = num("avg_ts") }
            END { print (r == "" ? "-" : r) }'
    }
    for bytes in 2200 8600; do
        runs="$(for ((r = 0; r < rounds; ++r)); do
                    locus_run "$m" "$bytes"
                done)"
        toks="$(echo "$runs" | awk 'NF == 3 {print $1; exit}')"
        pre="$(echo "$runs" | awk 'NF == 3 {print $2}' | median)"
        size=512
        [[ "$bytes" == 8600 ]] && size=2048
        rows=("prefill $size|$pre|$(ll_rate "$size" 0)")
        if [[ "$bytes" == 2200 ]]; then
            dec="$(echo "$runs" | awk 'NF == 3 {print $3}' | median)"
            rows+=("decode 128|$dec|$(ll_rate 0 128)")
        fi
        for row in "${rows[@]}"; do
            IFS='|' read -r phase lr cr <<<"$row"
            share="$(awk -v a="$lr" -v b="$cr" 'BEGIN {
                if (a == "-" || b == "-" || b + 0 == 0) print "-";
                else printf "%.2fx", a / b }')"
            label="$phase"
            [[ "$phase" == prefill* && -n "$toks" ]] &&
                label="$phase ($toks tok)"
            printf '| %s | %s | %s | %s | %s |\n' "$name" "$label" \
                "$(cell "$lr")" "$(cell "$cr")" "$share"
        done
    done
done
