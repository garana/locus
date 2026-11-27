#!/usr/bin/env bash
#
# Removes the `assigned:<name>` labels from closed issues and merged
# PRs, so those labels list only work that is still open.
#
# It finds every label with the prefix, then each closed item that
# carries it, and removes the label through the GitHub REST API (no
# GraphQL), sleeping between removals to stay clear of GitHub's
# secondary rate limits. Closed PRs that were not merged keep their
# labels unless --closed-prs is given.
#
# Usage:
#   scripts/ops/gh-unassign-done.sh [--dry-run] [--sleep SECONDS]
#       [--repo OWNER/NAME] [--prefix PREFIX] [--closed-prs]
#
#   --dry-run     print what would be removed; change nothing
#   --sleep N     seconds between removals (default 2)
#   --repo R      the repository (default: the current one, from gh)
#   --prefix P    the label prefix (default "assigned:")
#   --closed-prs  also clear closed PRs that were not merged
#
# Needs: jq, and gh (logged in) with write access to the repository.

set -euo pipefail

dry_run=0
sleep_s=2
repo=""
prefix="assigned:"
closed_prs=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --sleep) sleep_s="$2"; shift ;;
        --repo) repo="$2"; shift ;;
        --prefix) prefix="$2"; shift ;;
        --closed-prs) closed_prs=1 ;;
        -h | --help) sed -n '2,23p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if [[ -z "$repo" ]]; then
    repo="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
fi

# A closed issue, or a merged PR (or any closed PR with --closed-prs).
if [[ $closed_prs -eq 1 ]]; then
    filter='.[] | .number'
else
    filter='.[] | select(.pull_request == null
        or .pull_request.merged_at != null) | .number'
fi

removed=0
while IFS= read -r label; do
    [[ -z "$label" ]] && continue
    # Labels go into a URL path; escape ":" and anything else.
    enc="$(jq -rn --arg s "$label" '$s | @uri')"
    numbers="$(gh api -X GET "repos/$repo/issues" -f state=closed \
        -f labels="$label" -f per_page=100 --paginate \
        --jq "$filter")"
    count=0
    [[ -n "$numbers" ]] && count="$(wc -l <<<"$numbers" | tr -d ' ')"
    echo "$label: $count closed item(s)"
    for n in $numbers; do
        if [[ $dry_run -eq 1 ]]; then
            echo "  would remove $label from #$n"
            continue
        fi
        gh api -X DELETE "repos/$repo/issues/$n/labels/$enc" >/dev/null
        echo "  removed $label from #$n"
        removed=$((removed + 1))
        sleep "$sleep_s"
    done
done < <(gh api -X GET "repos/$repo/labels" -f per_page=100 --paginate \
    --jq '.[].name' | awk -v p="$prefix" 'index($0, p) == 1')

echo "done: $removed label(s) removed"
