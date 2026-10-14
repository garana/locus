#!/bin/sh
# Send a unified diff (read from stdin) to a running locus server and
# print the model's code review. locus exposes an OpenAI-compatible
# chat endpoint, so this just packages the diff into one chat request.
#
# usage:
#   gh pr diff 123       | locus-review.sh [--locus-endpoint host:port]
#   git show <sha>       | locus-review.sh --locus-endpoint vx.local:8084
#   git diff main...HEAD | locus-review.sh
#
# The endpoint is host:port (no scheme; plain HTTP is used), resolved
# in this order: --locus-endpoint, then $LOCUS_ENDPOINT, then the
# default 127.0.0.1:8080. Requires curl and jq, and a locus-server
# already running (see scripts/locus-setup.sh).
set -eu

die() { echo "locus-review: $*" >&2; exit 2; }
need() {
  command -v "$1" >/dev/null 2>&1 ||
    { echo "locus-review: $1 not found on PATH" >&2; exit 1; }
}
usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; }

endpoint="${LOCUS_ENDPOINT:-127.0.0.1:8080}"

while [ $# -gt 0 ]; do
  case "$1" in
    --locus-endpoint)
      [ $# -ge 2 ] || die "--locus-endpoint needs a host:port value"
      endpoint="$2"; shift 2 ;;
    --locus-endpoint=*)
      endpoint="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

need curl
need jq

diff="$(cat)"
[ -n "$diff" ] || die "no diff on stdin (pipe a diff in)"

sys="You are a senior C++ reviewer. Report correctness bugs first, \
then risks, then nits. Cite file:line. Be terse."

jq -n --arg s "$sys" --arg d "$diff" '{
  model: "local",
  stream: false,
  messages: [
    {role: "system", content: $s},
    {role: "user", content: ("Review this diff:\n\n" + $d)}
  ]
}' | curl -sS "http://$endpoint/v1/chat/completions" \
       -H "content-type: application/json" -d @- \
   | jq -r '.choices[0].message.content'
