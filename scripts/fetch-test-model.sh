#!/bin/sh
# Downloads the tiny (~1 MB) stories260K test model used by the
# [e2e] integration tests. The file is gitignored; tests SKIP
# cleanly when it is absent, so running this is optional.
#
# With the argument "bge" it downloads bge-small-en-v1.5 Q8_0 (36 MB)
# instead, for the embedding tests ([bert]), pinned to a commit and
# checked by sha256. It lands under a temporary name first, so a test
# never sees a partial file at the final path.
set -eu

dir="$(cd "$(dirname "$0")/.." && pwd)/tests/models"
mkdir -p "$dir"

if [ "${1:-}" = "bge" ]; then
    url="https://huggingface.co/ggml-org/bge-small-en-v1.5-Q8_0-GGUF"
    url="$url/resolve/f2068edd9b54f2a369549ccc71f70ed273a2a801"
    url="$url/bge-small-en-v1.5-q8_0.gguf"
    sum="f046db1dc724cf4f6f0a0c5917e922823b73eb1d27b8f9a9c2797f7866974804"
    out="$dir/bge-small-en-v1.5-q8_0.gguf"
    curl -fSL -o "$out.part" "$url"
    got="$(shasum -a 256 "$out.part" | cut -d' ' -f1)"
    if [ "$got" != "$sum" ]; then
        echo "sha256 mismatch: $got" >&2
        rm -f "$out.part"
        exit 1
    fi
    mv "$out.part" "$out"
    echo "$sum  $out"
    exit 0
fi

url="https://huggingface.co/ggml-org/models/resolve/main"
url="$url/tinyllamas/stories260K.gguf"
curl -fSL -o "$dir/stories260K.gguf" "$url"
shasum -a 256 "$dir/stories260K.gguf"
