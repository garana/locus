#!/bin/sh
# Downloads the tiny (~1 MB) stories260K test model used by the
# [e2e] integration tests. The file is gitignored; tests SKIP
# cleanly when it is absent, so running this is optional.
set -eu

dir="$(cd "$(dirname "$0")/.." && pwd)/tests/models"
url="https://huggingface.co/ggml-org/models/resolve/main"
url="$url/tinyllamas/stories260K.gguf"

mkdir -p "$dir"
curl -fSL -o "$dir/stories260K.gguf" "$url"
shasum -a 256 "$dir/stories260K.gguf"
