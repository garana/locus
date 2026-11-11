#!/bin/sh
# Downloads the tiny (~1 MB) stories260K test model used by the
# [e2e] integration tests. The file is gitignored; tests SKIP
# cleanly when it is absent, so running this is optional.
#
# With an argument it downloads an embedding model instead, for the
# [bert] tests, pinned to a commit and checked by sha256:
#   bge     bge-small-en-v1.5 Q8_0 (36 MB)
#   e5      e5-small-v2 Q8_0 (37 MB)
#   bge-m3  bge-m3 Q8_0 (635 MB)
#   nomic   nomic-embed-text v1.5 Q8_0 (146 MB)
# A file lands under a temporary name first, so a test never sees a
# partial file at the final path.
set -eu

dir="$(cd "$(dirname "$0")/.." && pwd)/tests/models"
mkdir -p "$dir"

# fetch REPO COMMIT FILE SHA256 (REPO under ggml-org unless it has a /)
fetch() {
    out="$dir/$3"
    case "$1" in */*) repo="$1" ;; *) repo="ggml-org/$1" ;; esac
    curl -fSL -o "$out.part" \
        "https://huggingface.co/$repo/resolve/$2/$3"
    got="$(shasum -a 256 "$out.part" | cut -d' ' -f1)"
    if [ "$got" != "$4" ]; then
        echo "sha256 mismatch: $got" >&2
        rm -f "$out.part"
        exit 1
    fi
    mv "$out.part" "$out"
    echo "$4  $out"
}

case "${1:-}" in
bge)
    fetch bge-small-en-v1.5-Q8_0-GGUF \
        f2068edd9b54f2a369549ccc71f70ed273a2a801 \
        bge-small-en-v1.5-q8_0.gguf \
        f046db1dc724cf4f6f0a0c5917e922823b73eb1d27b8f9a9c2797f7866974804
    ;;
e5)
    fetch e5-small-v2-Q8_0-GGUF \
        14aa7ca88ca627e2ea0051929636561158e289ab \
        e5-small-v2-q8_0.gguf \
        afdfb5c342d2efc2a051c426dd1d00913495d5f2bbceaea100d2f3892aa31cbc
    ;;
bge-m3)
    fetch bge-m3-Q8_0-GGUF \
        9eba04c5d75ba5a1595e45de734d36bef4e5cb98 \
        bge-m3-q8_0.gguf \
        aa473d51f451a22f0fcf39ba3330c14bed38a385712b1113440f69df4047a173
    ;;
nomic)
    fetch nomic-ai/nomic-embed-text-v1.5-GGUF \
        0188c9bf409793f810680a5a431e7b899c46104c \
        nomic-embed-text-v1.5.Q8_0.gguf \
        3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7
    ;;
"")
    url="https://huggingface.co/ggml-org/models/resolve/main"
    url="$url/tinyllamas/stories260K.gguf"
    curl -fSL -o "$dir/stories260K.gguf" "$url"
    shasum -a 256 "$dir/stories260K.gguf"
    ;;
*)
    echo "usage: $0 [bge|e5|bge-m3|nomic]" >&2
    exit 2
    ;;
esac
