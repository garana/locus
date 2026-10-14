#!/bin/sh
# Clone-or-update locus into a target folder and build it (Release).
# Idempotent: updates the folder if it is already a locus checkout,
# clones otherwise, then configures and builds. Binaries land in
# <dir>/build (locus-server, locus-run, locus-stage, ...).
#
# Run it straight from the repo, without checking out first:
#   gh api repos/garana/locus/contents/scripts/locus-setup.sh \
#     --jq '.content' | base64 --decode | bash -s - [dir] [ref]
#
# Args (both optional):
#   dir   target folder (default: ./locus)
#   ref   git branch/tag/sha to check out (default: main)
#
# Prerequisites: gh (authenticated), git, cmake, a C++20 compiler.
# Optional GPU backends are auto-detected at configure time; a
# missing one (CUDA, Vulkan) just disables that backend and the
# build still succeeds.
set -eu

repo="garana/locus"
dir="${1:-locus}"
ref="${2:-main}"

if [ -d "$dir/.git" ]; then
  echo ">> updating existing checkout in $dir"
  git -C "$dir" fetch origin
  git -C "$dir" checkout "$ref"
  # Fast-forward when on a branch; a detached tag/sha just stays put.
  git -C "$dir" pull --ff-only origin "$ref" 2>/dev/null || true
else
  echo ">> cloning $repo into $dir"
  gh repo clone "$repo" "$dir"
  git -C "$dir" checkout "$ref"
fi

echo ">> configuring (Release)"
cmake -B "$dir/build" -S "$dir" -DCMAKE_BUILD_TYPE=Release

echo ">> building (-j)"
cmake --build "$dir/build" -j

echo ">> done. binaries in $dir/build:"
ls "$dir/build" | grep locus || true
cat <<EOF

next:
  # OpenAI + Anthropic compatible server on port 8080
  $dir/build/locus-server <model.gguf> 8080
EOF
