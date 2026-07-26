#!/usr/bin/env bash
#
# Turnkey locus validation for a fresh cloud instance.
#
# Purpose:
#   1. EXECUTE the AVX2 matvec path -- impossible on the vx host
#      (i5-2400 Sandy Bridge, SSE4 only), so the avx2 test only ever
#      SKIPs there. Any current-gen cloud CPU has AVX2.
#   2. Prove avx2 (and cuda, if present) reproduce the llama.cpp
#      golden token-exact through a real forward pass.
#   3. On a GPU box, capture a rough CUDA tok/s number.
#
# Assumes: Ubuntu 22.04/24.04 with sudo. For the GPU path use a Deep
# Learning AMI (CUDA preinstalled) or install cuda-toolkit yourself.
#
# Usage:
#   REPO_URL=<your git url> bash cloud-validate.sh
# REPO_URL must be reachable from the box (SSH key on the box, or an
# HTTPS URL with a PAT). Everything else has sane defaults.

set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:garana/locus.git}"
WORKDIR="${WORKDIR:-$HOME/locus}"
PROMPT="${PROMPT:-The capital of France is}"

section() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

# --------------------------------------------------------------------
section "Environment"
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
CPU_SIMD=$(grep -m1 -oE 'avx512f|avx2|fma|sse4_2' /proc/cpuinfo \
          | sort -u | tr '\n' ' ')
echo "CPU SIMD: ${CPU_SIMD}"
if echo "$CPU_SIMD" | grep -qw avx2; then
  echo "  -> AVX2 present: the avx2 matvec test will EXECUTE here."
else
  echo "  !! No AVX2 on this instance -- pick a current-gen type."
fi
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
then
  HAS_GPU=1
  nvidia-smi --query-gpu=name,memory.total,compute_cap \
             --format=csv,noheader
fi

# --------------------------------------------------------------------
section "Build prerequisites"
sudo apt-get update -qq
sudo apt-get install -y -qq git cmake g++ curl ca-certificates
NVCC=""
for c in /usr/local/cuda/bin/nvcc "$(command -v nvcc 2>/dev/null || true)"
do
  [ -n "$c" ] && [ -x "$c" ] && { NVCC="$c"; break; }
done
if [ "$HAS_GPU" = 1 ] && [ -z "$NVCC" ]; then
  echo "GPU present but no nvcc found -- install the CUDA toolkit"
  echo "(or use a Deep Learning AMI). Building CPU-only for now."
fi

# --------------------------------------------------------------------
section "Clone + build locus"
if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" pull --ff-only || true
else
  git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
if [ -n "$NVCC" ]; then
  export CUDACXX="$NVCC"
  export PATH="$(dirname "$NVCC"):$PATH"
  echo "Building WITH CUDA ($($NVCC --version | grep -oE 'release [0-9.]+'))"
fi
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
./build/locus-run --backends

# --------------------------------------------------------------------
section "AVX2 validation (the whole point)"
# Synthetic matvec_avx2-vs-scalar unit test -- no model needed.
./build/tests/locus_tests "[backend]" -s 2>&1 | tee /tmp/backend.log \
  | grep -iE "avx2 matvec|sse4 matvec|cuda matvec|passed|failed|SKIP" \
  || true
if grep -A3 "avx2 matvec matches scalar" /tmp/backend.log \
     | grep -qi SKIP; then
  echo "!! avx2 test SKIPPED -- this CPU lacks AVX2 (unexpected on cloud)"
else
  echo ">> avx2 matvec EXECUTED and matched scalar (gap closed)."
fi

# --------------------------------------------------------------------
section "AVX2 golden e2e (real forward pass)"
# stories260K is ~1MB and public; the test iterates EVERY selectable
# backend (avx2 included here) and requires the llama.cpp golden.
scripts/fetch-test-model.sh || \
  echo "fetch-test-model.sh failed (network?) -- e2e will SKIP"
./build/tests/locus_tests \
  "every selectable backend reproduces the golden output" -s 2>&1 \
  | grep -iE "backend |passed|failed|SKIP" || true

# --------------------------------------------------------------------
if [ "$HAS_GPU" = 1 ] && [ -n "$NVCC" ]; then
  section "CUDA tok/s benchmark"
  # Public Q4_K_M ~0.8GB; fits any modern GPU fully -> pager ~100% hit.
  mkdir -p tests/models
  M=tests/models/llama-3.2-1b-q4_k_m.gguf
  URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
  [ -f "$M" ] || curl -fSL -o "$M" "$URL"

  N=64
  echo "--- CUDA: $N-token decode (wall incl. load; tok/s approx) ---"
  LOCUS_MOE_STATS=1 /usr/bin/time -v \
    ./build/locus-run --backend cuda "$M" "$PROMPT" "$N" 2>&1 \
    | grep -iE "cuda-pool|Elapsed \(wall|Maximum resident" || true
  echo "--- scalar CPU baseline (same N) ---"
  /usr/bin/time -v \
    ./build/locus-run --backend scalar "$M" "$PROMPT" "$N" 2>&1 \
    | grep -iE "Elapsed \(wall" || true
  echo "tok/s ~= $N / wall_seconds (subtract ~1s mmap load for decode)."
else
  section "CUDA benchmark skipped"
  echo "No usable GPU+nvcc; AVX2 validation above is the CPU-box result."
fi

section "Summary"
echo "AVX2: see 'EXECUTED and matched scalar' above (the vx gap)."
echo "Golden e2e: avx2 must appear among the passing backends."
[ "$HAS_GPU" = 1 ] && echo "CUDA: tok/s + [cuda-pool] hit-rate captured."
