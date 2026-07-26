---
name: build
description: Configure and build cpp-llm (CMake, Release by default).
  Use whenever the user asks to build, compile, or after source or
  CMake changes need verification.
---

# Build cpp-llm

The same two commands build on every host (M2 / vx / Pi):

    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j

`-j` with no number is fine; pass `-jN` (e.g. `-j$(nproc)`, or `-j4`
on the 4-core vx/Pi) to bound parallelism. Both lines are allowlisted
in `.claude/settings.json`, so builds do not re-prompt -- run them
plain (do NOT pipe through `grep`, which adds an un-allowlisted
pipeline stage and re-prompts).

- Configure output reports detected SIMD variants and whether the
  Vulkan loader was found; mention both if the user asked about
  capabilities.
- A Debug build goes in a separate tree: `-B build-debug
  -DCMAKE_BUILD_TYPE=Debug`. Release-only in practice on all hosts.
- The build must stay warning-clean under -Wall -Wextra -Wpedantic;
  treat new warnings as failures to fix, not to ignore. Note GCC
  (vx/Pi) warns where clang (M2) does not -- e.g. designated-init
  missing-field-initializers; keep it clean for both.
- `compile_commands.json` is symlinked at the repo root for clangd;
  it regenerates on configure.

## Per-host toolchain (optional GPU backends)

Backends are auto-detected at configure; a missing one just disables
that backend, the build still succeeds. Confirmed recipes:

- M2 (arm64): NEON + Vulkan (MoltenVK). No env needed. No CUDA.
- Pi 5 (arm64, Debian 12): NEON + Vulkan (Mesa V3DV). Stock apt
  packages, no SDK paths. No CUDA.
- vx (x86-64, Ubuntu 24.04): SSE4/AVX2 + CUDA. The ONLY host knob is
  making nvcc discoverable -- export `CUDACXX=/usr/local/cuda/bin/nvcc`
  (or put `/usr/local/cuda/bin` on PATH) so `find_package(CUDAToolkit)`
  resolves. `CUDA_ARCHITECTURES` is set inside CMakeLists, not on the
  command line. Vulkan dev headers are not installed, so
  `find_package(Vulkan)` reports not-found at configure ("locus
  Vulkan: not found, GPU backend disabled") and that backend is
  skipped -- expected, not an error.
- Shaders: CMake uses `glslc` when present (M2/vx) and falls back to
  `glslangValidator` (Pi) automatically -- either satisfies the
  shader custom-command; no action needed.
