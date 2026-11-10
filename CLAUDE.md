# locus

Pure C/C++ LLM inference server: CPU, Vulkan and CUDA backends,
paged KV cache, continuous batching, OpenAI and Anthropic APIs, with a
minimal vendored dependency surface. Design and rationale:
docs/DESIGN.md. Current state: README.md.

## Resuming a session

1. Read the Status section of README.md (what works, and what is
   pending).
2. Work is tracked in GitHub issues; the pending list in README.md
   links the open ones.
3. Keep README.md and this file in step with the code: update them in
   the same commit as the work. They describe the current state, not
   the history of changes.

## Build and test

Use the `build` and `test` skills (.claude/skills/). Quick form:

    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j && ./build/tests/locus_tests

## Conventions

- C++20, -Wall -Wextra -Wpedantic, warning-clean.
- JSDoc-style comments (/** ... @param ... @returns */) on public
  APIs; only when adding or updating a comment.
- Text files wrap at 80 columns; markdown tables plain ascii with
  content-aligned widths.
- Documentation: terse, plain text; the current state, not a work
  log; no `--` joining ideas in a sentence; math as GitHub-rendered
  LaTeX: display math in a ` ```math ` block (not indented in a
  list), inline `$...$` on one line with no backslash before
  punctuation (GitHub strips it: use `\ `, not `\,`) and `\lt`, not
  `<`.
- Unit tests with vendored Catch2 for every component; all tests
  green before any commit.
- Commit as each unit of work completes. No Co-Authored-By lines.
- Dependency policy (DESIGN.md 6): no FetchContent, no network at
  build time; vendor pinned single-file deps with recorded sha256.
- SIMD kernel variants: one source file per variant
  (`backend/matvec_sse4.cpp`, `matvec_avx2.cpp`, `matvec_neon.cpp`,
  `matvec_neon_dotprod.cpp`), and only that file gets the variant's
  instruction flags (`set_source_files_properties` in CMakeLists.txt,
  or a target pragma opened after the headers, as the dotprod file
  does), so the rest of the binary stays baseline and runs on any CPU
  of the arch. CMake builds every variant the compiler accepts
  (`check_cxx_compiler_flag` / `check_cxx_source_compiles`); none is
  chosen at build time. At startup `locus::sys::detect()` reads the
  CPU's features, `backend/registry.cpp` marks each variant available
  or not from them and uses the first available one (`--backend` or
  `LOCUS_BACKEND` overrides). Each variant is tested against the
  scalar kernels.
