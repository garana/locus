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
  LaTeX (`$...$`, `$$...$$` or a ` ```math ` block).
- Unit tests with vendored Catch2 for every component; all tests
  green before any commit.
- Commit as each unit of work completes. No Co-Authored-By lines.
- Dependency policy (DESIGN.md 6): no FetchContent, no network at
  build time; vendor pinned single-file deps with recorded sha256.
- SIMD kernel variants follow the ../pbw pattern: one source file
  per variant with per-source compile flags, runtime dispatch via
  locus::sys::detect().
