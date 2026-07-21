# locus

Pure C/C++ LLM inference server: vendor-neutral GPU (Vulkan),
paged KV cache, continuous batching, OpenAI-compatible API, with a
minimal vendored dependency surface. Full design and milestone
plan: docs/DESIGN.md. Current milestone status: README.md.

## Resuming a session

1. Read the Status table in README.md and docs/DESIGN.md section 5
   to find the current milestone.
2. Continue the first milestone not marked done; DESIGN.md defines
   each milestone's exit test.
3. Keep README.md (status table) and this file up to date as work
   progresses -- update them in the same commit as the work.

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
- Unit tests with vendored Catch2 for every component; all tests
  green before any commit.
- Commit as each unit of work completes. No Co-Authored-By lines.
- Dependency policy (DESIGN.md 6): no FetchContent, no network at
  build time; vendor pinned single-file deps with recorded sha256.
- SIMD kernel variants follow the ../pbw pattern: one source file
  per variant with per-source compile flags, runtime dispatch via
  locus::sys::detect().
