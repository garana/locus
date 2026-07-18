---
name: test
description: Build and run the cpp-llm Catch2 unit tests. Use after
  any code change, before commits, or when the user asks to run or
  fix tests.
---

# Test cpp-llm

    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j --target cppllm_tests
    ./build/tests/cppllm_tests

- Run a subset with a tag filter: `./build/tests/cppllm_tests
  "[kv]"` (tags: [sanity], [kv], [sys], [gguf], [tok], [ops],
  [backend], [engine], [server], [vulkan], [e2e]; keep this list
  current as suites are added).
- [e2e] needs the ~1 MB test model: `scripts/fetch-test-model.sh`
  (gitignored under tests/models/; tests SKIP if absent).
- Every new component gets its own `tests/test_<name>.cpp`,
  registered in `tests/CMakeLists.txt`, using the vendored Catch2
  (`#include "catch_amalgamated.hpp"`).
- All tests must pass before any commit. If a test is flaky, that
  is a bug to fix now, not to retry.
