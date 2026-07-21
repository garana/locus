---
name: test
description: Build and run the locus Catch2 unit tests. Use after
  any code change, before commits, or when the user asks to run or
  fix tests.
---

# Test locus

    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j --target locus_tests
    ./build/tests/locus_tests

- Run a subset with a tag filter: `./build/tests/locus_tests
  "[kv]"` (tags: [sanity], [kv], [sys], [gguf], [tok], [ops],
  [backend], [engine], [server], [vulkan], [e2e]; keep this list
  current as suites are added).
- [e2e] needs the ~1 MB test model: `scripts/fetch-test-model.sh`
  (gitignored under tests/models/; tests SKIP if absent). Larger
  optional models (llama-3.2-1b Q8_0/Q4_K_M, deepseek-v2-lite)
  enable more goldens; [deepseek] alone takes ~2 min of CPU.
- Every new component gets its own `tests/test_<name>.cpp`,
  registered in `tests/CMakeLists.txt`, using the vendored Catch2
  (`#include "catch_amalgamated.hpp"`).
- All tests must pass before any commit. If a test is flaky, that
  is a bug to fix now, not to retry.
