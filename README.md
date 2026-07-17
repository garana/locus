# cpp-llm

A pure C/C++ LLM inference server aiming at the unfilled niche
between llama.cpp and the Python serving stacks (LightLLM, vLLM):

- vendor-neutral (CPU first, Vulkan for GPU later),
- token-block paged KV cache with copy-on-write prefix sharing,
- continuous (iteration-level) batching,
- OpenAI-compatible HTTP API,
- minimal, auditable, vendored dependency surface.

See [docs/DESIGN.md](docs/DESIGN.md) for the full MVP design and
milestone plan.

## Status

| Milestone | Description                             | State       |
|-----------|-----------------------------------------|-------------|
| M0        | Scaffolding, CMake, Catch2, allocator   | done        |
| M1        | GGUF loader + tokenizer                 | done        |
| M2        | CPU forward pass, single sequence       | not started |
| M3        | Paged KV + continuous batching          | not started |
| M4        | HTTP server (OpenAI API, SSE)           | not started |
| M5        | Vulkan backend                          | not started |

## Building

    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build

## Testing

    cmake --build build --target cppllm_tests
    ./build/tests/cppllm_tests

## Dependencies

None required at runtime. Catch2 (tests only) is vendored under
`third_party/catch2/` -- see docs/DESIGN.md section 6 for the
dependency policy. Optional, auto-detected at configure time:

- Vulkan loader (GPU backend, milestone M5)
- SIMD compiler flags per arch (neon/sse4/avx2/avx512); kernel
  variants are selected at runtime via `cppllm::sys::detect()`
