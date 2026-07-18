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
| M2        | CPU forward pass, single sequence       | done        |
| M3        | Paged KV + continuous batching          | done        |
| M4        | HTTP server (OpenAI API, SSE)           | done        |
| M5        | Vulkan backend                          | in progress |

## Building

    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build

## Testing

    cmake --build build --target cppllm_tests
    ./build/tests/cppllm_tests

## Math backends

Every SIMD/GPU variant is compiled into the one binary; at startup
the best one supported by the running machine is picked (pbw-style
runtime dispatch). Inspect and override:

    cppllm-run --backends
    cppllm-run --backend scalar model.gguf "prompt" 32
    CPPLLM_BACKEND=neon cppllm-server model.gguf 8080

Current entries: neon (arm64), scalar (reference), vulkan (listed;
selectable once M5 completes). x86 sse4/avx2/avx512 and CUDA slot
into src/cppllm/backend/registry.cpp when their kernels land.

## Dependencies

No external packages at runtime. Vendored (pinned, checksummed,
with license files; see docs/DESIGN.md section 6): Catch2 (tests
only), cpp-httplib, nlohmann/json. Optional, auto-detected at
configure time:

- Vulkan loader (GPU backend, milestone M5)
- SIMD compiler flags per arch (neon/sse4/avx2/avx512); kernel
  variants are selected at runtime via `cppllm::sys::detect()`
