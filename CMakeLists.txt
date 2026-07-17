cmake_minimum_required(VERSION 3.24)
project(cppllm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(cppllm
    src/cppllm/gguf/gguf.cpp
    src/cppllm/kv/block_allocator.cpp
    src/cppllm/sys/features.cpp
    src/cppllm/sys/mmap_file.cpp
    src/cppllm/tok/tokenizer.cpp
)
target_include_directories(cppllm PUBLIC src)

# ----------------------------------------------------------------
# Vectorized CPU support (same scheme as ../pbw): probe which SIMD
# flags the compiler accepts; kernel variants (M2) are added as
# per-source files with these flags, and cppllm::sys::detect()
# picks the best variant at runtime.
# ----------------------------------------------------------------
include(CheckCXXCompilerFlag)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    # NEON is baseline on arm64; nothing to probe.
    set(CPPLLM_SIMD_VARIANTS "neon")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CPPLLM_SIMD_VARIANTS "sse4")
    check_cxx_compiler_flag("-mavx2" CPPLLM_CC_AVX2)
    if(CPPLLM_CC_AVX2)
        list(APPEND CPPLLM_SIMD_VARIANTS "avx2")
    endif()
    check_cxx_compiler_flag("-mavx512f" CPPLLM_CC_AVX512)
    if(CPPLLM_CC_AVX512)
        list(APPEND CPPLLM_SIMD_VARIANTS "avx512")
    endif()
endif()
message(STATUS "cppllm SIMD variants: ${CPPLLM_SIMD_VARIANTS}")

# ----------------------------------------------------------------
# Vulkan (optional): detection + loader link now; compute kernels
# land in M5. Shader-to-SPIR-V embedding will follow ../pbw's
# glslc/glslangValidator custom-command pattern.
# ----------------------------------------------------------------
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    target_compile_definitions(cppllm PRIVATE CPPLLM_HAS_VULKAN=1)
    target_link_libraries(cppllm PRIVATE Vulkan::Vulkan)
    message(STATUS "cppllm Vulkan: found (${Vulkan_LIBRARY})")
else()
    message(STATUS "cppllm Vulkan: not found, GPU backend disabled")
endif()

enable_testing()
add_subdirectory(tests)
