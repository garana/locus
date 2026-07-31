cmake_minimum_required(VERSION 3.24)
project(locus VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warning flags are C++-only: nvcc does not understand -Wpedantic on
# CUDA sources (host-side warnings go through -Xcompiler instead).
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")

add_library(locus
    src/locus/backend/cpu_ops.cpp
    src/locus/backend/matvec_avx2.cpp
    src/locus/backend/matvec_neon.cpp
    src/locus/backend/matvec_sse4.cpp
    src/locus/backend/matvec_vulkan.cpp
    src/locus/backend/cuda_stub.cpp
    src/locus/backend/registry.cpp
    src/locus/backend/vulkan/context.cpp
    src/locus/chat/template.cpp
    src/locus/engine/engine.cpp
    src/locus/gguf/gguf.cpp
    src/locus/kv/block_allocator.cpp
    src/locus/kv/paged_cache.cpp
    src/locus/sys/features.cpp
    src/locus/model/arch.cpp
    src/locus/model/llama.cpp
    src/locus/model/moe_stats.cpp
    src/locus/model/sampling.cpp
    src/locus/server/engine_loop.cpp
    src/locus/server/server.cpp
    src/locus/server/tools.cpp
    src/locus/sys/mmap_file.cpp
    src/locus/sys/thread_pool.cpp
    src/locus/tok/bpe_tokenizer.cpp
    src/locus/tok/tokenizer.cpp
)
target_include_directories(locus SYSTEM PRIVATE
    third_party/ggml
    third_party/httplib
    third_party/json
)
find_package(Threads REQUIRED)
target_link_libraries(locus PUBLIC Threads::Threads)

add_executable(locus-run tools/run_main.cpp)
target_link_libraries(locus-run PRIVATE locus)

add_executable(locus-server tools/server_main.cpp)
target_link_libraries(locus-server PRIVATE locus)

add_executable(locus-gguf tools/gguf_info_main.cpp)
target_link_libraries(locus-gguf PRIVATE locus)
target_include_directories(locus PUBLIC src)

# ----------------------------------------------------------------
# Vectorized CPU support (same scheme as ../pbw): probe which SIMD
# flags the compiler accepts; kernel variants (M2) are added as
# per-source files with these flags, and locus::sys::detect()
# picks the best variant at runtime.
# ----------------------------------------------------------------
include(CheckCXXCompilerFlag)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    # NEON is baseline on arm64; nothing to probe.
    set(LOCUS_SIMD_VARIANTS "neon")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(LOCUS_SIMD_VARIANTS "sse4")
    set_source_files_properties(
        src/locus/backend/matvec_sse4.cpp
        PROPERTIES COMPILE_FLAGS "-msse4.1")
    check_cxx_compiler_flag("-mavx2" LOCUS_CC_AVX2)
    if(LOCUS_CC_AVX2)
        list(APPEND LOCUS_SIMD_VARIANTS "avx2")
        set_source_files_properties(
            src/locus/backend/matvec_avx2.cpp
            PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
    endif()
    check_cxx_compiler_flag("-mavx512f" LOCUS_CC_AVX512)
    if(LOCUS_CC_AVX512)
        list(APPEND LOCUS_SIMD_VARIANTS "avx512")
    endif()
endif()
message(STATUS "locus SIMD variants: ${LOCUS_SIMD_VARIANTS}")

# ----------------------------------------------------------------
# Vulkan (optional): detection + loader link now; compute kernels
# land in M5. Shader-to-SPIR-V embedding will follow ../pbw's
# glslc/glslangValidator custom-command pattern.
# ----------------------------------------------------------------
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    target_compile_definitions(locus PRIVATE LOCUS_HAS_VULKAN=1)
    target_link_libraries(locus PRIVATE Vulkan::Vulkan)
    message(STATUS "locus Vulkan: found (${Vulkan_LIBRARY})")

    find_program(GLSLC glslc)
    find_program(GLSLANG_VALIDATOR glslangValidator)
    if(GLSLANG_VALIDATOR OR GLSLC)
        set(LOCUS_SHADERS
            matvec matvec_f16 matvec_q8_0 matvec_q4_0 matvec_q5_0
            matvec_t matvec_q4_k matvec_q5_k matvec_q6_k matvec_q2_k
            matvec_iq2_xxs matvec_iq3_xxs matvec_iq4_xs
            rmsnorm rope silu_mul attn_paged attn_mla)
        set(SHADER_HDRS "")
        foreach(shader ${LOCUS_SHADERS})
            set(SRC ${CMAKE_SOURCE_DIR}/src/locus/backend/vulkan/${shader}.comp)
            set(HDR ${CMAKE_BINARY_DIR}/shaders/${shader}_spv.h)
            if(GLSLANG_VALIDATOR)
                add_custom_command(
                    OUTPUT ${HDR}
                    COMMAND ${CMAKE_COMMAND} -E make_directory
                            ${CMAKE_BINARY_DIR}/shaders
                    COMMAND ${GLSLANG_VALIDATOR} -V
                            --vn locus_${shader}_spv
                            ${SRC} -o ${HDR}
                    DEPENDS ${SRC}
                    COMMENT "Compiling ${shader}.comp -> SPIR-V")
            else()
                # glslc has no header emitter; wrap its -mfmt=c.
                add_custom_command(
                    OUTPUT ${HDR}
                    COMMAND ${CMAKE_COMMAND} -E make_directory
                            ${CMAKE_BINARY_DIR}/shaders
                    COMMAND ${GLSLC} --target-env=vulkan1.1
                            -mfmt=c ${SRC}
                            -o ${CMAKE_BINARY_DIR}/shaders/${shader}_spv.inc
                    COMMAND ${CMAKE_COMMAND}
                            -DIN=${CMAKE_BINARY_DIR}/shaders/${shader}_spv.inc
                            -DOUT=${HDR}
                            -DVAR=locus_${shader}_spv
                            -P ${CMAKE_SOURCE_DIR}/cmake/wrap_spv.cmake
                    DEPENDS ${SRC}
                    COMMENT "Compiling ${shader}.comp -> SPIR-V")
            endif()
            list(APPEND SHADER_HDRS ${HDR})
        endforeach()
        add_custom_target(locus_shaders DEPENDS ${SHADER_HDRS})
        add_dependencies(locus locus_shaders)
        target_include_directories(locus PRIVATE
            ${CMAKE_BINARY_DIR}/shaders)
        target_compile_definitions(locus PRIVATE
            LOCUS_HAS_VULKAN_KERNELS=1)
        message(STATUS "locus Vulkan kernels: enabled")
    else()
        message(STATUS
            "locus Vulkan kernels: no shader compiler found")
    endif()
else()
    message(STATUS "locus Vulkan: not found, GPU backend disabled")
endif()

# ----------------------------------------------------------------
# CUDA (optional): when a toolkit is present, compile the .cu matvec
# variant and define LOCUS_HAS_CUDA so registry.cpp links the real
# kernels instead of the scalar stub. Absent a toolkit, cuda_stub.cpp
# supplies the fallbacks and the backend reports itself unusable.
# sm_50 (Maxwell, e.g. GTX 750 Ti) still compiles in the CUDA 12.x
# series; a 12.x binary runs on driver >=525 via minor-version compat.
# ----------------------------------------------------------------
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    target_sources(locus PRIVATE
        src/locus/backend/cuda/cuda_matvec.cu)
    set_target_properties(locus PROPERTIES CUDA_ARCHITECTURES "50")
    target_compile_definitions(locus PRIVATE LOCUS_HAS_CUDA=1)
    target_compile_options(locus PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
    target_link_libraries(locus PRIVATE CUDA::cudart)
    message(STATUS
        "locus CUDA: found (${CUDAToolkit_VERSION}), sm_50")
else()
    message(STATUS "locus CUDA: not found, GPU backend disabled")
endif()

enable_testing()
add_subdirectory(tests)
