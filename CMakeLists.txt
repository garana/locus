cmake_minimum_required(VERSION 3.24)
project(cppllm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(cppllm
    src/cppllm/backend/cpu_ops.cpp
    src/cppllm/backend/vulkan/context.cpp
    src/cppllm/engine/engine.cpp
    src/cppllm/gguf/gguf.cpp
    src/cppllm/kv/block_allocator.cpp
    src/cppllm/kv/paged_cache.cpp
    src/cppllm/sys/features.cpp
    src/cppllm/model/llama.cpp
    src/cppllm/server/engine_loop.cpp
    src/cppllm/server/server.cpp
    src/cppllm/sys/mmap_file.cpp
    src/cppllm/tok/tokenizer.cpp
)
target_include_directories(cppllm SYSTEM PRIVATE
    third_party/httplib
    third_party/json
)
find_package(Threads REQUIRED)
target_link_libraries(cppllm PUBLIC Threads::Threads)

add_executable(cppllm-run tools/run_main.cpp)
target_link_libraries(cppllm-run PRIVATE cppllm)

add_executable(cppllm-server tools/server_main.cpp)
target_link_libraries(cppllm-server PRIVATE cppllm)
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

    find_program(GLSLC glslc)
    find_program(GLSLANG_VALIDATOR glslangValidator)
    set(SHADER_SRC
        ${CMAKE_SOURCE_DIR}/src/cppllm/backend/vulkan/matvec.comp)
    set(SHADER_HDR ${CMAKE_BINARY_DIR}/shaders/matvec_spv.h)
    if(GLSLANG_VALIDATOR)
        add_custom_command(
            OUTPUT ${SHADER_HDR}
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    ${CMAKE_BINARY_DIR}/shaders
            COMMAND ${GLSLANG_VALIDATOR} -V --vn cppllm_matvec_spv
                    ${SHADER_SRC} -o ${SHADER_HDR}
            DEPENDS ${SHADER_SRC}
            COMMENT "Compiling matvec.comp -> SPIR-V header")
    elseif(GLSLC)
        # glslc has no header emitter; go via xxd-style embed.
        add_custom_command(
            OUTPUT ${SHADER_HDR}
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    ${CMAKE_BINARY_DIR}/shaders
            COMMAND ${GLSLC} --target-env=vulkan1.1 -mfmt=c
                    ${SHADER_SRC}
                    -o ${CMAKE_BINARY_DIR}/shaders/matvec_spv.inc
            COMMAND ${CMAKE_COMMAND}
                    -DIN=${CMAKE_BINARY_DIR}/shaders/matvec_spv.inc
                    -DOUT=${SHADER_HDR}
                    -P ${CMAKE_SOURCE_DIR}/cmake/wrap_spv.cmake
            DEPENDS ${SHADER_SRC}
            COMMENT "Compiling matvec.comp -> SPIR-V header")
    endif()
    if(GLSLANG_VALIDATOR OR GLSLC)
        add_custom_target(cppllm_shaders DEPENDS ${SHADER_HDR})
        add_dependencies(cppllm cppllm_shaders)
        target_include_directories(cppllm PRIVATE
            ${CMAKE_BINARY_DIR}/shaders)
        target_compile_definitions(cppllm PRIVATE
            CPPLLM_HAS_VULKAN_KERNELS=1)
        message(STATUS "cppllm Vulkan kernels: enabled")
    else()
        message(STATUS
            "cppllm Vulkan kernels: no shader compiler found")
    endif()
else()
    message(STATUS "cppllm Vulkan: not found, GPU backend disabled")
endif()

enable_testing()
add_subdirectory(tests)
