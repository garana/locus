# Catch2 v3.7.1, vendored amalgamated build (see docs/DESIGN.md
# section 6 for the dependency policy).
add_library(catch2 STATIC
    ${CMAKE_SOURCE_DIR}/third_party/catch2/catch_amalgamated.cpp
)
target_include_directories(catch2 SYSTEM PUBLIC
    ${CMAKE_SOURCE_DIR}/third_party/catch2
)

add_executable(cppllm_tests
    test_sanity.cpp
    test_block_allocator.cpp
    test_features.cpp
    test_backend_registry.cpp
    test_bpe_tokenizer.cpp
    test_cpu_ops.cpp
    test_engine.cpp
    test_gguf.cpp
    test_paged_cache.cpp
    test_server.cpp
    test_vulkan.cpp
    test_tokenizer.cpp
    test_model_e2e.cpp
)
target_link_libraries(cppllm_tests PRIVATE cppllm catch2)
target_include_directories(cppllm_tests SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/httplib
    ${CMAKE_SOURCE_DIR}/third_party/json
)
target_compile_definitions(cppllm_tests PRIVATE
    CPPLLM_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

add_test(NAME unit COMMAND cppllm_tests)
