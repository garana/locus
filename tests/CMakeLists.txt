# Catch2 v3.7.1, vendored amalgamated build (see docs/DESIGN.md
# section 6 for the dependency policy).
add_library(catch2 STATIC
    ${CMAKE_SOURCE_DIR}/third_party/catch2/catch_amalgamated.cpp
)
target_include_directories(catch2 SYSTEM PUBLIC
    ${CMAKE_SOURCE_DIR}/third_party/catch2
)

add_executable(locus_tests
    test_sanity.cpp
    test_block_allocator.cpp
    test_features.cpp
    test_arch_registry.cpp
    test_backend_registry.cpp
    test_bpe_tokenizer.cpp
    test_chat_template.cpp
    test_cpu_ops.cpp
    test_deepseek.cpp
    test_engine.cpp
    test_gguf.cpp
    test_paged_cache.cpp
    test_server.cpp
    test_vulkan.cpp
    test_tokenizer.cpp
    test_model_e2e.cpp
    test_moe.cpp
)
target_link_libraries(locus_tests PRIVATE locus catch2)
target_include_directories(locus_tests SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/httplib
    ${CMAKE_SOURCE_DIR}/third_party/json
)
target_compile_definitions(locus_tests PRIVATE
    LOCUS_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

add_test(NAME unit COMMAND locus_tests)
