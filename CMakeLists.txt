cmake_minimum_required(VERSION 3.24)
project(cppllm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(cppllm
    src/cppllm/kv/block_allocator.cpp
)
target_include_directories(cppllm PUBLIC src)

enable_testing()
add_subdirectory(tests)
