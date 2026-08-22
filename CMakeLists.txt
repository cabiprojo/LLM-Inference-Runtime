cmake_minimum_required(VERSION 3.16)
project(gpt2_inference_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -march=native lets the compiler target this machine's actual instruction set
# (AVX2/FMA here) instead of a generic x86-64 baseline -- without it, none of
# the auto-vectorization we just measured actually happens in the real build
add_compile_options(-march=native)

add_library(ops STATIC
    src/ops.cpp
    src/io/tensor_io.cpp
)
target_include_directories(ops PUBLIC src)

add_executable(gpt2 src/main.cpp)
target_link_libraries(gpt2 PRIVATE ops)

add_executable(bench_matmul src/benchmark_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE ops)
