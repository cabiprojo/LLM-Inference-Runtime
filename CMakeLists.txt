cmake_minimum_required(VERSION 3.16)
project(gpt2_inference_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(gpt2
    src/main.cpp
    src/io/tensor_io.cpp
)

target_include_directories(gpt2 PRIVATE src)
