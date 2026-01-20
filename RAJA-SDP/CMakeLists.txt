cmake_minimum_required(VERSION 3.14)
project(STP LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11 --default-stream per-thread --expt-extended-lambda")
set(CMAKE_CXX_COMPILER g++)
#set(CMAKE_CXX_COMPILER /usr/tce/packages/gcc/gcc-7.3.1/bin/g++)
option(ENABLE_TESTS Off)

message(STATUS "--------- Building RAJA-SDP ----------")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/")
add_subdirectory(src)
add_subdirectory(exp)
