cmake_minimum_required(VERSION 3.14)

# Project-level configuration.
set(PROJECT_NAME "llamalib")
project(${PROJECT_NAME} LANGUAGES CXX)

set(FLUTTER_TARGET_PLATFORM "windows")
set(BUILD_SHARED_LIBS ON)
set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src/llama.cpp)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(LLAMA_NATIVE OFF CACHE BOOL "llama: disable -march=native flag" FORCE)

add_subdirectory(${LLAMA_DIR} shared)