cmake_minimum_required(VERSION 3.14)

# Project-level configuration.
set(PROJECT_NAME "llamalib")
project(${PROJECT_NAME})

set(BUILD_SHARED_LIBS ON)
set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src/llama.cpp)

if(ANDROID)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DANDROID_ARM_NEON=TRUE -DANDROID_TOOLCHAIN=clang -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DANDROID_ARM_NEON=TRUE -DANDROID_TOOLCHAIN=clang -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23")
    # Log the flags
    message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
endif()

set(LLAMA_NATIVE OFF CACHE BOOL "llama: disable -march=native flag" FORCE)

set(LLAMA_SOURCES
  ${LLAMA_DIR}/ggml/src/ggml.c
  ${LLAMA_DIR}/ggml/include/ggml.h
  ${LLAMA_DIR}/ggml/src/ggml-alloc.c
  ${LLAMA_DIR}/ggml/include/ggml-alloc.h
  ${LLAMA_DIR}/ggml/src/ggml-backend.c
  ${LLAMA_DIR}/ggml/include/ggml-backend.h
  ${LLAMA_DIR}/ggml/src/ggml-quants.c
  ${LLAMA_DIR}/ggml/include/ggml-quants.h
  ${LLAMA_DIR}/src/llama.cpp
  ${LLAMA_DIR}/include/llama.h
  ${LLAMA_DIR}/ggml/src/sgemm.cpp
  ${LLAMA_DIR}/ggml/src/sgemm.h
  ${LLAMA_DIR}/src/unicode.h
  ${LLAMA_DIR}/src/unicode.cpp
  ${LLAMA_DIR}/src/unicode-data.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE ${LLAMA_DIR})

add_library(${PROJECT_NAME} SHARED ${LLAMA_SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib/bin
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib/bin
)