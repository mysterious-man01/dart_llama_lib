cmake_minimum_required(VERSION 3.14)

# Project-level configuration.
set(PROJECT_NAME "llamalib")
project(${PROJECT_NAME})

set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "library install dir" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "llama: disable -march=native flag" FORCE)


if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DANDROID_ARM_NEON=TRUE -DANDROID_TOOLCHAIN=clang -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DANDROID_ARM_NEON=TRUE -DANDROID_TOOLCHAIN=clang -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23")
    # Log the flags
    message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
endif()

set(LLAMA_NATIVE OFF CACHE BOOL "llama: disable -march=native flag" FORCE)

add_subdirectory(${LLAMA_DIR})

#add_library(${PROJECT_NAME} SHARED
#    ${LLAMA_DIR}/ggml/src/ggml.c
#    ${LLAMA_DIR}/ggml/src/ggml-alloc.c
#    ${LLAMA_DIR}/ggml/src/ggml-backend.c
#    ${LLAMA_DIR}/ggml/src/ggml-quants.c
#    ${LLAMA_DIR}/ggml/src/ggml-aarch64.c
#    ${LLAMA_DIR}/src/llama.cpp
#    ${LLAMA_DIR}/src/unicode.cpp
#    ${LLAMA_DIR}/src/unicode-data.cpp
#)

#target_include_directories(${PROJECT_NAME} PRIVATE
#    ${LLAMA_DIR}/ggml/include
#    ${LLAMA_DIR}/ggml/src
#    ${LLAMA_DIR}/include
#)

#set_target_properties(${PROJECT_NAME} PROPERTIES
#    PUBLIC_HEADER ${LLAMA_DIR}/include/llama.h
#    OUTPUT_NAME "libllama"
#    BUILD_WITH_INSTALL_RPATH TRUE
#    INSTALL_RPATH "$ORIGIN"
#)

#target_compile_definitions(${PROJECT_NAME} PUBLIC DART_SHARED_LIB)

#if(ANDROID_ABI STREQUAL "arm64-v8a")
#    target_compile_options(${PROJECT_NAME} PRIVATE -march=armv8.4-a+fp16+dotprod)
#endif()