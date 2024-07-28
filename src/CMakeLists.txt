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

add_subdirectory(${LLAMA_DIR} shared)