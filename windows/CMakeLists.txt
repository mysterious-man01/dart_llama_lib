cmake_minimum_required(VERSION 3.14)

# Project-level configuration.
set(PROJECT_NAME "llama_cpp_dart")
project(${PROJECT_NAME} LANGUAGES CXX)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../src" src)