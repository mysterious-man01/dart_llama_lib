# Dart Llama Lib [**DEPRECATED**]

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A library that pack [Llama.cpp](https://github.com/ggerganov/llama.cpp) to work on Dart.

Run any .gguf model file on Linux, Windows and Android.

> [IMPORTANT]
- Use [llamacpp4dart](https://github.com/mysterious-man01/llamacpp4dart) instead
- This project is on alfa preview and can have unespected bugs!
- It is not fully implemented!

## How to use

- A Dart/Flutter instalation.
- A C/C++ and CMake instalation. 
- Clone this repository.
- Clone the [Llama.cpp](https://github.com/ggerganov/llama.cpp) project inside `dart_llama_lib/src/llama.cpp`.
- Inside `dart_llama_lib/src` run the folowing comands:

```bash
cmake .
```
and
```bash
cmake --build .
```
- Put the resultant files (`*.dll`, `*.so`) somewhere your code can access.

## Source Projects and References

- Ggerganov's [Llama.cpp](https://github.com/ggerganov/llama.cpp) project.
- Netdur's [llama_cpp_dart](https://github.com/netdur/llama_cpp_dart) project.
- Dane Madsen's [maid_llm](https://github.com/Mobile-Artificial-Intelligence/maid_llm) project.

## Licence

This project is licensed under the MIT License - see the `LICENSE.md` file for details.
