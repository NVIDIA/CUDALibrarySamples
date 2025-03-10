# nvCOMP Library API examples and benchmarks

## Introduction

nvCOMP is a CUDA library that features generic compression interfaces to enable developers to use high-performance GPU compressors and decompressors in their applications. Example benchmarking results and a brief description of each algorithm are available on the [nvCOMP Developer Page](https://developer.nvidia.com/nvcomp).

This folder demonstrates the nvCOMP library API usage alongside our performance benchmarks.

More information can be found about the examples [here](examples/) and about the benchmarks [here](benchmarks/)

## Supported SM Architectures

- Pascal (SM 6.x)
- Volta (SM 7.0)
- Turing (SM 7.5)
- Ampere (SM 8.0, SM 8.6)
- Ada Lovelace (SM 8.9)
- Hopper (SM 9.0)
- Blackwell (SM 10.0, SM 12.0)

More information can be found about the architectures and compute capabilities on the official NVIDIA website [here](https://developer.nvidia.com/cuda-gpus).

## Supported OSes

Linux, Windows

## Supported CPU Architectures

x86_64, aarch64

## CUDA APIs involved

[nvCOMP](https://docs.nvidia.com/cuda/nvcomp/index.html)

# Prerequisites
- A Linux or Windows system with recent NVIDIA drivers.
- CUDA 11.8 or CUDA 12.5 [toolkit](https://developer.nvidia.com/cuda-downloads).
- CMake 3.18 or above
- gcc-8 or above on aarch64 (with C++11 support)
- gcc-9 or above on x86-64 (with C++11 support)
- The latest nvCOMP release from the NVIDIA [developer zone](https://developer.nvidia.com/nvcomp).