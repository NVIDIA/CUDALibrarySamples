# cusparseLt - `cusparseMatMul`

## Description

This sample demonstrates the usage of `cuspaseLt` library and `cusparseMatMul` APIs for performing *structured matrix - dense matrix multiplication* by exploiting NVIDIA *Sparse Tensor Cores*, where the structured matrix is compressed with 2:4 ratio.

[cusparseLt Documentation](https://docs.nvidia.com/cuda/cusparselt/index.html)

<center>

`C = alpha * A * B + beta * C`

</center>

where `A`, `B`, `C` are dense matrices

## Building

`cusparseLt` is currently available only on Linux x86_64. The support to other platforms will be extended in the next releases.

* Linux
    ```bash
    make
    ```

* or in alternative:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

## Support

* **Supported SM Architectures:** SM 8.0
* **Supported OSes:** Linux
* **Supported CPU Architectures**: x86_64
* **Supported Compilers**: gcc, clang, icc, xlc, msvc, pgi
* **Language**: `C++14`

## Prerequisites

* [CUDA 11.0 toolkit](https://developer.nvidia.com/cuda-11.0-download-archive) and compatible driver (see [CUDA Driver Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)).
* [cusparseLt v0.0.1](https://developer.nvidia.com/cusparselt/downloads) (or above)
* [CMake 3.9](https://cmake.org/download/) or above
