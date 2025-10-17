# cuSPARSE Generic APIs - `cusparseSpGEMM`

## Description

This sample demonstrates the usage of `cusparseSpGEMM` for performing *sparse matrix - sparse matrix multiplication*, where all operands are sparse matrices represented in CSR (Compressed Sparse Row) storage format.

[cusparseSpGEMM Documentation](https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spgemm)

<center>

`C = alpha * A * B + beta * C`

![](spgemm.png)
</center>

## Building

* Linux
    ```bash
    make
    ```

* Windows/Linux
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```
    On Windows, instead of running the last build step, open the Visual Studio Solution that was created and build.

## Support

* **Supported SM Architectures:** SM 3.5, SM 3.7, SM 5.0, SM 5.2, SM 5.3, SM 6.0, SM 6.1, SM 6.2, SM 7.0, SM 7.2, SM 7.5, SM 8.0, SM 8.6
* **Supported OSes:** Linux, Windows, QNX, Android
* **Supported CPU Architectures**: x86_64, ppc64le, arm64
* **Supported Compilers**: gcc, clang, Intel icc, IBM xlc, Microsoft msvc, Nvidia HPC SDK nvc
* **Language**: `C99`

## Prerequisites

* [CUDA 11.0 toolkit](https://developer.nvidia.com/cuda-downloads) (or above) and compatible driver (see [CUDA Driver Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)).
* [CMake 3.9](https://cmake.org/download/) or above on Windows
