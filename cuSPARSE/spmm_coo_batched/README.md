# cuSPARSE Generic APIs - `Batched cusparseSpMM COO`

## Description

This sample demonstrates the usage of `cusparseSpMM` for performing *batched sparse matrix - dense matrix multiplication*, where the sparse matrix is represented in COO (Coordinate) storage format.

[cusparseSpMM Documentation](https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spmm)

<center>

`C[1] = alpha * A[1] * B[1] + beta * C[1]`
`C[2] = alpha * A[2] * B[2] + beta * C[2]`

![](spmm_coo_batched.png)
</center>

## Building

* Command line
    ```bash
    nvcc -I<cuda_toolkit_path>/include spmm_coo_batched_example.c -o spmm_coo_batched_example -lcusparse
    ```

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

* **Supported SM Architectures:** SM 3.5, SM 3.7, SM 5.0, SM 5.2, SM 5.3, SM 6.0, SM 6.1, SM 6.2, SM 7.0, SM 7.2, SM 7.5, SM 8.0, SM 8.6, SM 8.9, SM 9.0
* **Supported OSes:** Linux, Windows, QNX, Android
* **Supported CPU Architectures**: x86_64, ppc64le, arm64
* **Supported Compilers**: gcc, clang, Intel icc, IBM xlc, Microsoft msvc, Nvidia HPC SDK nvc
* **Language**: `C99`

## Prerequisites

* [CUDA 11.2 toolkit](https://developer.nvidia.com/cuda-downloads) (or above) and compatible driver (see [CUDA Driver Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)).
* [CMake 3.9](https://cmake.org/download/) or above on Windows
