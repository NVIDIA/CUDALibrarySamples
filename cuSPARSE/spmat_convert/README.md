# cuSPARSE Generic APIs - `cusparseSpMatConvert CSR to CSC` [EXPERIMENTAL]

## Description

This sample demonstrates the usage of `cusparseSpMatConvert` for performing *sparse matrix to sparse matrix conversion*, converting a matrix from CSR (Compressed Sparse Row) into CSC (Compressed Sparse Column) storage format.

The sample also shows the generic descriptor helpers `cusparseSpMatGetSizes` and `cusparseSpMatSetPointers`: the target descriptor is created without its row-index/value storage, the sizes are queried after the analysis step, and the storage is attached before the conversion runs.

> **Experimental:** these APIs are only available when the macro `CUSPARSE_ENABLE_EXPERIMENTAL_API` is defined before including `cusparse.h` (as this sample does).

[cusparseSpMatConvert Documentation](https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spmatconvert)

<center>

`A (csr) -> C (csc)`

</center>

## Building

* Command line
    ```bash
    nvcc -I<cuda_toolkit_path>/include spmat_convert_example.c -o spmat_convert_example -lcusparse
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

* **Supported SM Architectures:** SM 7.5, SM 8.0, SM 8.6, SM 8.9, SM 9.0, SM 10.0
* **Supported OSes:** Linux, Windows, QNX, Android
* **Supported CPU Architectures**: x86_64, arm64
* **Supported Compilers**: gcc, clang, Intel icc, Microsoft msvc, Nvidia HPC SDK nvc
* **Language**: `C99`

## Prerequisites

* [CUDA 13.5.0 toolkit](https://developer.nvidia.com/cuda-downloads) (or above) and compatible driver (see [CUDA Driver Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)).
* [CMake 3.9](https://cmake.org/download/) or above on Windows
