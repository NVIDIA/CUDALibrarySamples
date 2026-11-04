# cuSPARSE Generic APIs - Symmetric SpMV

## Description

This sample demonstrates descriptor-based sparse matrix-dense vector multiplication with `cusparseSpMVGeneric`. The symmetric matrix is represented in CSR (Compressed Sparse Row) format with only its lower triangle stored.

`cusparseSpMVGeneric` writes its result to a separate output vector and leaves the input vector `Y` unchanged:

`Z = alpha * A * X + beta * Y`

The sample sets `CUSPARSE_SPMAT_SYMMETRY_TYPE` to `CUSPARSE_SYMMETRIC` and `CUSPARSE_SPMAT_STORAGE` to `CUSPARSE_STORE_LOWER`. It also shows how to configure the operation, algorithm, and external workspace through a `cusparseSpMVDescr_t` descriptor.

## Building

* Command line
    ```bash
    nvcc -I<cuda_toolkit_path>/include spmv_symmetric_example.c -o spmv_symmetric_example -lcusparse
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
    cmake --build .
    ```

To compile against a cuSPARSE source build instead of an installed library,
configure with `-DCUSPARSE_INCLUDE_DIR=<source>/include` and
`-DCUSPARSE_LIBRARY_DIR=<source>/build/src`.

The Makefile variables can be overridden for the same purpose. Add the source
library directory to the executable's runtime search path so it loads the
intended cuSPARSE library:
`make INC="-I<source>/include -I<cuda_toolkit_path>/include" LIBS="-L<source>/build/src -Xlinker=-rpath -Xlinker=<source>/build/src -lcusparse"`.

## Support

* **Supported SM Architectures:** SM 7.5, SM 8.0, SM 8.6, SM 8.9, SM 9.0, SM 10.0
* **Supported OSes:** Linux, Windows, QNX, Android
* **Supported CPU Architectures:** x86_64, arm64
* **Supported Compilers:** gcc, clang, Intel icc, Microsoft msvc, NVIDIA HPC SDK nvc
* **Language:** `C99`

## Prerequisites

* A CUDA Toolkit release that provides `cusparseSpMVGeneric` and a compatible driver.
* CMake 3.9 or above for the CMake build.
