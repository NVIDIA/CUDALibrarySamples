# libmathdx Library - API Examples

All examples are shipped within [libmathdx package](https://developer.nvidia.com/cublasdx-downloads).

## Description

This folder demonstrates libmathdx APIs usage. libmathdx is a runtime library for MathDx enabling JIT compilation of MathDx operations.
libmathdx exposes a simple C-API interface for ease of adoption in non-C++ projects, and is the backend powering NVIDIA Python projects such as [Warp](https://developer.nvidia.com/warp-python) and [nvmath-python](https://developer.nvidia.com/nvmath-python).

* [libmathdx download page](https://developer.nvidia.com/cublasdx-downloads)
* [libmathdx API documentation](https://docs.nvidia.com/cuda/libmathdx/index.html)

## Requirements

* CMake 3.31 or newer
* Linux or Windows system with installed NVIDIA drivers
* NVIDIA GPU of Turing (SM75) or newer architecture
* CUDA Toolkit 13.X

## Build

* The examples use `find_package(libmathdx CONFIG REQUIRED)`, which consumes the `libmathdx-config.cmake` shipped under `lib/cmake/libmathdx/` in the libmathdx package. Point CMake at the package with `-DCMAKE_PREFIX_PATH=<libmathdx_path>` or `-Dlibmathdx_ROOT=<libmathdx_path>`.

Linux build:
```bash
mkdir build && cd build
cmake .. # or cmake -Dlibmathdx_ROOT=<libmathdx_path> ..
make
# Run
LD_LIBRARY_PATH=<cuda_root_directory>/lib64 ctest --parallel 8
```

Windows build:
```bat
mkdir build && cd build
cmake .. # or cmake -Dlibmathdx_ROOT=<libmathdx_path> ..
cmake --build . --config Release
REM Run
set PATH=%PATH%;<libmathdx_dll_path>;<cuda_library_path>
ctest -C Release --parallel 8
```

Python wheels:
libmathdx is also shipped as downloadable wheels on [PyPI](https://pypi.org/project/nvidia-libmathdx-cu13/):
```bash
pip install nvidia-libmathdx-cu13
```

Optional environment variables:
* `LIBMATHDX_EXAMPLE_VERBOSE` - print generated kernel source to stdout
* `LIBMATHDX_EXAMPLE_DUMP_FILENAME=<path>` - write JITed intermediates to a file

See the [libmathdx documentation](https://docs.nvidia.com/cuda/libmathdx/index.html) for more details.

## Examples

For the detailed descriptions of the examples please visit the Examples section of the [libmathdx documentation](https://docs.nvidia.com/cuda/libmathdx/index.html).

| Group               | Example                                  | Description                                                                     |
|---------------------|------------------------------------------|---------------------------------------------------------------------------------|
| cuFFTDx Examples    | cufftdx                                  | cuFFTDx API introduction example, performs a C2C FFT                            |
|                     | cufftdx_heuristics                       | Performs a C2C FFT with queryable tuning knobs                                  |
| cuBLASDx Examples   | cublasdx                                 | cuBLASDx API introduction example, performs a GEMM                              |
|                     | cublasdx_tensor                          | Performs a GEMM with the Opaque Tensor API                                      |
|                     | cublasdx_pipeline                        | Performs a GEMM using the pipelining APIs                                       |
|                     | cublasdx_pipelining_callback             | Performs a GEMM using pipelining/TMA instructions with a user callback function |
| cuSolverDx Examples | cusolverdx_trsm                          | cuSolverDx API introduction example, performs a triangular matrix solve         |
|                     | cusolverdx_potrf                         | Performs Cholesky factorization                                                 |
|                     | cusolverdx_potrs                         | Performs a linear solve using Cholesky factors                                  |
| cuRANDDx Examples   | curanddx                                 | cuRANDDx API introduction example, generates random numbers                     |
| nvCOMPDx Examples   | nvcompdx                                 | nvCOMPDx API introduction example, builds an LZ4 compress device function       |
|                     | nvcompdx_batch                           | End-to-end LZ4 compress/decompress round-trip across a batch of chunks          |
