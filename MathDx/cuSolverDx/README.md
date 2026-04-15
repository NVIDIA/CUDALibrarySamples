# cuSolverDx Library - API Examples

All examples are shipped within [cuSolverDx package](https://developer.nvidia.com/cusolverdx-downloads), and the github examples could be more frequently updated than the examples shipped in the release packages.

## Description

This folder demonstrates cuSolverDx APIs usage.

* [cuSolverDx API documentation](https://docs.nvidia.com/cuda/cusolverdx/index.html)

## Requirements

* [See cuSolverDx requirements](https://docs.nvidia.com/cuda/cusolverdx/get_started/requirement.html)
* CMake 3.18 or newer
* Linux system with installed NVIDIA drivers
* NVIDIA GPU of Turing (SM75) or newer architecture

## Build

* You may specify `CUSOLVERDX_CUDA_ARCHITECTURES` to limit CUDA architectures used for compilation (see [CMake:CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES))
* `mathdx_ROOT` - path to mathDx package (XX.Y - version of the package)

```
mkdir build && cd build
cmake -DCUSOLVERDX_CUDA_ARCHITECTURES=80-real -Dmathdx_ROOT=/opt/nvidia/mathdx/XX.Y ..
make
# Run
ctest
```

## Examples

For the detailed descriptions of the examples please visit [Examples](https://docs.nvidia.com/cuda/cusolverdx/examples/index.html) section of the cuSolverDx documentation.

| Group                        | Example                           | Description                                                                                        |
|------------------------------|-----------------------------------|----------------------------------------------------------------------------------------------------|
| Introduction Examples        | posv_batched_block                | Introduction example with Cholesky factorization and solve (block execution)                       |
| Linear Solve Examples        | potrf_batched_thread              | Cholesky factorization (thread execution)                                                          |
|                              | posv_batched_thread               | Cholesky factorization and solve (thread execution)                                                |
|                              | potrf_block                       | Cholesky factorization (block execution)                                                           |
|                              | potrf_runtime_ld_block            | Cholesky factorization with runtime leading dimensions (block execution)                           |
|                              | getrf_wo_pivot_block              | LU factorization without pivoting (block execution)                                                |
|                              | getrf_partial_pivot_block         | LU factorization with partial pivoting (block execution)                                           |
|                              | gesv_batched_wo_pivot_thread      | Solves batched linear systems without pivoting (thread execution)                                  |
|                              | gesv_batched_wo_pivot_block       | Solves batched linear systems without pivoting (block execution)                                   |
|                              | gesv_batched_partial_pivot_thread | Solves batched linear systems with partial pivoting (thread execution)                             |
|                              | gesv_batched_partial_pivot_block  | Solves batched linear systems with partial pivoting (block execution)                              |
|                              | gtsv_batched_wo_pivot_thread      | Solves batched tridiagonal linear systems without pivoting (thread execution)                      |
|                              | gtsv_batched_wo_pivot_block       | Solves batched tridiagonal linear systems without pivoting (block execution)                       |
| Least Squares Examples       | gels_batched_thread               | Solves batched least squares problems (thread execution)                                           |
|                              | gels_batched_block                | Solves batched least squares problems (block execution)                                            |
| Orthogonal Factors Examples  | geqrf_batched_thread              | QR factorization for batched matrices (thread execution)                                           |
|                              | geqrf_batched_block               | QR factorization for batched matrices (block execution)                                            |
|                              | unmqr_batched_thread              | Multiplies matrix by Q from QR factorization (thread execution)                                    |
|                              | unmqr_batched_block               | Multiplies matrix by Q from QR factorization (block execution)                                     |
|                              | ungqr_batched_thread              | Generates orthogonal matrix Q from QR factorization (thread execution)                             |
|                              | ungqr_batched_block               | Generates orthogonal matrix Q from QR factorization (block execution)                              |
| Symmetric Eigenvalue Examples| heev_batched_thread               | Eigenvalues and eigenvectors of batched Hermitian matrices (thread execution)                      |
|                              | heev_batched_block                | Eigenvalues and eigenvectors of batched Hermitian matrices (block execution)                       |
|                              | htev_batched_thread               | Eigenvalues and eigenvectors of batched Hermitian tridiagonal matrices (thread execution)          |
|                              | htev_batched_block                | Eigenvalues and eigenvectors of batched Hermitian tridiagonal matrices (block execution)           |
| SVD Examples                 | gesvd_batched_thread              | Singular value decomposition for batched general matrices (thread execution)                       |
|                              | gesvd_batched_block               | Singular value decomposition for batched general matrices (block execution)                        |
|                              | bdsvd_batched_thread              | Singular value decomposition for batched bidiagonal matrices (thread execution)                    |
|                              | bdsvd_batched_block               | Singular value decomposition for batched bidiagonal matrices (block execution)                     |
| BLAS Examples                | trsm_batched_block                | Batched triangular solve with multiple right-hand sides (block execution)                          |
|                              | trsm_batched_thread               | Batched triangular solve with multiple right-hand sides (thread execution)                         |
|                              | trsm_batched_thread_advanced      | Batched triangular solve with advanced options (thread execution)                                  |
| NVRTC Examples               | nvrtc_potrs                       | cuSolverDx with NVRTC runtime compilation and nvJitLink runtime linking                            |
| Performance Examples         | geqrf_batched_performance         | Performance analysis of batched QR factorization                                                   |
| Advanced Examples            | blocked_potrf                     | Cholesky factorization using blocked algorithm for large matrices (requires cuBLASDx)              |
|                              | reg_least_squares                 | Regularized least squares solver (requires cuBLASDx)                                               |
