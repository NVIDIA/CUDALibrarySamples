# cuSolverDx Library - API Examples

All examples are shipped within [cuSolverDx package](https://developer.nvidia.com/cusolverdx-downloads), and the github examples could be more frequently updated than the examples shipped in the release packages.

## Description

This folder demonstrates cuSolverDx APIs usage.

* [cuSolverDx API documentation](https://docs.nvidia.com/cuda/cusolverdx/index.html)

## Requirements

* [See cuSolverDx requirements](https://docs.nvidia.com/cuda/cusolverdx/get_started/requirement.html)
* CMake 3.18 or newer
* Linux system with installed NVIDIA drivers
* NVIDIA GPU of Volta (SM70) or newer architecture

## Build

* You may specify `CUSOLVERDX_CUDA_ARCHITECTURES` to limit CUDA architectures used for compilation (see [CMake:CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES))
* `mathdx_ROOT` - path to mathDx package (XX.Y - version of the package)

```
mkdir build && cd build
cmake -DCUSOLVERDX_CUDA_ARCHITECTURES=80-real -Dmathdx_ROOT=<path_of_mathdx>/XX.Y ..
make
// Run
ctest
```

## Examples

For the detailed descriptions of the examples please visit [Examples](https://docs.nvidia.com/cuda/cusolverdx/examples/index.html) section of the cuSolverDx documentation.

|              Group           |            Example                |                                  Description                                                      |
|------------------------------|-----------------------------------|---------------------------------------------------------------------------------------------------|
| Introduction Examples        | posv_batched                      | Introduction example with Cholesky factorization and solve                                        |
| Linear Solve Examples        | potrf                             | Cholesky factorization                                                                            |
|                              | potrf_runtime_ld                  | Cholesky factorization with runtime leading dimensions                                            |
|                              | getrf_wo_pivot                    | LU factorization without pivoting                                                                 |
|                              | getrf_partial_pivot               | LU factorization with partial pivoting                                                            |
|                              | gesv_batched_wo_pivot             | Solves batched linear systems without pivoting                                                    |
|                              | gesv_batched_partial_pivot        | Solves batched linear systems with partial pivoting                                               |
|                              | gtsv_batched_wo_pivot             | Solves batched tridiagonal linear systems without pivoting                                        |
| Least Squares Examples       | gels_batched                      | Solves batched least squares problems                                                             |
| Orthogonal Factors Examples  | geqrf_batched                     | QR factorization for batched matrices                                                             |
|                              | ungqr_batched                     | Generates orthogonal/unitary matrix Q from QR factorization                                       |
|                              | unmqr_batched                     | Multiplies matrix by orthogonal/unitary matrix Q from QR factorization                            |
| Symmetric Eigenvalue Examples| heev_batched                      | Computes eigenvalues and eigenvectors of batched Hermitian matrices                               |
|                              | htev_batched                      | Computes eigenvalues and eigenvectors of batched Hermitian tridiagonal matrices                   |
| SVD Examples                 | gesvd_batched                     | Singular value decomposition for batched general matrices                                                 |
|                              | bdsvd_batched                     | Singular value decomposition for batched bidiagonal matrices                                      |
| BLAS Examples                | trsm_batched                      | Batched triangular solve with multiple right-hand sides                                           |
| NVRTC Examples               | nvrtc_potrs                       | Using cuSolverDx with NVRTC runtime compilation and nvJitLink runtime linking                     |
| Performance Examples         | geqrf_batched_performance         | Performance analysis of batched QR factorization                                                  |
| Advanced Examples            | blocked_potrf                     | Cholesky factorization using blocked algorithm for large matrices                                 |
|                              | reg_least_squares                 | Regularized least squares solver                                                                  |
