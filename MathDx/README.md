# MathDx Package

The **MathDx** package is a comprehensive collection of NVIDIA device extension libraries that empower CUDA developers to run
advanced mathematical operations directly inside their GPU kernels, leveraging kernel fusion for maximum efficiency and flexibility.
These libraries are crafted to work seamlessly together, providing a unified solution for high-performance computations, data processing,
and random number generation — all without unnecessary host-device data transfers.
MathDx delivers performance portability across hardware generations, abstracting low-level GPU architecture details
so developers can focus on algorithms rather than hardware-specific tuning.

* **cuBLASDx**: Device-side extensions for selected linear algebra routines, including efficient General Matrix Multiplication (GEMM) performed within kernels.
* **cuFFTDx**: Device-side Fast Fourier Transform library, enabling in-kernel FFT calculations for signal processing and scientific computation.
* **cuSolverDx**: Device-side matrix factorization, linear solve, least squares, eigenvalue solver, and singular value decomposition routines, supporting scientific and engineering workflows within a kernel.
* **cuRANDDx**:  Random number generation library aiming to be a modern replacement for `cuRAND RNG device APIs <https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview>`_.
* **nvCOMPDx**: Compression and decompression capabilities built into device code, essential for high-throughput streaming and storage applications.

**See [MathDx documentation](https://docs.nvidia.com/cuda/mathdx/index.html) for more details.**

# MathDx Examples

This folder includes examples for all libraries that are part of [MathDx package](https://docs.nvidia.com/cuda/mathdx/index.html): cuFFTDx, cuBLASDx, cuSolverDx, cuRANDDx, and nvCOMPDx. The examples are also shipped in the latest MathDx package, however, this repository may be updated more often, i.e., between MathDx releases.

## [cuBLASDx](cuBLASDx)

* [cuBLASDx download page](https://developer.nvidia.com/cublasdx-downloads)
* [cuBLASDx API documentation](https://docs.nvidia.com/cuda/cublasdx/index.html)

#### Examples

| Group                 | Subgroup       | Example                           | Description                                                                    |
|-----------------------|----------------|-----------------------------------|--------------------------------------------------------------------------------|
| Introduction Examples |                | 01_introduction_example           | cuBLASDx API introduction example                                              |
| 			|                | 01_introduction_pipeline          | cuBLASDx Pipeline API introduction example                                     |
| Simple GEMM Examples  | Basic Example  | 02_simple_gemm_fp32               | Performs fp32 GEMM                                                             |
|                       |                | 02_simple_gemm_int8_int8_int32    | Performs integral GEMM using Tensor Cores                                      |
|                       |                | 02_simple_gemm_fp8                | Performs fp8 GEMM                                                              |
|                       |                | 02_simple_gemm_mixed_precision    | Performs a mixed precision GEMM                                                |
|                       |                | 03_simple_gemm_cfp16              | Performs complex fp16 GEMM                                                     |
|                       |                | 03_simple_gemm_std_complex_fp32   | Performs GEMM with cuda::std::complex as data type                             |
|                       |                | 04_blockdim_gemm_fp16             | BLAS execution with different block dimensions                                 |
|                       | Other          | 05_batched_gemm_fp64              | Manual batching in a single CUDA block                                         |
|                       |                | 06_simple_gemm_leading_dimensions | Performs GEMM with non-default leading dimensions                              |
|                       |                | 07_simple_gemm_transform          | Performs GEMM with custom load and store operators                             |
|                       |                | 08_simple_gemm_fp32_decoupled     | Performs fp32 GEMM using 16-bit input type to save on storage and transfers    |
|                       |                | 09_simple_gemm_custom_layout      | Performs GEMM with a custom user provided CuTe layout                          |
|                       |                | 09_simple_gemm_aat                | Performs GEMM where C = A * A^T                                                |
| GEMM Performance      |                | 10_single_gemm_performance        | Benchmark for single GEMM                                                      |
|                       |                | 11_device_gemm_performance        | Benchmark entire device GEMMs using cuBLASDx for single tile                   |
| Advanced Examples     |                | 12_gemm_device_partial_sums       | Enhance GEMM precision by performing higher precision partial sum accumulation |
|                       |                | 14_fused_gemm_performance         | Benchmark for 2 GEMMs fused into a single kernel                               |
| Advanced Examples     | Fusion         | 14_gemm_fusion                    | Performs 2 GEMMs in a single kernel                                            |
|                       |                | 13_gemm_fft                       | Perform GEMM and FFT in a single kernel                                        |
|                       |                | 13_gemm_fft_fp16                  | Perform GEMM and FFT in a single kernel (half-precision complex type)          |
|                       |                | 13_gemm_fft_performance           | Benchmark for GEMM and FFT fused into a single kernel                          |
|                       | Emulation      | 16_dgemm_emulation                | Emulate double precision GEMM using lower precision operations (Ozaki scheme)  |
| NVRTC Examples        |                | 15_nvrtc_gemm                     | Performs GEMM, kernel is compiled using NVRTC                                  |


## [cuFFTDx](cuFFTDx)

* [cuFFTDx download page](https://developer.nvidia.com/cufftdx-downloads)
* [cuFFTDx API documentation](https://docs.nvidia.com/cuda/cufftdx/index.html)

#### Examples


|              Group           |         Subgroup         |            Example                |                                  Description                                  |
|------------------------------|--------------------------|-----------------------------------|-------------------------------------------------------------------------------|
| Introduction Examples        |                          | introduction_example              | cuFFTDx API introduction                                                      |
| Simple FFT Examples          | Thread FFT Examples      | simple_fft_thread                 | Complex to complex thread FFT                                                 |
|                              |                          | simple_fft_thread_fp16            | Complex to complex thread FFT half precision                                  |
|                              |                          |                                   |                                                                               |
|                              | Block FFT Examples       | simple_fft_block                  | Complex to complex block FFT                                                  |
|                              |                          | simple_fft_block_r2c              | Real to complex block FFT                                                     |
|                              |                          | simple_fft_block_c2r              | Complex to real block FFT                                                     |
|                              |                          | simple_fft_block_half2            | Complex to complex block FFT with `__half2` as data type                      |
|                              |                          | simple_fft_block_fp16             | Complex to complex block FFT half precision                                   |
|                              |                          | simple_fft_block_r2c_fp16         | Real to complex block FFT half precision                                      |
|                              |                          | simple_fft_block_c2r_fp16         | Complex to real block FFT half precision                                      |
|                              |                          |                                   |                                                                               |
|                              | Extra Block FFT Examples | simple_fft_block_shared           | Complex to complex block FFT shared memory API                                |
|                              |                          | simple_fft_block_std_complex      | Complex to complex block FFT with `cuda::std::complex` as data type           |
|                              |                          | simple_fft_block_cub_io           | Complex to complex block FFT with `CUB` used for loading/storing data         |
|                              |                          |                                   |                                                                               |
| NVRTC Examples               |                          | nvrtc_fft_thread                  | Complex to complex thread FFT                                                 |
|                              |                          | nvrtc_fft_block                   | Complex to complex block FFT                                                  |
|                              |                          |                                   |                                                                               |
| FFT Performance              |                          | block_fft_performance             | Benchmark for C2C block FFT                                                   |
|                              |                          | block_fft_performance_many        | Benchmark for C2C/R2C/C2R block FFT                                           |
|                              |                          |                                   |                                                                               |
| Convolution Examples         |                          | convolution                       | Simplified FFT convolution                                                    |
|                              |                          | convolution_r2c_c2r               | Simplified R2C C2R FFT convolution                                            |
|                              |                          | convolution_padded                | R2C C2R FFT convolution with optimization and zero padding                    |
|                              |                          | convolution_performance           | Benchmark for FFT convolution using cuFFTDx and cuFFT                         |
|                              |                          | conv_3d/convolution_3d            | cuFFTDx fused 3D convolution with preprocessing, filtering and postprocessing |
|                              |                          | conv_3d/convolution_3d_r2c        | cuFFTDx fused 3D R2C/C2R FFT convolution                                      |
|                              |                          | conv_3d/convolution_3d_c2r        | cuFFTDx fused 3D C2R/R2C FFT convolution                                      |
|                              |                          | conv_3d/convolution_3d_padded     | cuFFTDx fused 3D FFT convolution using zero padding                           |
|                              |                          | conv_3d/convolution_3d_padded_r2c | uFFTDx fused 3D R2C/C2R FFT convolution with zero padding                     |
|                              |                          |                                   |                                                                               |
| 2D/3D FFT Advanced Examples  |                          | fft_2d                            | Example showing how to perform 2D FP32 C2C FFT with cuFFTDx                   |
|                              |                          | fft_2d_r2c_c2r                    | Example showing how to perform 2D FP32 R2C/C2R convolution with cuFFTDx       |
|                              |                          | fft_2d_single_kernel              | 2D FP32 FFT in a single kernel using Cooperative Groups kernel launch         |
|                              |                          | fft_3d_box_single_block           | Small 3D FP32 FFT that fits into a single block, each dimension is different  |
|                              |                          | fft_3d_cube_single_block          | Small 3D (equal dimensions) FP32 FFT that fits into a single block            |
|                              |                          | fft_3d                            | Example showing how to perform 3D FP32 C2C FFT with cuFFTDx                   |
|                              |                          |                                   |                                                                               |
| Mixed Precision Examples     |                          | mixed_precision_fft_1d            | Example showing how to use separate storage and compute precisions            |
|                              |                          | mixed_precision_fft_2d            | Mixed precision 2D FFT with benchmarking and accuracy comparison              |
|                              |                          |                                   |                                                                               |


## [cuSolverDx](cuSolverDx)

* [cuSolverDx download page](https://developer.nvidia.com/cusolverdx-downloads)
* [cuSolverDx API documentation](https://docs.nvidia.com/cuda/cusolverdx/index.html)

#### Examples

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


## [cuRANDDx](cuRANDDx)

* [cuRANDDx download page](https://developer.nvidia.com/curanddx-downloads)
* [cuRANDDx API documentation](https://docs.nvidia.com/cuda/curanddx/index.html)

#### Examples

|              Group           |            Example                  |                                  Description                                                    |
|------------------------------|-------------------------------------|-------------------------------------------------------------------------------------------------|
| Introduction Examples        | philox_thread_api                   | Introduction example with Philox random number generator                                        |
| Thread API Examples          | xorwow_init_and_generate_thread_api | XORWOW random number generator with initialization and generation                               |
|                              | sobol_thread_api                    | Sobol quasi-random number generator                                                             |
|                              | pcg_thread_api                      | PCG random number generator                                                                     |
|                              | mrg_two_distributions_thread_api    | Multiple random number distributions using MRG32k3a generator                                   |
| NVRTC Examples               | nvrtc_pcg_thread_api                | Using cuRANDDx with NVTRC runtime compilation and nvJitLink runtime linkin g                    |


## [nvCOMPDx](nvCOMPDx)

* [nvCOMPDx download page](https://developer.nvidia.com/nvcompdx-downloads)
* [nvCOMPDx API documentation](https://docs.nvidia.com/cuda/nvcompdx/index.html)

#### Examples

|              Group           |            Example                      |                                  Description                                                |
|------------------------------|-----------------------------------------|---------------------------------------------------------------------------------------------|
| Introduction Examples        | lz4_gpu_compression_introduction        | Introductory example showcasing GPU LZ4 compression                                         |
| LZ4 GPU                      | lz4_gpu_compression_decompression       | Warp-level GPU LZ4 compression and decompression                                            |
| LZ4 GPU and CPU              | lz4_cpu_compression_gpu_decompression   | CPU compression with Warp-level GPU LZ4 decompression                                       |
|                              | lz4_gpu_compression_cpu_decompression   | Warp-level GPU LZ4 compression with CPU decompression                                       |
| ANS GPU                      | ans_gpu_compression_decompression       | Block-level GPU ANS compression and decompression                                           |
|                              | ans_gpu_decompression_reduction         | Fused block-level GPU ANS decompression followed by block-wide reduction                    |
| NVRTC Examples               | lz4_cpu_compression_nvrtc_decompression | CPU compression, Warp-level GPU NVTRC + nvJitLink LZ4 decompression                         |
