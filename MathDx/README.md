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
* **cuRANDDx**:  Random number generation library aiming to be a modern replacement for [cuRAND RNG device APIs](https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview).
* **nvCOMPDx**: Compression and decompression capabilities built into device code, essential for high-throughput streaming and storage applications.

**See [MathDx documentation](https://docs.nvidia.com/cuda/mathdx/index.html) for more details.**

# MathDx Examples

This folder includes examples for all libraries that are part of [MathDx package](https://docs.nvidia.com/cuda/mathdx/index.html): cuFFTDx, cuBLASDx, cuSolverDx, cuRANDDx, and nvCOMPDx. The examples are also shipped in the latest MathDx package, however, this repository may be updated more often, i.e., between MathDx releases.

## [cuBLASDx](cuBLASDx)

* [cuBLASDx download page](https://developer.nvidia.com/cublasdx-downloads)
* [cuBLASDx API documentation](https://docs.nvidia.com/cuda/cublasdx/index.html)

#### Examples

| Group               | Subgroup       | Example                           | Description                                                                                                           |
| ------------------- | -------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Introduction        | GEMM           | 01_introduction_example           | Walks through descriptor creation and the shared-memory and register-result GEMM APIs.                                |
|                     | Pipeline       | 01_introduction_pipeline          | Introduces the host pipeline object, device handle, tile pipeline, and epilogue flow.                                 |
| Simple GEMM         | Basic          | 02_simple_gemm_fp32               | Performs a checked fp32 GEMM and is the smallest general-purpose starting point.                                      |
|                     | Precision      | 02_simple_gemm_mixed_precision    | Uses different precisions and storage types for matrices A, B, and C.                                                 |
|                     | Precision      | 02_simple_gemm_int8_int8_int32    | Performs integral GEMM using int8 inputs and int32 accumulation.                                                      |
|                     | Precision      | 02_simple_gemm_fp8                | Performs fp8 GEMM on architectures that support the required fp8 operations.                                          |
|                     | Complex        | 03_simple_gemm_cfp16              | Performs complex half-precision GEMM.                                                                                 |
|                     | Complex        | 03_simple_gemm_std_complex_fp32   | Uses cuda::std::complex<float> as the matrix element type.                                                            |
|                     | Layout         | 06_simple_gemm_leading_dimensions | Demonstrates non-default leading dimensions and padded matrix storage.                                                |
|                     | Layout         | 09_simple_gemm_custom_layout      | Uses custom CuTe layouts for shared-memory matrices.                                                                  |
|                     | Layout         | 09_simple_gemm_aat                | Computes C = A * A^T while reusing one shared-memory allocation for both views of A.                                  |
|                     | Transform      | 07_simple_gemm_transform          | Applies element-wise load and store transform operators around GEMM.                                                  |
|                     | Transform      | 18_gemm_conj_transpose            | Applies tensor views such as conjugate transpose to GEMM inputs without materializing a copy.                         |
|                     | Register I/O   | 08_simple_gemm_fp32_decoupled     | Uses lower-precision input/output storage with higher-precision computation and register fragments.                   |
| Runtime Compilation | NVRTC          | 15_nvrtc_gemm                     | Compiles a GEMM kernel at runtime with NVRTC and passes cuBLASDx headers to device code.                              |
|                     | NVRTC          | 15_nvrtc_trsm                     | Compiles a block-level TRSM kernel at runtime with NVRTC. Requires cuBLASDx fatbin/LTO support.                       |
| Performance         | Block GEMM     | 10_single_gemm_performance        | Benchmarks a single block-level GEMM tile configuration.                                                              |
|                     | Device GEMM    | 11_device_gemm_performance        | Builds a full-device GEMM from cuBLASDx tiles and compares with a reference path.                                     |
|                     | Fusion         | 14_fused_gemm_performance         | Benchmarks two fused GEMMs against an unfused reference path.                                                         |
| Advanced GEMM       | Batching       | 19_batched_gemm                   | Demonstrates rank-3 tensor batching for non-pipelined GEMM. Requires CUDA Toolkit 13.1 or newer.                      |
|                     | Batching       | 19_batched_gemm_pipeline          | Demonstrates rank-3 tensor batching with the pipeline API. Requires CUDA Toolkit 13.1 or newer.                       |
|                     | Batching       | 05_batched_gemm_fp64              | Shows manual batching inside one CUDA block with BlockDim.                                                            |
|                     | Block shape    | 04_blockdim_gemm_fp16             | Shows how launch dimensions interact with BlockDim and participating threads.                                         |
|                     | Accuracy       | 12_gemm_device_partial_sums       | Offloads partial accumulation to a higher-precision register array.                                                   |
|                     | Fusion         | 14_gemm_fusion                    | Performs two dependent GEMMs in one CUDA kernel.                                                                      |
|                     | cuFFTDx fusion | 13_gemm_fft                       | Fuses GEMM and FFT in one kernel. Requires cuFFTDx.                                                                   |
|                     | cuFFTDx fusion | 13_gemm_fft_fp16                  | Fuses half-precision complex GEMM and FFT. Requires cuFFTDx.                                                          |
|                     | cuFFTDx fusion | 13_gemm_fft_performance           | Benchmarks GEMM and FFT fusion. Requires cuFFTDx.                                                                     |
| TRSM                | Block          | 17_trsm_block                     | Solves triangular systems cooperatively in a CUDA block using shared memory.                                          |
|                     | Thread         | 17_trsm_thread                    | Solves many small triangular systems independently, one per CUDA thread.                                              |
|                     | Tensor views   | 18_trsm_conj_transpose            | Uses conj_transpose_view for TRSM without copying or transposing matrix data.                                         |
| Emulation           | Ozaki          | 16_dgemm_emulation                | Emulates double-precision GEMM using lower-precision GEMM operations. Requires CUDA Toolkit 13.1 or newer.            |
|                     | Pipeline       | 20_dgemm_emulation_pipeline       | Runs pipelined double-precision emulation controlled by required mantissa bits. Requires CUDA Toolkit 13.1 or newer.  |
|                     | Pipeline       | 21_sgemm_emulation_pipeline       | Runs pipelined single-precision emulation using the same mantissa-bit mechanism. Requires CUDA Toolkit 13.1 or newer. |


## [cuFFTDx](cuFFTDx)

* [cuFFTDx download page](https://developer.nvidia.com/cufftdx-downloads)
* [cuFFTDx API documentation](https://docs.nvidia.com/cuda/cufftdx/index.html)

#### Examples


| Group                    | Subgroup            | Example                        | Description                                                                                            |
| ------------------------ | ------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------ |
| Introduction Examples    |                     | introduction_example           | cuFFTDx API introduction                                                                               |
| Simple FFT Examples      | Thread FFT Examples | simple_fft_thread              | Complex-to-complex thread FFT                                                                          |
|                          |                     | simple_fft_thread_fp16         | Complex-to-complex thread FFT half-precision                                                           |
|                          | Block FFT Examples  | simple_fft_block               | Complex-to-complex block FFT                                                                           |
|                          |                     | simple_fft_block_shared        | Complex-to-complex block FFT shared-memory API                                                         |
|                          |                     | simple_fft_block_std_complex   | Complex-to-complex block FFT with `cuda::std::complex` as data type                                    |
|                          |                     | simple_fft_block_half2         | Complex-to-complex block FFT with `__half2` as data type                                               |
|                          |                     | simple_fft_block_fp16          | Complex-to-complex block FFT half-precision                                                            |
|                          |                     | simple_fft_block_c2r           | Complex-to-real block FFT                                                                              |
|                          |                     | simple_fft_block_r2c           | Real-to-complex block FFT                                                                              |
|                          |                     | simple_fft_block_c2r_fp16      | Complex-to-real block FFT half-precision                                                               |
|                          |                     | simple_fft_block_r2c_fp16      | Real-to-complex block FFT half-precision                                                               |
|                          |                     | simple_fft_block_cub_io        | Complex-to-complex block FFT with `CUB` used for loading/storing data                                  |
|                          |                     | simple_fft_block_block_dim     | Complex-to-complex block FFT with custom `BlockDim` operator                                           |
| FFT Performance          |                     | block_fft_performance          | Benchmark for C2C block FFT                                                                            |
|                          |                     | block_fft_performance_many     | Benchmark for C2C/R2C/C2R block FFT                                                                    |
| NVRTC Examples           |                     | nvrtc_fft_thread               | Complex-to-complex thread FFT                                                                          |
|                          |                     | nvrtc_fft_block                | Complex-to-complex block FFT                                                                           |
|                          |                     | nvrtc_query_database_fft_block | Complex-to-complex block FFT using database queries to get required traits at runtime                  |
| 2D/3D FFT Examples       |                     | fft_2d                         | Example showing how to perform 2D FP32 C2C FFT with cuFFTDx                                            |
|                          |                     | fft_2d_single_kernel           | 2D FP32 FFT in a single kernel using Cooperative Groups kernel launch                                  |
|                          |                     | fft_2d_single_kernel_block_dim | 2D FP32 FFT in a single kernel using Cooperative Groups kernel launch with non square block dimensions |
|                          |                     | fft_2d_r2c_c2r                 | Example showing how to perform 2D FP32 R2C/C2R convolution with cuFFTDx                                |
|                          |                     | fft_3d                         | Example showing how to perform 3D FP32 C2C FFT with cuFFTDx                                            |
|                          |                     | fft_3d_box_single_block        | Small 3D FP32 FFT that fits into a single block, each dimension is different                           |
|                          |                     | fft_3d_cube_single_block       | Small 3D (equal dimensions) FP32 FFT that fits into a single block                                     |
| Convolution Examples     |                     | convolution                    | Simplified FFT convolution                                                                             |
|                          |                     | convolution_padded             | R2C-C2R FFT convolution with optimization and zero padding                                             |
|                          |                     | convolution_r2c_c2r            | Simplified R2C-C2R FFT convolution                                                                     |
|                          |                     | convolution_performance        | Benchmark for FFT convolution using cuFFTDx and cuFFT                                                  |
| 3D Convolution Examples  |                     | convolution_3d                 | cuFFTDx fused 3D convolution with preprocessing, filtering and postprocessing                          |
|                          |                     | convolution_3d_c2r             | cuFFTDx fused 3D C2R/R2C FFT convolution                                                               |
|                          |                     | convolution_3d_r2c             | cuFFTDx fused 3D R2C/C2R FFT convolution                                                               |
|                          |                     | convolution_3d_padded          | cuFFTDx fused 3D FFT convolution using zero padding                                                    |
|                          |                     | convolution_3d_padded_r2c      | cuFFTDx fused 3D R2C/C2R FFT convolution with zero padding                                             |
| Mixed Precision Examples |                     | mixed_precision_fft_1d         | Example showing how to use separate storage and compute precisions                                     |
|                          |                     | mixed_precision_fft_2d         | Mixed precision 2D FFT with benchmarking and accuracy comparison                                       |


## [cuSolverDx](cuSolverDx)

* [cuSolverDx download page](https://developer.nvidia.com/cusolverdx-downloads)
* [cuSolverDx API documentation](https://docs.nvidia.com/cuda/cusolverdx/index.html)

#### Examples

| Group                         | Example                           | Description                                                                                                          |
| ----------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Introduction Examples         | posv_batched_block                | Introduction example with Cholesky factorization and solve (block execution)                                         |
| Linear Solve Examples         | potrf_batched_thread              | Cholesky factorization (thread execution)                                                                            |
|                               | posv_batched_thread               | Cholesky factorization and solve (thread execution)                                                                  |
|                               | potrf_block                       | Cholesky factorization (block execution)                                                                             |
|                               | potrf_runtime_ld_block            | Cholesky factorization with runtime leading dimensions (block execution)                                             |
|                               | potrf_batched_cluster             | Batched Cholesky factorization with experimental cluster execution, Hopper+ (cluster execution)                      |
|                               | getrf_wo_pivot_block              | LU factorization without pivoting (block execution)                                                                  |
|                               | getrf_partial_pivot_block         | LU factorization with partial pivoting (block execution)                                                             |
|                               | gesv_batched_wo_pivot_thread      | Solves batched linear systems without pivoting (thread execution)                                                    |
|                               | gesv_batched_wo_pivot_block       | Solves batched linear systems without pivoting (block execution)                                                     |
|                               | gesv_batched_partial_pivot_thread | Solves batched linear systems with partial pivoting (thread execution)                                               |
|                               | gesv_batched_partial_pivot_block  | Solves batched linear systems with partial pivoting (block execution)                                                |
|                               | gtsv_batched_wo_pivot_thread      | Solves batched tridiagonal linear systems without pivoting (thread execution)                                        |
|                               | gtsv_batched_wo_pivot_block       | Solves batched tridiagonal linear systems without pivoting (block execution)                                         |
| Least Squares Examples        | gels_batched_thread               | Solves batched least squares problems (thread execution)                                                             |
|                               | gels_batched_block                | Solves batched least squares problems (block execution)                                                              |
| Orthogonal Factors Examples   | geqrf_batched_thread              | QR factorization for batched matrices (thread execution)                                                             |
|                               | geqrf_batched_block               | QR factorization for batched matrices (block execution)                                                              |
|                               | unmqr_batched_thread              | Multiplies matrix by Q from QR factorization (thread execution)                                                      |
|                               | unmqr_batched_block               | Multiplies matrix by Q from QR factorization (block execution)                                                       |
|                               | ungqr_batched_thread              | Generates orthogonal matrix Q from QR factorization (thread execution)                                               |
|                               | ungqr_batched_block               | Generates orthogonal matrix Q from QR factorization (block execution)                                                |
| Symmetric Eigenvalue Examples | heev_batched_thread               | Eigenvalues and eigenvectors of batched Hermitian matrices (thread execution)                                        |
|                               | heev_batched_block                | Eigenvalues and eigenvectors of batched Hermitian matrices (block execution)                                         |
|                               | htev_batched_thread               | Eigenvalues and eigenvectors of batched Hermitian tridiagonal matrices (thread execution)                            |
|                               | htev_batched_block                | Eigenvalues and eigenvectors of batched Hermitian tridiagonal matrices (block execution)                             |
|                               | hegst_batched_thread              | Reduces batched Hermitian-definite generalized eigenproblem to standard form (thread execution)                      |
|                               | hegst_batched_block               | Reduces batched Hermitian-definite generalized eigenproblem to standard form (block execution)                       |
|                               | hegv_batched_block                | Eigenvalues and eigenvectors of batched Hermitian-definite generalized eigenproblems (block execution)               |
| SVD Examples                  | gesvd_batched_thread              | Singular value decomposition for batched general matrices (thread execution)                                         |
|                               | gesvd_batched_block               | Singular value decomposition for batched general matrices (block execution)                                          |
|                               | bdsvd_batched_thread              | Singular value decomposition for batched bidiagonal matrices (thread execution)                                      |
|                               | bdsvd_batched_block               | Singular value decomposition for batched bidiagonal matrices (block execution)                                       |
| BLAS Examples                 | trsm_batched_block                | Batched triangular solve with multiple right-hand sides (block execution)                                            |
|                               | trsm_batched_thread               | Batched triangular solve with multiple right-hand sides (thread execution)                                           |
|                               | trsm_batched_thread_advanced      | Batched triangular solve with advanced options (thread execution)                                                    |
| NVRTC Examples                | nvrtc_potrs                       | cuSolverDx with NVRTC runtime compilation and nvJitLink runtime linking                                              |
| Performance Examples          | geqrf_batched_performance         | Performance analysis of batched QR factorization                                                                     |
| Advanced Examples             | blocked_potrf                     | Cholesky factorization using blocked left-looking algorithm for large matrices (requires cuBLASDx)                   |
|                               | cga_blocked_potrf_right_looking   | Blocked right-looking Cholesky with thread block clusters and distributed shared memory, Hopper+ (requires cuBLASDx) |
|                               | cga_tsqrhr                        | Batched tall-skinny QR with Householder reconstruction using thread block clusters, Hopper+ (requires cuBLASDx)      |
|                               | reg_least_squares                 | Regularized least squares solver (requires cuBLASDx)                                                                 |


## [cuRANDDx](cuRANDDx)

* [cuRANDDx download page](https://developer.nvidia.com/curanddx-downloads)
* [cuRANDDx API documentation](https://docs.nvidia.com/cuda/curanddx/index.html)

#### Examples

| Group                 | Example                             | Description                                                                                          |
| --------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Introduction Examples | philox_thread_api                   | Introduction example explaining the basics of cuRANDDx using the Philox generator                    |
| Thread API Examples   | pcg_thread_api                      | Use the PCG generator to create 32-bit random numbers matching NVPL RAND with strict ordering        |
|                       | philox_random_bits_thread_api       | Use the Philox generator to generate a sequence of random bits                                       |
|                       | xorwow_init_and_generate_thread_api | Set up generator states first, then generate random numbers in later kernels                         |
|                       | mrg_two_distributions_thread_api    | Generate two sequences of different distributions in a single kernel using skip functions (MRG32k3a) |
|                       | sobol_thread_api                    | Generate quasirandom numbers using the 64-bit scrambled Sobol generator                              |
| NVRTC Examples        | nvrtc_pcg_thread_api                | Use cuRANDDx with NVRTC runtime compilation                                                          |


## [nvCOMPDx](nvCOMPDx)

* [nvCOMPDx download page](https://developer.nvidia.com/nvcompdx-downloads)
* [nvCOMPDx API documentation](https://docs.nvidia.com/cuda/nvcompdx/index.html)

#### Examples

| Group                 | Example                                 | Description                                                              |
| --------------------- | --------------------------------------- | ------------------------------------------------------------------------ |
| Introduction Examples | lz4_gpu_compression_introduction        | Introductory example showcasing GPU LZ4 compression                      |
| LZ4 GPU               | lz4_gpu_compression_decompression       | Warp-level GPU LZ4 compression and decompression                         |
| LZ4 GPU and CPU       | lz4_cpu_compression_gpu_decompression   | CPU compression with Warp-level GPU LZ4 decompression                    |
|                       | lz4_gpu_compression_cpu_decompression   | Warp-level GPU LZ4 compression with CPU decompression                    |
| ANS GPU               | ans_gpu_compression_decompression       | Block-level GPU ANS compression and decompression                        |
|                       | ans_gpu_decompression_reduction         | Fused block-level GPU ANS decompression followed by block-wide reduction |
| NVRTC Examples        | lz4_cpu_compression_nvrtc_decompression | CPU compression, warp-level GPU NVRTC + nvJitLink LZ4 decompression      |
