# cuBLASDx Library - API Examples

All examples, including more advanced ones, are shipped within [cuBLASDx package](https://developer.nvidia.com/cublasdx-downloads).

## Description

This folder demonstrates cuBLASDx APIs usage.

* [cuBLASDx download page](https://developer.nvidia.com/cublasdx-downloads)
* [cuBLASDx API documentation](https://docs.nvidia.com/cuda/cublasdx/index.html)

## Requirements

* [cuBLASDx/MathDx package](https://developer.nvidia.com/cublasdx-downloads)
* [See cuBLASDx requirements](https://docs.nvidia.com/cuda/cublasdx/requirements_func.html)
* CMake 3.18 or newer
* Linux system with installed NVIDIA drivers
* NVIDIA GPU of Volta (SM70) or newer architecture

## Build

* You may specify `CUBLASDX_CUDA_ARCHITECTURES` to limit CUDA architectures used for compilation (see [CMake:CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES))
* `mathdx_ROOT` - path to mathDx package (XX.Y - version of the package)

```
mkdir build && cd build
cmake -DCUBLASDX_CUDA_ARCHITECTURES=70-real -Dmathdx_ROOT=/opt/nvidia/mathdx/XX.Y ..
make
# Run
ctest
```

## Examples

For the detailed descriptions of the examples please visit [Examples](https://docs.nvidia.com/cuda/cublasdx/examples.html) section of the cuBLASDx documentation.

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
