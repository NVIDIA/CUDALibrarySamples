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
// Run
ctest
```

## Examples

For the detailed descriptions of the examples please visit [Examples](https://docs.nvidia.com/cuda/cublasdx/examples.html) section of the cuBLASDx documentation.

| Group                 | Subgroup       | Example                           | Description                                                                    |
|-----------------------|----------------|-----------------------------------|--------------------------------------------------------------------------------|
| Introduction Examples |                | 01_introduction_example           | cuBLASDx API introduction example                                              |
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