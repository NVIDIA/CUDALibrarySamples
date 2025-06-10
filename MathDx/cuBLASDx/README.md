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

| Group                 | Subgroup       | Example                        | Description                                                                    |
|-----------------------|----------------|--------------------------------|--------------------------------------------------------------------------------|
| Introduction Examples |                | introduction_example           | cuBLASDx API introduction example                                              |
| Simple GEMM Examples  | Basic Example  | simple_gemm_fp32               | Performs fp32 GEMM                                                             |
|                       |                | simple_gemm_int8_int8_int32    | Performs integral GEMM using Tensor Cores                                      |
|                       |                | simple_gemm_cfp16              | Performs complex fp16 GEMM                                                     |
|                       |                | simple_gemm_fp8                | Performs fp8 GEMM                                                              |
|                       | Extra Examples | simple_gemm_leading_dimensions | Performs GEMM with non-default leading dimensions                              |
|                       |                | simple_gemm_fp32_decoupled     | Performs fp32 GEMM using 16-bit input type to save on storage and transfers    |
|                       |                | simple_gemm_std_complex_fp32   | Performs GEMM with cuda::std::complex as data type                             |
|                       |                | simple_gemm_mixed_precision    | Performs a mixed precision GEMM                                                |
|                       |                | simple_gemm_transform          | Performs GEMM with custom load and store operators                             |
|                       |                | simple_gemm_custom_layout      | Performs GEMM with a custom user provided CuTe layout                          |
|                       |                | simple_gemm_aat                | Performs GEMM where C = A * A^T                                                |
| NVRTC Examples        |                | nvrtc_gemm                     | Performs GEMM, kernel is compiled using NVRTC                                  |
| GEMM Performance      |                | single_gemm_performance        | Benchmark for single GEMM                                                      |
|                       |                | fused_gemm_performance         | Benchmark for 2 GEMMs fused into a single kernel                               |
|                       |                | device_gemm_performance        | Benchmark entire device GEMMs using cuBLASDx for single tile                   |
|                       |                | gemm_device_partial_sums       | Use partial accumulations in higher precision to minimize big-K rounding error |
| Advanced Examples     | Fusion         | fused_gemm                     | Performs 2 GEMMs in a single kernel                                            |
|                       |                | gemm_fft                       | Perform GEMM and FFT in a single kernel                                        |
|                       |                | gemm_fft_fp16                  | Perform GEMM and FFT in a single kernel (half-precision complex type)          |
|                       |                | gemm_fft_performance           | Benchmark for GEMM and FFT fused into a single kernel                          |
|                       | Other          | batched_gemm_fp64              | Manual batching in a single CUDA block                                         |
|                       |                | blockdim_gemm_fp16             | BLAS execution with different block dimensions                                 |
|                       | Emulation      | dgemm_emulation                | Emulate double precision GEMM using lower precision operations (Ozaki scheme)  |