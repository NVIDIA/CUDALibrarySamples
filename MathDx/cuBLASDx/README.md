# cuBLASDx Library - API Examples

All example, including more advanced onces, are shipped within [cuBLASDx package](https://developer.nvidia.com/cublasdx-downloads).

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

| Group                 | Subgroup       | Example                        | Description                                                           |
|-----------------------|----------------|--------------------------------|-----------------------------------------------------------------------|
| Introduction Examples |                | introduction_example           | cuBLASDx API introduction example                                     |
|                       |                |                                |                                                                       |
| Simple GEMM Examples  | Basic Example  | simple_gemm_fp32               | Performs fp32 GEMM                                                    |
|                       |                | simple_gemm_cfp16              | Performs complex fp16 GEMM                                            |
|                       |                |                                |                                                                       |
|                       | Extra Examples | simple_gemm_leading_dimensions | Performs GEMM with non-default leading dimensions                     |
|                       |                | simple_gemm_std_complex_fp32   | Performs GEMM with cuda::std::complex as data type                    |
|                       |                |                                |                                                                       |
| NVRTC Examples        |                | nvrtc_gemm                     | Performs GEMM, kernel is compiled using NVRTC                         |
|                       |                |                                |                                                                       |
| GEMM Performance      |                | single_gemm_performance        | Benchmark for single GEMM                                             |
|                       |                | fused_gemm_performance         | Benchmark for 2 GEMMs fused into a single kernel                      |
|                       |                |                                |                                                                       |
| Advanced Examples     | Fusion         | fused_gemm                     | Performs 2 GEMMs in a single kernel                                   |
|                       |                | gemm_fft                       | Perform GEMM and FFT in a single kernel                               |
|                       |                | gemm_fft_fp16                  | Perform GEMM and FFT in a single kernel (half-precision complex type) |
|                       |                | gemm_fft_performance           | Benchmark for GEMM and FFT fused into a single kernel                 |
|                       |                |                                |                                                                       |
|                       | Deep Learning  | scaled_dot_prod_attn           | Scaled dot product attention using cuBLASDx                           |
|                       |                | scaled_dot_prod_attn_batched   | Multi-head attention using cuBLASDx                                   |
|                       |                |                                |                                                                       |
|                       | Other          | multiblock_gemm                | Proof-of-concept for single large GEMM using multiple CUDA blocks     |
|                       |                | batched_gemm_fp64              | Manual batching in a single CUDA block                                |
|                       |                | blockdim_gemm_fp16             | BLAS execution with different block dimensions                        |
