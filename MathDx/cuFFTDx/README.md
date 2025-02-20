# cuFFTDx Library - API Examples

All example, including more advanced onces, are shipped within [cuFFTDx package](https://developer.nvidia.com/cufftdx-downloads).

## Description

This folder demonstrates cuFFTDx APIs usage.

* [cuFFTDx download page](https://developer.nvidia.com/cufftdx-downloads)
* [cuFFTDx API documentation](https://docs.nvidia.com/cuda/cufftdx/index.html)

## Requirements

* [cuFFTDx/MathDx package](https://developer.nvidia.com/cufftdx-downloads)
* [See cuFFTDx requirements](https://docs.nvidia.com/cuda/cufftdx/requirements_func.html)
* CMake 3.18 or newer
* Linux system with installed NVIDIA drivers
* NVIDIA GPU of Volta (SM70) or newer architecture

## Build

* You may specify `CUFFTDX_CUDA_ARCHITECTURES` to limit CUDA architectures used for compilation (see [CMake:CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES))
* `mathdx_ROOT` - path to mathDx package (XX.Y - version of the package)

```
mkdir build && cd build
cmake -DCUFFTDX_CUDA_ARCHITECTURES=70-real -Dmathdx_ROOT=/opt/nvidia/mathdx/XX.Y ..
make
// Run
ctest
```

## Examples

For the detailed descriptions of the examples please visit [Examples](https://docs.nvidia.com/cuda/cufftdx/examples.html) section of the cuFFTDx documentation.

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