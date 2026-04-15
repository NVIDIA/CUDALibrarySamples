# cuFFTDx + cuFFT LTO Library - API Examples


## Description

This folder demonstrates cuFFTDx API usage and provides examples showing how to integrate LTO features into existing cuFFTDx projects.

* [cuFFTDx + cuFFT LTO documentation](https://docs.nvidia.com/cuda/cufftdx/cufft_lto/index.html)

## Requirements

* [cuFFTDx package](https://developer.nvidia.com/cufftdx-downloads)
* [See cuFFTDx + cuFFT LTO requirements](https://docs.nvidia.com/cuda/cufftdx/requirements_func.html#requirements-for-cufftdx-with-cufft-lto)
* CUDA Toolkit 13.1 or newer
* CMake 3.26 or newer
* Linux system with installed NVIDIA drivers
* NVIDIA GPU of Turing (SM75) or newer architecture

## Build

* You may specify `CUFFTDX_CUDA_ARCHITECTURES` to limit CUDA architectures used for compilation (see [CMake:CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES))
* `mathdx_ROOT` - path to the mathDx installation directory
* `cufft_ROOT` - path to `cufft` directory. If not specified, the version of cuFFT used will be the one included in your default CUDA Toolkit.

```
mkdir build && cd build
cmake -DCUFFTDX_EXAMPLES_LTO=ON \
      -DCUFFTDX_CUDA_ARCHITECTURES=80-real \
      -Dmathdx_ROOT=/opt/nvidia/mathdx/XX.Y  \
      -Dcufft_ROOT=<your_cufft_installation> ..
make
# Run
ctest
```

## LTO Examples

For the detailed descriptions of the examples please visit [Examples](https://docs.nvidia.com/cuda/cufftdx/lto_examples.html) section of the cuFFTDx documentation.

| Group                    | Subgroup                    | Example                       | Description                                            |
|--------------------------|-----------------------------|-------------------------------|--------------------------------------------------------|
| Introduction Examples    | 09_introduction_lto_example | introduction_lto_example      | (offline) cuFFTDx LTO introduction                     |
|                          | 04_nvrtc_fft                | nvrtc_fft_block_lto           | (online) cuFFTDx LTO introduction                      |
|                          | 10_cufft_device_api_example | cufft_device_api_example.     | (offline) cuFFT Device API introduction                |
| Simple FFT Examples      | 01_simple_fft_thread        | simple_fft_thread_lto         | (offline) Complex-to-complex (C2C) thread FFT using LTO|
|                          | 02_simple_fft_block.        | simple_fft_block_c2r_lto      | (offline) Complex-to-real block FFT using LTO          |
| NVRTC Examples           | 04_nvrtc_fft                | nvrtc_fft_thread_lto          | (online) Complex-to-complex thread FFT using LTO       |
| FFT Performance          | 03_block_fft_performance    | block_fft_lto_ptx_performance | (offline) Benchmark for C2C block FFT (LTO vs PTX)     |
