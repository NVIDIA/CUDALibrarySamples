# cuFFTDx + cuFFT LTO EA Library - API Examples

All examples are shipped within [cuFFTDx + cuFFT LTO EA package](https://developer.nvidia.com/cufftea).

## Description

This folder demonstrates cuFFTDx API usage and provides examples showing how to integrate LTO features into existing cuFFTDx projects.

* [cuFFTDx + cuFFT LTO EA download page](https://developer.nvidia.com/cufftea)
* [cuFFTDx + cuFFT LTO EA documentation](https://docs.nvidia.com/cuda/cufftdx/1.4.0-ea/index.html)

## Requirements

* [cuFFTDx + cuFFT LTO EA package](https://developer.nvidia.com/cufftea)
* [See cuFFTDx + cuFFT LTO EA requirements](https://docs.nvidia.com/cuda/cufftdx/1.4.0-ea/requirements_func.html)
* CUDA Toolkit 12.8 or newer
* CMake 3.26 or newer
* Linux system with installed NVIDIA drivers
* NVIDIA GPU of Volta (SM70) or newer architecture

## Build

* You may specify `CUFFTDX_CUDA_ARCHITECTURES` to limit CUDA architectures used for compilation (see [CMake:CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES))
* `cufftdx_ROOT` - path to `cufftdx` directory in the cuFFTDx + cuFFT LTO EA package
* `cufft_ROOT` - path to `cufftd` directory in the cuFFTDx + cuFFT LTO EA package

```
mkdir build && cd build
cmake -DCUFFTDX_CUDA_ARCHITECTURES=70-real
      -Dcufftdx_ROOT=/opt/nvidia-cufft-11.5.0-cufftdx-1.4.0-<x86_64,aarch64>
      -Dcufft_ROOT=/opt/nvidia-cufft-11.5.0-cufftdx-1.4.0-<x86_64,aarch64> ..
make
// Run
ctest
```

## LTO Examples

For the detailed descriptions of the examples please visit [Examples](https://docs.nvidia.com/cuda/cufftdx/1.4.0-ea/examples.html) section of the cuFFTDx + cuFFT LTO EA documentation.

<table>
    <thead>
        <tr>
            <th colspan="2">Group</th>
            <th rowspan="2">Example</th>
            <th rowspan="2">Description</th>
        </tr>
        <tr>
            <th></th>
            <th>Subgroup</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="3">Introduction Examples</td>
            <td>09_introduction_lto_example</td>
            <td>00_introduction_lto_example</td>
            <td>(offline) cuFFTDx LTO introduction</td>
        </tr>
        <tr>
            <td>04_nvrtc_fft</td>
            <td>03_nvrtc_fft_block_lto</td>
            <td>(online) cuFFTDx LTO introduction</td>
        </tr>
        <tr>
            <td>10_cufft_device_api_example</td>
            <td>00_cufft_device_api_lto_example</td>
            <td>(offline) cuFFT Device API introduction</td>
        </tr>
        <tr>
            <td rowspan="2">Simple FFT Examples</td>
            <td>01_simple_fft_thread</td>
            <td>02_simple_fft_thread_lto</td>
            <td>(offline) Complex-to-complex (C2C) thread FFT using LTO</td>
        </tr>
        <tr>
            <td>02_simple_fft_block_lto</td>
            <td>10_simple_fft_block_c2r_lto</td>
            <td>(offline) Complex-to-real block FFT using LTO</td>
        </tr>
        <tr>
            <td>NVRTC Examples (additional)</td>
            <td>04_nvrtc_fft</td>
            <td>02_nvrtc_fft_thread_lto</td>
            <td>(online) Complex-to-complex thread FFT using LTO</td>
        </tr>
        <tr>
            <td>FFT Performance</td>
            <td>03_block_fft_performance</td>
            <td>02_block_fft_lto_ptx_performance</td>
            <td>(offline) Benchmark for C2C block FFT (LTO vs PTX)</td>
        </tr>
    </tbody>
</table>
