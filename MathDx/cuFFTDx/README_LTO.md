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
* NVIDIA GPU of Volta (SM70) or newer architecture

## Build

* You may specify `CUFFTDX_CUDA_ARCHITECTURES` to limit CUDA architectures used for compilation (see [CMake:CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES))
* `cufftdx_ROOT` - path to `cufftdx` directory
* `cufft_ROOT` - path to `cufft` directory. If not specified, teh version of cuFFT used will be the one included in your default CUDA Toolkit.

```
mkdir build && cd build
cmake -DCUFFTDX_CUDA_ARCHITECTURES=70-real
      -Dmathdx_ROOT="<your_directory>/nvidia/mathdx/yy.mm/"
      -Dcufft_ROOT="<your_cufft_installation>" ..
make
// Run
ctest
```

## LTO Examples

For the detailed descriptions of the examples please visit [Examples](https://docs.nvidia.com/cuda/cufftdx/lto_examples.html) section of the cuFFTDx documentation.

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
            <td>introduction_lto_example</td>
            <td>(offline) cuFFTDx LTO introduction</td>
        </tr>
        <tr>
            <td>04_nvrtc_fft</td>
            <td>nvrtc_fft_block_lto</td>
            <td>(online) cuFFTDx LTO introduction</td>
        </tr>
        <tr>
            <td>10_cufft_device_api_example</td>
            <td>cufft_device_api_lto_example</td>
            <td>(offline) cuFFT Device API introduction</td>
        </tr>
        <tr>
            <td rowspan="2">Simple FFT Examples</td>
            <td>01_simple_fft_thread</td>
            <td>simple_fft_thread_lto</td>
            <td>(offline) Complex-to-complex (C2C) thread FFT using LTO</td>
        </tr>
        <tr>
            <td>02_simple_fft_block_lto</td>
            <td>simple_fft_block_c2r_lto</td>
            <td>(offline) Complex-to-real block FFT using LTO</td>
        </tr>
        <tr>
            <td>NVRTC Examples (additional)</td>
            <td>04_nvrtc_fft</td>
            <td>nvrtc_fft_thread_lto</td>
            <td>(online) Complex-to-complex thread FFT using LTO</td>
        </tr>
        <tr>
            <td>FFT Performance</td>
            <td>03_block_fft_performance</td>
            <td>block_fft_lto_ptx_performance</td>
            <td>(offline) Benchmark for C2C block FFT (LTO vs PTX)</td>
        </tr>
    </tbody>
</table>
