# cuFFTDx Library - API Examples

> **_NOTE:_** For information about cuFFTDx + cuFFT LTO, please visit [cuFFTDx + cuFFT LTO EA Library - API Examples](README_LTO.md).

All example, including more advanced onces, are shipped within [cuFFTDx package](https://developer.nvidia.com/cufftdx-downloads).

## Description

This folder demonstrates cuFFTDx APIs usage.

* [cuFFTDx download page](https://developer.nvidia.com/cufftdx-downloads)
* [cuFFTDx API documentation](https://docs.nvidia.com/cuda/cufftdx/index.html)

## Requirements

* [cuFFTDx/MathDx package](https://developer.nvidia.com/cufftdx-downloads)
* [See cuFFTDx requirements](https://docs.nvidia.com/cuda/cufftdx/requirements_func.html)
* CMake 3.26 or newer
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
            <td colspan="2">00_introduction_example</td>
            <td>introduction_example</td>
            <td>cuFFTDx API introduction</td>
        </tr>
        <tr>
            <td rowspan="12">Simple FFT Examples</td>
            <td rowspan="2">01_simple_fft_thread</td>
            <td>simple_fft_thread</td>
            <td>Complex-to-complex thread FFT</td>
        </tr>
        <tr>
            <td>simple_fft_thread_fp16</td>
            <td>Complex-to-complex thread FFT half-precision</td>
        </tr>
        <tr>
            <td rowspan="10">02_simple_fft_block</td>
            <td>simple_fft_block</td>
            <td>Complex-to-complex block FFT</td>
        </tr>
        <tr>
            <td>simple_fft_block_shared</td>
            <td>Complex-to-complex block FFT shared-memory API</td>
        </tr>
        <tr>
            <td>simple_fft_block_std_complex</td>
            <td>Complex-to-complex block FFT with <code>cuda::std::complex</code> as data type</td>
        </tr>
        <tr>
            <td>simple_fft_block_half2</td>
            <td>Complex-to-complex block FFT with <code>__half2</code> as data type</td>
        </tr>
        <tr>
            <td>simple_fft_block_fp16</td>
            <td>Complex-to-complex block FFT half-precision</td>
        </tr>
        <tr>
            <td>simple_fft_block_c2r</td>
            <td>Complex-to-real block FFT</td>
        </tr>
        <tr>
            <td>simple_fft_block_r2c</td>
            <td>Real-to-complex block FFT</td>
        </tr>
        <tr>
            <td>simple_fft_block_c2r_fp16</td>
            <td>Complex-to-real block FFT half-precision</td>
        </tr>
        <tr>
            <td>simple_fft_block_r2c_fp16</td>
            <td>Real-to-complex block FFT half-precision</td>
        </tr>
                <tr>
            <td>simple_fft_block_block_dim</td>
            <td>Complex-to-complex block FFT with BlockDim operator</td>
        </tr>
        <tr>
            <td colspan="2" rowspan="2">03_block_fft_performance</td>
            <td>block_fft_performance</td>
            <td>Benchmark for C2C block FFT</td>
        </tr>
        <tr>
            <td>block_fft_performance_many</td>
            <td>Benchmark for C2C/R2C/C2R block FFT</td>
        </tr>
        <tr>
            <td colspan="2" rowspan="2">04_nvrtc_fft</td>
            <td>nvrtc_fft_thread</td>
            <td>Complex-to-complex thread FFT</td>
        </tr>
        <tr>
            <td>nvrtc_fft_block</td>
            <td>Complex-to-complex block FFT</td>
        </tr>
        <tr>
            <td colspan="2" rowspan="7">05_fft_Xd</td>
            <td>fft_2d</td>
            <td>Example showing how to perform 2D FP32 C2C FFT with cuFFTDx</td>
        </tr>
        <tr>
            <td>fft_2d_single_kernel</td>
            <td>2D FP32 FFT in a single kernel using Cooperative Groups kernel launch</td>
        </tr>
        <tr>
            <td>fft_2d_r2c_c2r</td>
            <td>Example showing how to perform 2D FP32 R2C/C2R convolution with cuFFTDx</td>
        </tr>
        <tr>
            <td>fft_3d</td>
            <td>Example showing how to perform 3D FP32 C2C FFT with cuFFTDx</td>
        </tr>
        <tr>
            <td>fft_3d_box_single_block</td>
            <td>Small 3D FP32 FFT that fits into a single block, each dimension is different</td>
        </tr>
        <tr>
            <td>fft_3d_cube_single_block</td>
            <td>Small 3D (equal dimensions) FP32 FFT that fits into a single block</td>
        </tr>
        <tr>
            <td>fft_2d_single_kernel_block_dim</td>
            <td>2D FP32 FFT in a single kernel using Cooperative Groups kernel launch and dimensions with different ept</td>
        </tr>
        <tr>
            <td colspan="2" rowspan="4">06_convolution</td>
            <td>convolution</td>
            <td>Simplified FFT convolution</td>
        </tr>
        <tr>
            <td>convolution_padded</td>
            <td>R2C-C2R FFT convolution with optimization and zero padding</td>
        </tr>
        <tr>
            <td>convolution_r2c_c2r</td>
            <td>Simplified R2C-C2R FFT convolution</td>
        </tr>
        <tr>
            <td>convolution_performance</td>
            <td>Benchmark for FFT convolution using cuFFTDx and cuFFT</td>
        </tr>
        <tr>
            <td colspan="2" rowspan="5">07_convolution_3d</td>
            <td>convolution_3d</td>
            <td>cuFFTDx fused 3D convolution with preprocessing, filtering and postprocessing</td>
        </tr>
        <tr>
            <td>convolution_3d_c2r</td>
            <td>cuFFTDx fused 3D C2R/R2C FFT convolution</td>
        </tr>
        <tr>
            <td>convolution_3d_r2c</td>
            <td>cuFFTDx fused 3D R2C/C2R FFT convolution</td>
        </tr>
        <tr>
            <td>convolution_3d_padded</td>
            <td>cuFFTDx fused 3D FFT convolution using zero padding</td>
        </tr>
        <tr>
            <td>convolution_3d_padded_r2c</td>
            <td>cuFFTDx fused 3D R2C/C2R FFT convolution with zero padding</td>
        </tr>
        <tr>
            <td colspan="2" rowspan="2">08_mixed_precision</td>
            <td>mixed_precision_fft_1d</td>
            <td>Example showing how to use separate storage and compute precisions</td>
        </tr>
        <tr>
            <td>mixed_precision_fft_2d</td>
            <td>Mixed precision 2D FFT with benchmarking and accuracy comparison</td>
        </tr>
    </tbody>
</table>
