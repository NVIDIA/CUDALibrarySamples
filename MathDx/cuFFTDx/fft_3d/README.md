# cuFFTDx 3D FFT

## Description

Two examples which demonstrate how to perform small 3D FFTs using cuFFTDx thread level 1D FFTs.

## Requirements

* CMake 3.18 or newer
* MathDx package (see requirements of mathDx libraries for more details)
* CUDA Toolkit 11.0 or newer
* Linux system with installed NVIDIA drivers
* NVIDIA GPU of Volta (SM70) or newer architecture

## Build

```
mkdir build && cd build
# You may specify CMAKE_CUDA_ARCHITECTURES to limit CUDA architectures used for compilation
# mathdx_ROOT - path to mathDx package (XX.Y - version of the package)
cmake -Dmathdx_ROOT=/opt/nvidia/mathdx/XX.Y ..
make
```

## Run

```
./fft_2d
./fft_2d_single_kernel
./fft_2d_r2c_c2r
```
