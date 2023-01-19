# cuFFTDx 2D FFT

## Description

Three examples `fft_2d`, `fft_2d_single_kernel`, `fft_2d_r2c_c2r` demonstrate how to use cuFFTDx block level 1D FFT
to perform 2D FFTs.

* `fft_2d` performs 2D FFT in two kernels.
* `fft_2d_single_kernel` shows how to perform 2D FFT in a single kernel using a cooperative launch and grid synchronization.
* `fft_2d_r2c_c2r` executes R2C->C2R 2D FFTs.

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
