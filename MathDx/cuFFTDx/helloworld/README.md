# cuFFTDx "Hello World"

## Description

This code is a very simple example how to perform a 1024-point complex-to-complex single-precision FFT in a CUDA block using cuFFTDx.

## Requirements

* MathDx package (see requirements of mathDx libraries for more details)
* CUDA Toolkit 11.0 or newer
* Linux system with installed NVIDIA drivers
* NVIDIA GPU of Volta (SM70) or newer architecture

## Build

```
mkdir build && cd build
# Change "--generate-code arch=compute_80,code=sm_80" and CUDA_ARCH to build for other supported CUDA architectures.
# /opt/nvidia/mathdx/XX.Y/include - path to mathDx package (XX.Y - version of the package)
nvcc -std=c++17 -O3 --generate-code arch=compute_80,code=sm_80 -DCUDA_ARCH=80 -I /opt/nvidia/mathdx/XX.Y/include -lcuda ../helloworld.cu -o helloworld
```

## Run

```
./helloworld
```
