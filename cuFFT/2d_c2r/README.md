# cuFFT 2D FFT C2R example

## Description

In this example a two-dimensional complex-to-real transform is applied to the input data arranged according to the requirements of the default FFTW padding mode.

## Supported SM Architectures

All GPUs supported by CUDA Toolkit (https://developer.nvidia.com/cuda-gpus)  

## Supported OSes

Linux  
Windows

## Supported CPU Architecture

x86_64  
ppc64le  
arm64-sbsa

## CUDA APIs involved
- [cufftExecC2R API](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftexecr2c-cufftexecd2z)

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- [CMake](https://cmake.org/download) version 3.18 minimum

## Build command on Linux
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
Make sure that CMake finds expected CUDA Toolkit. If that is not the case you can add argument `-DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc` to cmake command.

# Usage 1
```
$  ./bin/2d_c2r_example
```

Sample example output (batch_size=1):

```
Input array:
0.000000 + 0.000000j
1.000000 + -1.000000j
2.000000 + -2.000000j
3.000000 + -3.000000j
=====
Output array:
6.000000
-2.000000
-4.000000
0.000000
=====
```

Sample example output (batch_size=2):

```
Input array:
0.000000 + 0.000000j
1.000000 + -1.000000j
2.000000 + -2.000000j
3.000000 + -3.000000j
4.000000 + -4.000000j
5.000000 + -5.000000j
6.000000 + -6.000000j
7.000000 + -7.000000j
=====
Output array:
6.000000
-2.000000
-4.000000
0.000000
22.000000
-2.000000
-4.000000
0.000000
=====
```