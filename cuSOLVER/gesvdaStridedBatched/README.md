# cuSOLVER Approximate Singular Value Decomposition example

## Description

This code demonstrates a usage of cuSOLVER gesvdaStridedBatched function to approximate singular value decomposition by gesvdaStridedBatched.

_**A** = **U** * **&Sigma;** * **V**<sup>H</sup>_

A0 and A1 are a 3x2 dense matrices,
```
A0 = | 1.0 | 2.0 |
     | 4.0 | 5.0 |
     | 2.0 | 1.0 |

A1 = | 10.0 | 9.0 |
     |  8.0 | 7.0 |
     |  6.0 | 5.0 |
```

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
- [cusolverDnSgesvdaStridedBatched_bufferSize  API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-gesvda)
- [cusolverDnSgesvdaStridedBatched API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-gesvda)

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- [CMake](https://cmake.org/download) version 3.18 minimum
- Minimum [CUDA 10.1 toolkit](https://developer.nvidia.com/cuda-downloads) is required.

## Build command on Linux
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
Make sure that CMake finds expected CUDA Toolkit. If that is not the case you can add argument `-DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc` to cmake command.

## Build command on Windows
```
$ mkdir build
$ cd build
$ cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
$ Open cusolver_examples.sln project in Visual Studio and build
```

# Usage
```
$  ./cusolver_gesvdaStridedBatched_example
```

Sample example output:

```
A0 = (matlab base-1)
1.00 2.00
4.00 5.00
2.00 1.00
=====
A1 = (matlab base-1)
10.00 9.00
8.00 7.00
6.00 5.00
=====
0-th matrix, gesvda converges
1-th matrix, gesvda converges
S0 = (matlab base-1)
7.07
1.04
=====
U0 = (matlab base-1)
0.31 -0.49
0.91 -0.11
0.29 0.87
=====
V) = (matlab base-1)
0.64 0.77
0.77 -0.64
=====
|S0 - S0_exact|_sup = 0.000000E+00
residual |A0 - U0*S0*V0**H|_F = 4.768372E-07
S1 = (matlab base-1)
18.84
0.26
=====
U1 = (matlab base-1)
0.71 0.57
0.56 -0.12
0.41 -0.81
=====
V1 = (matlab base-1)
0.75 -0.66
0.66 0.75
=====
|S1 - S1_exact|_sup = 0.000000E+00
residual |A1 - U1*S1*V1**H|_F = 0.000000E+00
```
