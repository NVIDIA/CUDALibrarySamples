# cuSOLVER Approximate Singular Value Decomposition example

## Description

This code demonstrates a usage of cuSOLVER gesvdjBatched function to compute the SVD of a sequence of dense matrices

_**A**<sub>j</sub> = **U**<sub>j</sub> * **&Sigma;**<sub>j</sub> * **V**<sub>j</sub><sup>H</sup>_

A0 and A1 are a 3x2 dense matrices,
```
A0 = |  1.0 | -1.0 |
     | -1.0 |  2.0 |
     |  0.0 |  0.0 |

A1 = | 3.0 | 4.0 |
     | 4.0 | 7.0 |
     | 0.0 | 0.0 |
```

The following code uses gesvdjBatched to compute singular values and singular vectors.

The user can disable/enable sorting by the function cusolverDnXgesvdjSetSortEig.

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
- [cusolverDnDgesvdjBatched_bufferSize  API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-gesvdjbatch)
- [cusolverDnDgesvdjBatched API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-gesvdjbatch)

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
$  ./cusolver_gesvdjBatched_example
```

Sample example output:

```
m = 3, n = 2
tol = 1.000000E-07, default value is machine zero
max. sweeps = 15, default value is 100
A0 = (matlab base-1)
1.00 -1.00
-1.00 2.00
0.00 0.00
=====
A1 = (matlab base-1)
3.00 4.00
4.00 7.00
0.00 0.00
=====
matrix 0: gesvdj converges
matrix 1: gesvdj converges
====
S0(1) = 3.8196601125010510E-01
S0(2) = 2.6180339887498945E+00
====
S1(1) = 5.2786404500042117E-01
S1(2) = 9.4721359549995796E+00
====
U0 = (matlab base-1)
0.85 -0.53 0.00
0.53 0.85 0.00
-0.00 0.00 1.00
U1 = (matlab base-1)
0.85 0.53 0.00
-0.53 0.85 0.00
0.00 -0.00 1.00
V0 = (matlab base-1)
0.85 -0.53
0.53 0.85
V1 = (matlab base-1)
0.85 0.53
-0.53 0.85
```
