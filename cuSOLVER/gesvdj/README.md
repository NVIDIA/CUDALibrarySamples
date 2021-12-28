# cuSOLVER Singular Value Decomposition with singular vector (via Jacobi method) example

## Description

This code demonstrates a usage of cuSOLVER gesvdj function to perform singular value decomposition

_**A** = **U** * **&Sigma;** * **V**<sup>H</sup>_

A is a 3x2 dense matrices,
```
A = | 1.0 | 2.0 |
    | 4.0 | 5.0 |
    | 2.0 | 1.0 |
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
- [cusolverDnDgesvdj_bufferSize  API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-gesvdj)
- [cusolverDnDgesvdj API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-gesvdj)

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
$  ./cusolver_gesvdj_example
```

Sample example output:

```
m = 3, n = 2
tol = 1.000000E-07, default value is machine zero
max. sweeps = 15, default value is 100
econ = 0
A = (matlab base-1)
1.00 2.00
4.00 5.00
2.00 1.00
=====
gesvdj converges
S = singular values (matlab base-1)
7.07
1.04
=====
U = left singular vectors (matlab base-1)
0.31 -0.49 0.82
0.91 -0.11 -0.41
0.29 0.87 0.41
=====
V = right singular vectors (matlab base-1)
0.64 0.77
0.77 -0.64
=====
|S - S_exact|_sup = 4.440892E-16
residual |A - U*S*V**H|_F = 3.511066E-16
number of executed sweeps = 1
```
