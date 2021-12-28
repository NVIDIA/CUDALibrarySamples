# cuSOLVER Standard Symmetric Dense Eigenvalue solver (via Jacobi method) example

## Description

This code demonstrates a usage of cuSOLVER syevj  function for using syevj  to compute spectrum of a pair of dense symmetric matrices (A,B) by

_**A**x = &lambda;x_

where A is a 3x3 dense symmetric matrix
```
A = | 3.5 | 0.5 | 0.0 |
    | 0.5 | 3.5 | 0.0 |
    | 0.0 | 0.0 | 2.0 |
```

The following code uses syevj to compute eigenvalues and eigenvectors, then compare to exact eigenvalues {2,3,4}.

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
- [cusolverDnDsyevj_bufferSize API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-syevj)
- [cusolverDnDsyevj API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-syevj)
- [cusolverDnXsyevjSetTolerance API](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverDnXgesvdjSetTolerance)
- [cusolverDnXsyevjSetMaxSweeps API](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverDnXgesvdjSetMaxSweeps)

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- [CMake](https://cmake.org/download) version 3.18 minimum
- Minimum [CUDA 9.0 toolkit](https://developer.nvidia.com/cuda-downloads) is required.

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
$  ./cusolver_syevj_example
```

Sample example output:

```
tol = 1.000000E-07, default value is machine zero
max. sweeps = 15, default value is 100
A = (matlab base-1)
3.50 0.50 0.00
0.50 3.50 0.00
0.00 0.00 2.00
=====
syevj converges
Eigenvalue = (matlab base-1), ascending order
W[1] = 2.000000E+00
W[2] = 3.000000E+00
W[3] = 4.000000E+00
V = (matlab base-1)
0.00 0.71 0.71
0.00 -0.71 0.71
1.00 0.00 0.00
=====
|lambda - W| = 1.332268E-15
residual |A - V*W*V**H|_F = 3.344748E-17
number of executed sweeps = 1
```
