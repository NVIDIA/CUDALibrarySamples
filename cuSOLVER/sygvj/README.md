# cuSOLVER Generalized Symmetric-Definite Dense Eigenvalue solver (via Jacobi method) example

## Description

This code demonstrates a usage of cuSOLVER sygvj  function for using sygvj  to compute spectrum of a pair of dense symmetric matrices (A,B) by

_**A**x = &lambda;**B**x_

where A is a 3x3 dense symmetric matrix
```
A = | 3.5 | 0.5 | 0.0 |
    | 0.5 | 3.5 | 0.0 |
    | 0.0 | 0.0 | 2.0 |
```
where B is a 3x3 positive definite  matrix
```
B = | 10.0 | 2.0 | 3.0 |
    | 2.0 | 10.0 | 5.0 |
    | 3.0 | 5.0 | 10.0 |
```
The following code uses sygvj to compute eigenvalues and eigenvectors.

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
- [cusolverDnDsygvj_bufferSize API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-sygvj)
- [cusolverDnDsygvj API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-sygvj)
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
$  ./cusolver_sygvj_example
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
B = (matlab base-1)
10.00 2.00 3.00
2.00 10.00 5.00
3.00 5.00 10.00
=====
sygvj converges
Eigenvalue = (matlab base-1), ascending order
W[1] = 1.586603E-01
W[2] = 3.707515E-01
W[3] = 6.000000E-01
V = (matlab base-1)
0.05 -0.31 -0.12
0.09 0.16 -0.31
0.24 0.02 0.29
=====
|lambda - W| = 3.330669E-16
residual |A - V*W*V**H|_F = 1.135989E-11
number of executed sweeps = 4
```
