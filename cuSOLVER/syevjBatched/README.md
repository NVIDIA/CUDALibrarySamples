# cuSOLVER Batched Standard Symmetric Dense Eigenvalue solver (via Jacobi method) example

## Description

This code demonstrates a usage of cuSOLVER syevjBatched function for using syevjBatched to compute spectrum of a pair of dense symmetric matrices  by

_**A**<sub>j</sub>x = &lambda;x_

where A0 and A1 is a 3x3 dense symmetric matrices
```
A0 = |  1.0 | -1.0 | 0.0 |
     | -1.0 |  2.0 | 0.0 |
     |  0.0 |  0.0 | 0.0 |

A1 = | 3.0 | 4.0 | 0.0 |
     | 4.0 | 7.0 | 0.0 |
     | 0.0 | 0.0 | 0.0 |
```
The following code uses syevjBatched to compute eigenvalues and eigenvectors

_**A**<sub>j</sub>x = **V**<sub>j</sub> * **W**<sub>j</sub> * **V**<sup>T</sup><sub>j</sub>_

The user can disable/enable sorting by the function [cusolverDnXsyevjSetSortEig](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverDnXsyevjSetSortEig).
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
- [cusolverDnSsyevjBatched_bufferSize API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-syevjbatch)
- [cusolverDnSsyevjBatched API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-syevjbatch)

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
$  ./cusolver_syevjBatched_example
```

Sample example output:

```
A0 = (matlab base-1)
1.00 -1.00 0.00
-1.00 2.00 0.00
0.00 0.00 0.00
=====
A1 = (matlab base-1)
3.00 4.00 0.00
4.00 7.00 0.00
0.00 0.00 0.00
=====
matrix 0: syevj converges
matrix 1: syevj converges
====
W0[0] = 0.381966
W0[1] = 2.618034
W0[2] = 0.000000
====
W1[0] = 0.527864
W1[1] = 9.472136
W1[2] = 0.000000
====
V0 = (matlab base-1)
0.85 -0.53 0.00
0.53 0.85 0.00
0.00 0.00 1.00
V1 = (matlab base-1)
0.85 0.53 0.00
-0.53 0.85 0.00
0.00 0.00 1.00
```
