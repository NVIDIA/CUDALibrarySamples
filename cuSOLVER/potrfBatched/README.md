# cuSOLVER batched dense Cholesky factorization example

## Description

This code demonstrates a usage of cuSOLVER potrs_batched function for doing batched dense Cholesky factorization to solve a sequence of linear systems

_**A**[i] * x[i] = b[i]_

each A[i] is a 3x3 dense Hermitian matrix. In this example, there are two matrices, A0 and A1. A0 is positive definite and A1 is not.

The code uses potrfBatched to do Cholesky factorization and potrsBatched to do backward and forward solve. potrfBatched would report singularity on A1.

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
- [cusolverDnDpotrfBatched API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-batchpotrf)
- [cusolverDnDpotrsBatched API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-batchpotrs)

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- [CMake](https://cmake.org/download) version 3.18 minimum
- Minimum [CUDA 9.1 toolkit](https://developer.nvidia.com/cuda-downloads) is required.

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
$  ./cusolver_potrfBatched_example
```

Sample example output:

```
A0 = (matlab base-1)
1.00 2.00 3.00
2.00 5.00 5.00
3.00 5.00 12.00
=====
A1 = (matlab base-1)
1.00 2.00 3.00
2.00 4.00 5.00
3.00 5.00 12.00
=====
B0 = (matlab base-1)
1.00
1.00
1.00
=====
info[0] = 0
info[1] = 2
L = (matlab base-1), upper triangle is don't care
1.00 2.00 3.00
2.00 1.00 5.00
3.00 -1.00 1.41
=====
after potrsBatched: infoArray[0] = 0
X0 = (matlab base-1)
10.50
-2.50
-1.50
=====
```
