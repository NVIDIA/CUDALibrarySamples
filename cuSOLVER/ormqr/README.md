# cuSOLVER QR factorization dense linear solver example

## Description

This code demonstrates a usage of cuSOLVER ormqr function for doing dense QR factorization to solve a linear system 
```
Ax=b
```
A is a 3x3 dense matrix, nonsingular
```
A = | 1.0 | 2.0 | 3.0 |
    | 4.0 | 5.0 | 6.0 |
    | 2.0 | 1.0 | 1.0 |
```

Examples perform following steps for both APIs:
- A = Q*R by GEQRF
- B = Q^T*B by ORMQR
- Solve R*X = B by TRSM

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
- [cusolverDnDormqr_bufferSize API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-ormqr)
- [cusolverDnDormqr API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-ormqr)
- [cusolverDnDgeqrf API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-geqrf)

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

## Build command on Windows
```
$ mkdir build
$ cd build
$ cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
$ Open cusolver_examples.sln project in Visual Studio and build
```

# Usage
```
$  ./cusolver_ormqr_example
```

Sample example output:

```
A = (matlab base-1)
A(1,1) = 1.000000
A(1,2) = 2.000000
A(1,3) = 3.000000A = (matlab base-1)
1 2 3
4 5 6
2 1 1
=====
B = (matlab base-1)
6
15
4
=====
after geqrf: info_gpu = 0
after ormqr: info_gpu = 0
X = (matlab base-1)
1
1
1
```
