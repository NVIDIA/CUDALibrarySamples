# cuSOLVER orthgonalization by QR factorization example

## Description

This code demonstrates a usage of cuSOLVER orgqr function for doing orthgonalization by QR factorization 
```
A = Q * R
```
A is a 3x2 dense matrix
```
A = | 1.0 | 2.0 |
    | 4.0 | 5.0 |
    | 2.0 | 1.0 |
```

Examples perform following steps for both APIs:
- A = Q*R by GEQRF
- Form Q by ORGQR
- Check if Q is unitary or not

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
- [cusolverDnDorgqr_bufferSize API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-orgqr)
- [cusolverDnDorgqr API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-orgqr)
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
1.00 2.00
4.00 5.00
2.00 1.00
=====
after geqrf: info = 0
after ormqr: info = 0
Q = (matlab base-1)
-0.22 0.53
-0.87 0.27
-0.44 -0.80
|I - Q**T*Q| = 1.414214E+00
```
