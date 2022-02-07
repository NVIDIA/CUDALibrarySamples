# cuSOLVER QR Factorization example

## Description

This code demonstrates a usage of cuSOLVER Xgetrf/Xgetrs 64-bit functions for using QR factorization of a m×n matrix

_**A** = **Q** * **R**_

where `A` is a `m×n` matrix, `Q` is a `m×n` matrix, and `R` is a `n×n` upper triangular matrix using the generic API interface.

```
A = | 1.0 | 2.0 | 3.0 |
    | 2.0 | 5.0 | 5.0 |
    | 3.0 | 5.0 | 12.0 |
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
- [cusolverDnXgeqrf API](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverDnXgeqrf)

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- [CMake](https://cmake.org/download) version 3.18 minimum
- Minimum [CUDA 11.1 toolkit](https://developer.nvidia.com/cuda-downloads) is required.

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
$  ./cusolver_Xgeqrf_example
```

Sample example output:

```
A = (matlab base-1)
1.00 2.00 3.00
2.00 5.00 5.00
3.00 5.00 12.00
=====
B = (matlab base-1)
1.00
2.00
3.00
=====
after Xgeqrf: info = 0
tau = (matlab base-1)
1.27
1.80
0.00
=====
A = (matlab base-1)
-3.74 -7.22 -13.10
0.42 -1.39 2.52
0.63 -0.33 0.38
=====
```
