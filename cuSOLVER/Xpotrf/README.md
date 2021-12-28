# cuSOLVER Cholesky Factorization example

## Description

This code demonstrates a usage of cuSOLVER Xpotrf/Xpotrs 64-bit functions for using dense Cholesky factorization of a Hermitian positive-definite matrix

_**A** * **X** = **B**_

All matrices A<sub>i</sub> are small perturbations of
```
A = | 1.0 | 2.0 | 3.0 |
    | 4.0 | 5.0 | 6.0 |
    | 7.0 | 8.0 | 10.0 |
```

where `A` is a `n×n` Hermitian matrix, only lower or upper part is meaningful using the generic API interface. The input parameter uplo indicates which part of the matrix is used. The function would leave other part untouched.

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
- [cusolverDnXpotrf API](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverDnXpotrf)
- [cusolverDnXpotrs API](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverDnXpotrs)

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
$  ./cusolver_Xpotrf_example
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
after Xpotrf: info = 0
L and U = (matlab base-1)
1.00 2.00 3.00
2.00 1.00 5.00
3.00 -1.00 1.41
=====
X = (matlab base-1)
1.00
0.00
0.00
=====
```
