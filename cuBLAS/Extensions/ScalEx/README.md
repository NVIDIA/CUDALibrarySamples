# cuBLAS Extension APIs - `cublasScalEx`

## Description

This code demonstrates a usage of cuBLAS `ScalEx` function to compute the product of a vector by a scalar

```
A = | 1.0 | 2.0 | 3.0 | 4.0 |
``` 

This function is an extension to the routine `cublas<t>scal` where alpha, output data, and compute type can be specified independently.

See documentation for further details.

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
- [cublasScalEx API](https://docs.nvidia.com/cuda/cublas/index.html#cublasscalex)

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
$ Open cublas_examples.sln project in Visual Studio and build
```

# Usage
```
$  ./cublas_ScalEx_example
```

Sample example output:

```
A
1.00 2.00 3.00 4.00
=====
A (scaled)
2.20 4.40 6.60 8.80
=====
```
