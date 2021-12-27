# cuBLAS Extension APIs - `cublasCsyrkEx`

## Description

This code demonstrates a usage of cuBLAS `CSyrkEx` function to perform a symmetric rank-k update

```
A = | 1.1 + 1.2j | 3.5 + 3.6j |
    | 3.5 + 3.6j | 4.7 + 4.8j |
```

This function is an extension of `cublasCsyrk` where the input matrix and output matrix can have a lower precision but the computation is still done in the type `cuComplex`. 

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
- [cublasCsyrkEx API](https://docs.nvidia.com/cuda/cublas/index.html#cublas-csyrkEx)

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
$  ./cublas_CsyrkEx_example
```

Sample example output:

```
A
1.10 + 1.20j 3.50 + 3.60j 
3.50 + 3.60j 4.70 + 4.80j 
=====
C
-28.78 + 26.90j -43.18 + 40.58j 
0.00 + 0.00j -71.98 + 68.66j 
=====
```
