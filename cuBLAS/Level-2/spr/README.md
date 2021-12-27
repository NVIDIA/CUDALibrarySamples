# cuBLAS Level-2 APIs - `cublas<t>spr`

## Description

This code demonstrates a usage of cuBLAS `spr` function to compute a packed symmetric rank-1 update

```
A = | 1.0 | 3.0 | 
    | 3.0 | 4.0 |
    
x = | 5.0 | 6.0 |
```

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
- [cublas\<t>spr API](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-spr)

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
$  ./cublas_spr_example
```

Sample example output:

```
AP
1.00 3.00 
3.00 4.00 
=====
x
5.00 6.00 
=====
AP
26.00 39.00 
33.00 4.00 
=====
```
