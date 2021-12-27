# cuBLAS Level-2 APIs - `cublas<t>hemv`

## Description

This code demonstrates a usage of cuBLAS `hemv` function to compute a Hermitian matrix-vector multiplication

```
A = | 1.1 + 1.2j | 2.3 + 2.4j |
    | 3.5 + 3.6j | 4.7 + 4.8j |
    
x = | 5.1 + 6.2j | 7.3 + 8.4j |
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
- [cublas\<t>hemv API](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-hemv)

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
$  ./cublas_hemv_example
```

Sample example output:

```
A
1.10 + 1.20j 2.30 + 2.40j 
3.50 + 3.60j 4.70 + 4.80j 
=====
x
5.10 + 6.20j 7.30 + 8.40j 
=====
y
-41.42 + 45.90j 19.42 + 102.42j 
=====
```
