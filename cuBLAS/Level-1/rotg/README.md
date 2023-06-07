# cuBLAS Level-1 APIs - `cublas<t>rotg`

## Description

This code demonstrates a usage of cuBLAS `rotg` function to apply the Givens rotation matrix to vector _x_ and _y_

```
A = 2.10
B = 1.20
```

This functions zeros out the second entry of a _2 x 1_ vector _(a,b)_<sup>T<sup>

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
- [cublas\<t>rotg API](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-rotg)

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
$  ./cublas_rotg_example
```

Sample example output:

```
A
2.10
=====
B
1.20
=====
A
2.42
=====
B
0.50
=====
```
