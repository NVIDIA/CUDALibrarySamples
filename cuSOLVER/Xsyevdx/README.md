# cuSOLVER Standard Symmetric Dense Eigenvalue solver example

## Description

This code demonstrates a usage of cuSOLVER Xsyevdx 64-bit function for using syevdx to compute the spectrum of a dense symmetric system by

_**A**x = &lambda;x_

where A is a 3x3 dense symmetric matrix
```
A = | 3.5 | 0.5 | 0.0 |
    | 0.5 | 3.5 | 0.0 |
    | 0.0 | 0.0 | 2.0 |
```

The following code uses syevdx to compute eigenvalues and eigenvectors, then compare to exact eigenvalues {2,3,4}.

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
- [cusolverDnXsyevdx_bufferSize API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDnXsyevdx)
- [cusolverDnXsyevdx API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDnXsyevdx)

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
$  ./cusolver_Xsyevdx_example
```

Sample example output:

```
A = (matlab base-1)
3.50 0.50 0.00
0.50 3.50 0.00
0.00 0.00 2.00
=====
after Xsyevdx: info = 0
eigenvalue = (matlab base-1), ascending order
W[1] = 2.000000E+00
W[2] = 3.000000E+00
W[3] = 4.000000E+00
V = (matlab base-1)
0.00 -0.71 0.71
0.00 0.71 0.71
1.00 0.00 0.00
=====
|lambda - W| = 0.000000E+00
```
