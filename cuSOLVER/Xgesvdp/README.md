# cuSOLVER Singular Value Decomposition example

## Description

This code demonstrates a usage of cuSOLVER Xgesvdp 64-bit functions for using cusolverDnXgesvdp

_**A** = **U** * **&Sigma;** * **V**<sup>H</sup>_

All matrices A<sub>i</sub> are small perturbations of
```
A = | 1.0 | 2.0 |
    | 4.0 | 5.0 |
    | 2.0 | 1.0 |
```

The following code uses three steps:

Step 1: compute A = U * S * VT

Step 2: check accuracy of singular value

Step 3: measure residual A - U * S * VT

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
- [cusolverDnXgesvdp_bufferSize  API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDnXgesvdp)
- [cusolverDnXgesvdp API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDnXgesvdp)

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
$  ./cusolver_Xgesvdp_example
```

Sample example output:

```
AA = (matlab base-1)
1.00 2.00
4.00 5.00
2.00 1.00
=====
after Xgesvdp: info = 0
=====
S = (matlab base-1)
7.07
1.04
=====
U = (matlab base-1)
0.31 0.49
0.91 0.11
0.29 -0.87
=====
V = (matlab base-1)
0.64 -0.77
0.77 0.64
=====
|S - S_exact| = 8.881784E-16
|A - U*S*V**T| = 3.802413E-15
h_err_sigma = 0.000000E+00
h_err_sigma is 0 if the singular value of A is not close to zero
```
