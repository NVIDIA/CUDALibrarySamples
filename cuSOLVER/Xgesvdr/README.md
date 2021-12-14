# cuSOLVER Singular Value Decomposition example

## Description

This code demonstrates a usage of cuSOLVER Xgesvdp 64-bit functions for using cusolverDnXgesvdr. The code shows how to compute the rank-2 approximation of the 5x5 dense matrix A:

```
A = | 0.7640 | 0.0369 | 0.4201 | 0.5033 | 0.5832 |
    | 0.6141 | 0.8596 | 0.3920 | 0.9297 | 0.1158 |
    | 0.8172 | 0.6758 | 0.1265 | 0.2121 | 0.3983 |
    | 0.4204 | 0.4559 | 0.9025 | 0.6392 | 0.2149 |
    | 0.0344 | 0.0207 | 0.2307 | 0.5812 | 0.0054 |
```

The following code uses two steps:

Step 1: compute the approximated rank-2 decomposition of A

Step 2: check accuracy of singular values

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
- [cusolverDnXgesvdr_bufferSize  API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDnXgesvdr)
- [cusolverDnXgesvdr API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDnXgesvdr)

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
$  ./cusolver_Xgesvdr_example
```

Sample example output:

```
2, 2
A = (matlab base-1)
0.764207 0.0369729 0.420183 0.503398 0.583257
0.614115 0.859624 0.392045 0.92975 0.115899
0.817242 0.675841 0.126579 0.21214 0.398311
0.420409 0.455947 0.902506 0.639625 0.214927
0.0344609 0.0207483 0.230762 0.581246 0.00540355
=====
m = 5, n = 5, rank = 2, p = 2, iters = 2
S_ref[0]=2.365392  S_gpu=[0]=2.365392  AbsErr=4.975184E-09  RelErr=2.103323E-09
S_ref[1]=0.811178  S_gpu=[1]=0.811178  AbsErr=2.060349E-09  RelErr=2.539947E-09

max_err = 0.000000E+00, max_relerr = 0.000000E+00, eps = 1.000000E-08
Success: max_relerr is smaller than eps
```
