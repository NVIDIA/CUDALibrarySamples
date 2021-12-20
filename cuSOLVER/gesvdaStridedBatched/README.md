# cuSOLVER Approximate Singular Value Decomposition example

## Description

This code demonstrates a usage of cuSOLVER gesvdaStridedBatched function to approximate singular value decomposition by gesvdaStridedBatched.

_**A** = **U** * **&Sigma;** * **V**<sup>H</sup>_

A0 and A1 are a 3x2 dense matrices,
```
A0 = | 1.0 | 2.0 |
     | 4.0 | 5.0 |
     | 2.0 | 1.0 |

A1 = | 10.0 | 9.0 |
     |  8.0 | 7.0 |
     |  6.0 | 5.0 |
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
- [cusolverDnSgesvdaStridedBatched_bufferSize  API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-gesvda)
- [cusolverDnSgesvdaStridedBatched API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-gesvda)

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- [CMake](https://cmake.org/download) version 3.18 minimum
- Minimum [CUDA 10.1 toolkit](https://developer.nvidia.com/cuda-downloads) is required.

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
$  ./cusolver_gesvdaStridedBatched_example
```

Sample example output:

```
A0 = (matlab base-1)
1 2 
4 5 
2 1 
=====
A1 = (matlab base-1)
10 9 
8 7 
6 5 
=====
0-th matrix, gesvda converges 
1-th matrix, gesvda converges 
S0 = (matlab base-1)
7.06528 
1.04008 
=====
U0 = (matlab base-1)
0.308219 -0.488195 
0.906133 -0.110706 
0.289696 0.865685 
=====
V) = (matlab base-1)
0.638636 0.769509 
0.769509 -0.638636 
=====
|S0 - S0_exact|_sup = 0.000000E+00 
residual |A0 - U0*S0*V0**H|_F = 4.768372E-07 
S1 = (matlab base-1)
18.8396 
0.260036 
=====
U1 = (matlab base-1)
0.714069 0.568717 
0.564241 -0.122334 
0.414412 -0.813385 
=====
V1 = (matlab base-1)
0.750603 -0.660754 
0.660754 0.750603 
=====
|S1 - S1_exact|_sup = 0.000000E+00 
residual |A1 - U1*S1*V1**H|_F = 0.000000E+00
```
