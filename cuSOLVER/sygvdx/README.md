# cuSOLVER Generalized Symmetric-Definite Dense Eigenvalue solver example

## Description

This code demonstrates a usage of cuSOLVER `sygvdx` function for using sygvd to compute spectrum of a pair of dense symmetric matrices (A,B) by

_**A**x = &lambda;**B**x_

where A is a 3x3 dense symmetric matrix
```
A = | 3.5 | 0.5 | 0.0 |
    | 0.5 | 3.5 | 0.0 |
    | 0.0 | 0.0 | 2.0 |
```
where B is a 3x3 positive definite  matrix
```
B = | 10.0 | 2.0 | 3.0 |
    | 2.0 | 10.0 | 5.0 |
    | 3.0 | 5.0 | 10.0 |
```

The following code uses `sygvdx` to compute eigenvalues and eigenvectors, then compare to exact eigenvalues {0.158660256604, 0.370751508101882, 0.6}.

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
- [cusolverDnDsygvdx_bufferSize API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-sygvdx)
- [cusolverDnDsygvdx API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-sygvdx)

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
$ Open cusolver_examples.sln project in Visual Studio and build
```

# Usage
```
$  ./cusolver_sygvdx_example
```

Sample example output:

```
A = (matlab base-1)
3.50 0.50 0.00
0.50 3.50 0.00
0.00 0.00 2.00
=====
B = (matlab base-1)
10.00 2.00 3.00
2.00 10.00 5.00
3.00 5.00 10.00
=====
after sygvd: info = 0
eigenvalue = (matlab base-1), ascending order
W[1] = 1.586603E-01
W[2] = 3.707515E-01
W[3] = 6.000000E-01
V = (matlab base-1)
0.05 0.31 -0.03
0.09 -0.16 -0.07
0.24 -0.02 0.16
=====
Eigenvalues found = 2
|lambda - W| = 3.053113E-16
```
