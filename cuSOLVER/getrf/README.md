# cuSOLVER LU Factorization example

## Description

This code demonstrates a usage of cuSOLVER getrf/getrs functions for using dense LU factorization to solve a linear system

_**A**x = b_

All matrices A<sub>i</sub> are small perturbations of
```
A = | 1.0 | 2.0 | 3.0 |
    | 4.0 | 5.0 | 6.0 |
    | 7.0 | 8.0 | 10.0 |
```

The code uses getrf to do LU factorization and getrs to do backward and forward solve. The parameter pivot_on decides whether partial pivoting is performed or not.

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
- [cusolverDnDgetrf API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-getrf)
- [cusolverDnDgeqrs API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-getrs)

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
$  ./cusolver_getrf_example
```

Sample example output:

```
pivot is on : compute P*A = L*U
A = (matlab base-1)
1.00 2.00 3.00
4.00 5.00 6.00
7.00 8.00 10.00
=====
B = (matlab base-1)
1.00
2.00
3.00
=====
pivoting sequence, matlab base-1
Ipiv(1) = 3
Ipiv(2) = 3
Ipiv(3) = 3
L and U = (matlab base-1)
7.00 8.00 10.00
0.14 0.86 1.57
0.57 0.50 -0.50
=====
X = (matlab base-1)
-0.33
0.67
0.00
=====

pivot is off: compute A = L*U (not numerically stable)
A = (matlab base-1)
1.00 2.00 3.00
4.00 5.00 6.00
7.00 8.00 10.00
=====
B = (matlab base-1)
1.00
2.00
3.00
=====
L and U = (matlab base-1)
1.00 2.00 3.00
4.00 -3.00 -6.00
7.00 2.00 1.00
=====
X = (matlab base-1)
-0.33
0.67
0.00
=====
```
