# cuSOLVER iterative refinement solver example

## Description

This code demonstrates usage of cuSOLVER gesv functions introduced in CUDA 10.2 that provides interface to linear system solver with multiple right hand sides using factorization of initial system in specified precision. cuSOLVER provides two sets of APIs for Iterative Refinement Solver functionality - one is similar to LAPACK's GESV and another 'expert' API which gives more configurable options that the user can set through solver parameters.
Examples perform following steps for both APIs:
- Generating random diagonal dominant matrix of provided type on the host
- Generating random right hand side vectors for the linear system on the host
- Initializing required CUDA and cuSOLVER miscelaneous variables
- Allocating required device memory for input data and workbuffer for the solver
- Copying input data to the device 
- Solving the system of equations
- Checking return errors and information
- Releasing used resources

## Key Concepts

Linear Solver, Factorization, Mixed Precision, Tensor Cores

## Supported SM Architectures

[SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  
[SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  
[SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  
[SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  
[SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  

## Supported OSes

Linux  
Windows  

## Supported CPU Architecture

x86_64  
ppc64le  
arm64-sbsa

## CUDA APIs involved
- [cusolverDn gesv LAPACK style API](https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesv)
- [cusolverDnXgesv expert API](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverDNXgesv)

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- [CMake](https://cmake.org/download) version 3.18 minimum
- Minimum [CUDA 10.2 toolkit](https://developer.nvidia.com/cuda-downloads) is required.

## Build command on Linux
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
Make sure that CMake finds expected CUDA Toolkit. If that is not the case you can add argument `-DCMAKE_CUDA_COMPILER=/path/to/cuda-10.2/bin/nvcc` to cmake command.

## Build command on Windows
```
$ mkdir build
$ cd build
$ cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
$ Open cusolver_examples.sln project in Visual Studio and build
```

# Usage
Produced are two binaries - one uses expert API for gesv() function, and another uses lapack style API, with interface similar to LAPACK GESV function.

## Lapack style API

Usage:
```
$  ./cusolver_irs_lapack
```

Sample example output:

```
Generating matrix A on host...
make A diagonal dominant...
Generating matrix B on host...
Generating matrix X on host...
Initializing CUDA...
Allocating memory on device...
Workspace is 12591744 bytes
Solving matrix on device...
Solve info is: 0, iter is: 2
Releasing resources...
Done!

```

## Expert API

Usage:
```
$  ./cusolver_irs_expert

```

Sample example output:

```
Generating matrix A on host...
make A diagonal dominant...
Generating matrix B on host...
Generating matrix X on host...
Initializing CUDA...
Setting up gesv() parameters...
Allocating memory on device...
Workspace is 12591744 bytes
Solving matrix on device...
Solve info is: 0, iter is: 2
Solved matrix 1024x1024 with 1 right hand sides in 19.6782ms
Releasing resources...
Done!


```
