# cuSOLVER MultiGPU LU Decomposition example

## Description

This chapter provides examples to perform multiGPU linear solver.

The example code enables peer-to-peer access to take advantage of NVLINK. The user can check the performance by on/off peer-to-peer access.

The example 1 solves linear system by LU with partial pivoting (`getrf` and `getrs`). It allocates distributed matrix by calling `createMat`. Then generates the matrix on host memory and copies it to distributed device memory via `memcpyH2D`.

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
- [cusolverMgGetrf_bufferSize API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-getrf)
- [cusolverMgGetrs_bufferSize API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-getrs)
- [cusolverMgGetrf API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-getrf)
- [cusolverMgGetrs API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-getrs)
- [cusolverMgCreateDeviceGrid API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-grid)
- [cusolverMgDeviceSelect API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-device)

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- [CMake](https://cmake.org/download) version 3.18 minimum
- Minimum [CUDA 10.2 toolkit](https://developer.nvidia.com/cuda-downloads) is required.

## Build command on Linux
```
$ mkdir build
$ cd build
$ cmake .. # -DSHOW_FORMAT=ON
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

# Usage 1
```
$  ./cusolver_MgGetrf_example
```

Sample example output w/ 1 GPU:

```
Test 1D Laplacian of order 611
Step 1: Create Mg handle and select devices 
        There are 1 GPUs 
        Device 0, NVIDIA TITAN RTX, cc 7.5 
step 2: Enable peer access.
Step 3: Allocate host memory A 
Step 4: Prepare 1D Laplacian 
Step 5: Create matrix descriptors for A and D 
Step 6: Allocate distributed matrices A and D 
Step 7: Prepare data on devices 
Step 8: Allocate workspace space 
        Allocate device workspace, lwork = 2299904 
Step 9: Solve A*X = B by GETRF and GETRS 
Step 10: Retrieve IPIV and solution vector X
Step 11: Measure residual error |b - A*x| 

|b - A*x|_inf = 7.275958E-12
|x|_inf = 4.681800E+04
|b|_inf = 1.000000E+00
|A|_inf = 4.000000E+00
|b - A*x|/(|A|*|x|+|b|) = 3.885214E-17

step 12: Free resources
```

Sample example output w/ 2 GPU:

```
Test 1D Laplacian of order 611
Step 1: Create Mg handle and select devices 
        There are 2 GPUs 
        Device 0, NVIDIA TITAN RTX, cc 7.5 
        Device 1, NVIDIA TITAN RTX, cc 7.5 
step 2: Enable peer access.
         Enable peer access from gpu 0 to gpu 1
         Enable peer access from gpu 1 to gpu 0
Step 3: Allocate host memory A 
Step 4: Prepare 1D Laplacian 
Step 5: Create matrix descriptors for A and D 
Step 6: Allocate distributed matrices A and D 
Step 7: Prepare data on devices 
Step 8: Allocate workspace space 
        Allocate device workspace, lwork = 2299904 
Step 9: Solve A*X = B by GETRF and GETRS 
Step 10: Retrieve IPIV and solution vector X
Step 11: Measure residual error |b - A*x| 

|b - A*x|_inf = 7.275958E-12
|x|_inf = 4.681800E+04
|b|_inf = 1.000000E+00
|A|_inf = 4.000000E+00
|b - A*x|/(|A|*|x|+|b|) = 3.885214E-17

step 12: Free resources 
```