# cuSOLVER MultiGPU LU Decomposition example

## Description

This chapter provides examples to perform multiGPU linear solver.

The example code enables peer-to-peer access to take advantage of NVLINK. The user can check the performance by on/off peer-to-peer access.

The example 1 solves linear system by Cholesky factorization (`potrf` and `potrs`). It allocates distributed matrix by calling `createMat`. Then generates the matrix on host memory and copies it to distributed device memory via `memcpyH2D`.

The example 2 solves linear system using the inverse of an Hermitian positive definite matrix using (`potrf` and `potri`). It allocates distributed matrix by calling `createMat`. Then generates the matrix on host memory and copies it to distributed device memory via `memcpyH2D`.

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
- [cusolverMgPotrf_bufferSize API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-potrf)
- [cusolverMgPotrs_bufferSize API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-potrs)
- [cusolverMgPotri_bufferSize API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-potri)
- [cusolverMgPotrf API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-potrf)
- [cusolverMgPotrs API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-potrs)
- [cusolverMgPotri API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-potri)
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
$  ./cusolver_MgPotrf_example1
```

Sample example output w/ 1 GPU:

```
Test 1D Laplacian of order 8
Step 1: Create Mg handle and select devices 
        There are 1 GPUs 
        Device 0, NVIDIA TITAN RTX, cc 7.5 
step 2: Enable peer access.
Step 3: Allocate host memory A 
Step 4: Prepare 1D Laplacian for A and X = ones(N,NRHS) 
Step 5: Create RHS for reference solution on host B = A*X 
Step 6: Create matrix descriptors for A and D 
Step 7: Allocate distributed matrices A and B 
Step 8: Prepare data on devices 
Step 9: Allocate workspace space 
        Allocate device workspace, lwork = 1064960 
Step 10: Solve A*X = B by POTRF and POTRS 
Step 11: Solution vector B
Step 12: Measure residual error |b - A*x| 
errors for X[:,1] 
        |b - A*x|_inf = 2.220446E-16
        |x|_inf = 1.000000E+00
        |b|_inf = 1.000000E+00
        |A|_inf = 4.000000E+00
        |b - A*x|/(|A|*|x|+|b|) = 4.440892E-17

errors for X[:,2] 
        |b - A*x|_inf = 2.220446E-16
        |x|_inf = 1.000000E+00
        |b|_inf = 1.000000E+00
        |A|_inf = 4.000000E+00
        |b - A*x|/(|A|*|x|+|b|) = 4.440892E-17

step 12: Free resources
```

Sample example output w/ 2 GPU:

```
Test 1D Laplacian of order 8
Step 1: Create Mg handle and select devices 
        There are 2 GPUs 
        Device 0, NVIDIA TITAN RTX, cc 7.5 
        Device 1, NVIDIA TITAN RTX, cc 7.5 
step 2: Enable peer access.
         Enable peer access from gpu 0 to gpu 1
         Enable peer access from gpu 1 to gpu 0
Step 3: Allocate host memory A 
Step 4: Prepare 1D Laplacian for A and X = ones(N,NRHS) 
Step 5: Create RHS for reference solution on host B = A*X 
Step 6: Create matrix descriptors for A and D 
Step 7: Allocate distributed matrices A and B 
Step 8: Prepare data on devices 
Step 9: Allocate workspace space 
        Allocate device workspace, lwork = 1064960 
Step 10: Solve A*X = B by POTRF and POTRS 
Step 11: Solution vector B
Step 12: Measure residual error |b - A*x| 
errors for X[:,1] 
        |b - A*x|_inf = 2.220446E-16
        |x|_inf = 1.000000E+00
        |b|_inf = 1.000000E+00
        |A|_inf = 4.000000E+00
        |b - A*x|/(|A|*|x|+|b|) = 4.440892E-17

errors for X[:,2] 
        |b - A*x|_inf = 2.220446E-16
        |x|_inf = 1.000000E+00
        |b|_inf = 1.000000E+00
        |A|_inf = 4.000000E+00
        |b - A*x|/(|A|*|x|+|b|) = 4.440892E-17

step 12: Free resources
```

# Usage 2
```
$  ./cusolver_MgPotrf_example2
```

Sample example output w/ 1 GPU:

```
Test 1D Laplacian of order 8
Step 1: Create Mg handle and select devices 
        There are 1 GPUs 
        Device 0, NVIDIA TITAN RTX, cc 7.5 
step 2: Enable peer access.
Step 3: Allocate host memory A 
Step 4: Prepare 1D Laplacian for A and Xref = ones(N,NRHS) 
Step 5: Create RHS for reference solution on host B = A*X 
Step 6: Create matrix descriptors for A and D 
Step 7: Allocate distributed matrices A and B 
Step 8: Prepare data on devices 
Step 9: Allocate workspace space 
        Allocate device workspace, lwork = 1067008 
Step 10: Solve A*X = B by POTRF and POTRI 
Step 11: Gather INV(A) from devices to host
step 12: solve linear system B := inv(A) * B 
step 13: measure residual error |Xref - Xans| 
errors for X[:,1] 
        |b - A*x|_inf = 4.440892E-16
        |Xref|_inf = 1.000000E+00
        |Xans|_inf = 1.000000E+00
        |A|_inf = 4.000000E+00
        |b - A*x|/(|A|*|x|+|b|) = 8.881784E-17

errors for X[:,2] 
        |b - A*x|_inf = 4.440892E-16
        |Xref|_inf = 1.000000E+00
        |Xans|_inf = 1.000000E+00
        |A|_inf = 4.000000E+00
        |b - A*x|/(|A|*|x|+|b|) = 8.881784E-17

step 14: Free resources
```

Sample example output w/ 2 GPU:

```
Test 1D Laplacian of order 8
Step 1: Create Mg handle and select devices 
        There are 2 GPUs 
        Device 0, NVIDIA TITAN RTX, cc 7.5 
        Device 1, NVIDIA TITAN RTX, cc 7.5 
step 2: Enable peer access.
         Enable peer access from gpu 0 to gpu 1
         Enable peer access from gpu 1 to gpu 0
Step 3: Allocate host memory A 
Step 4: Prepare 1D Laplacian for A and Xref = ones(N,NRHS) 
Step 5: Create RHS for reference solution on host B = A*X 
Step 6: Create matrix descriptors for A and D 
Step 7: Allocate distributed matrices A and B 
Step 8: Prepare data on devices 
Step 9: Allocate workspace space 
        Allocate device workspace, lwork = 1067008 
Step 10: Solve A*X = B by POTRF and POTRI 
Step 11: Gather INV(A) from devices to host
step 12: solve linear system B := inv(A) * B 
step 13: measure residual error |Xref - Xans| 
errors for X[:,1] 
        |b - A*x|_inf = 4.440892E-16
        |Xref|_inf = 1.000000E+00
        |Xans|_inf = 1.000000E+00
        |A|_inf = 4.000000E+00
        |b - A*x|/(|A|*|x|+|b|) = 8.881784E-17

errors for X[:,2] 
        |b - A*x|_inf = 4.440892E-16
        |Xref|_inf = 1.000000E+00
        |Xans|_inf = 1.000000E+00
        |A|_inf = 4.000000E+00
        |b - A*x|/(|A|*|x|+|b|) = 8.881784E-17

step 14: Free resources
```