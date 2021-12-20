# cuSOLVER MultiGPU Standard Symmetric Dense Eigenvalue solver example

## Description

This chapter provides three examples to perform multiGPU symmetric eigenvalue solver. The difference among them is how to generate the testing matrix. The testing matrix is a tridiagonal matrix, from standard 3-point stencil of Laplacian operator with Dirichlet boundary condition, so each row has (-1, 2, -1) signature.

The spectrum has analytic formula, we can check the accuracy of eigenvalues easily. The user can change the dimension of the matrix to measure the performance of eigenvalue solver.

The example code enables peer-to-peer access to take advantage of NVLINK. The user can check the performance by on/off peer-to-peer access.

The procedures of these three examples are 1) to prepare a tridiagonal matrix in distributed sense, 2) to query size of the workspace and to allocate the workspace for each device, 3) to compute eigenvalues and eigenvectors, and 4) to check accuracy of eigenvalues.

The example 1 allocates distributed matrix by calling `createMat`. It generates the matrix on host memory and copies it to distributed device memory via `memcpyH2D`.

The example 2 allocates distributed matrix maunally, generates the matrix on host memory and copies it to distributed device memory manually. This example is for the users who are familiar with data layout of ScaLAPACK.

The example 3 allocates distributed matrix by calling `createMat` and generates the matrix element-by-element on distributed matrix via `memcpyH2D`. The user needs not to know the data layout of ScaLAPACK. It is useful when the matrix is sparse.

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
- [cusolverMgSyevd_bufferSize API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-syevd)
- [cusolverMgSyevd API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-syevd)
- [cusolverMgCreateDeviceGrid API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-grid)
- [cusolverMgDeviceSelect API](https://docs.nvidia.com/cuda/cusolver/index.html#mg-device)

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- [CMake](https://cmake.org/download) version 3.18 minimum
- Minimum [CUDA 10.1 toolkit](https://developer.nvidia.com/cuda-downloads) is required.

## Build command on Linux
```
$ mkdir build
$ cd build
$ cmake .. #-DSHOW_FORMAT=ON
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
$  ./cusolver_MgSyevd_example1
```

Sample example output:

```
Test 1D Laplacian of order 2111
Step 1: create Mg handle and select devices
        There are 1 GPUs
        device 0, NVIDIA GeForce RTX 2080 with Max-Q Design, cc 7.5
step 2: Enable peer access.
Step 3: allocate host memory A
Step 4: prepare 1D Laplacian
Step 5: create matrix descriptors for A and D
Step 6: allocate distributed matrices A and D
Step 7: prepare data on devices
Step 8: allocate workspace space
        Allocate device workspace, lwork = 19648160
Step 9: compute eigenvalues and eigenvectors
Step 10: copy eigenvectors to A and eigenvalues to D
Step 11: verify eigenvales
     lambda(k) = 4 * sin(pi/2 *k/(N+1))^2 for k = 1:N

|D - lambda|_inf = 2.220446E-15

step 12: free resources
```

# Usage 2
```
$  ./cusolver_MgSyevd_example2
```

Sample example output:

```
Test 1D Laplacian of order 2111
Step 1: create Mg handle and select devices
        There are 1 GPUs
        device 0, NVIDIA GeForce RTX 2080 with Max-Q Design, cc 7.5
step 2: Enable peer access.
Step 3: allocate host memory A
Step 4: prepare 1D Laplacian
Step 5: create matrix descriptors for A and D
Step 6: allocate distributed matrices A and D
Step 7: prepare data on devices
Step 8: allocate workspace space
        Allocate device workspace, lwork = 19648160
Step 9: compute eigenvalues and eigenvectors
Step 10: copy eigenvectors to A and eigenvalues to D
Step 11: verify eigenvales
     lambda(k) = 4 * sin(pi/2 *k/(N+1))^2 for k = 1:N

|D - lambda|_inf = 2.220446E-15

step 12: free resources
```

# Usage 3
```
$  ./cusolver_MgSyevd_example3
```

Sample example output:

```
Test 1D Laplacian of order 2111
Step 1: Create Mg handle and select devices
        There are 1 GPUs
        Device 0, NVIDIA GeForce RTX 2080 with Max-Q Design, cc 7.5
step 2: Enable peer access.
Step 3: Allocate host memory A
Step 4: Create matrix descriptors for A and D
Step 5: Allocate distributed matrices A and D, A = 0 and D = 0
Step 6: Prepare 1D Laplacian
Step 7: Allocate workspace space
        Allocate device workspace, lwork = 19648160
Step 8: Compute eigenvalues and eigenvectors
Step 9: Copy eigenvectors to A and eigenvalues to D
Step 11: Verify eigenvales
     lambda(k) = 4 * sin(pi/2 *k/(N+1))^2 for k = 1:N

|D - lambda|_inf = 2.220446E-15

step 12: Free resources
```
