# cuSOLVER Triangular Matrix Inversion Computation example

## Description

This code demonstrates a usage of cuSOLVER Xtrtri function introduced in CUDA 11.4 that provides interface to compute the inverse of a upper or lower triangular
 matrix.
Examples perform following steps for both APIs:
- Generating random diagonal dominant matrix of provided type on the host
- Initializing required CUDA and cuSOLVER miscelaneous variables
- Allocating required device memory for input data and workbuffer for the solver
- Copying input data to the device 
- Computing the inverse if a upper or lower triangular matrix
- Checking return errors and information
- Releasing used resources

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
- [cusolverDnXtrtri_bufferSize API](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverDnXtrtri_bufferSize)
- [cusolverDnXtrtri API](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverDnXtrtri)

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- [CMake](https://cmake.org/download) version 3.18 minimum
- Minimum [CUDA 11.4 toolkit](https://developer.nvidia.com/cuda-downloads) is required.

## Build command on Linux
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
Make sure that CMake finds expected CUDA Toolkit. If that is not the case you can add argument `-DCMAKE_CUDA_COMPILER=/path/to/cuda-11.4/bin/nvcc` to cmake command.

## Build command on Windows
```
$ mkdir build
$ cd build
$ cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
$ Open cusolver_examples.sln project in Visual Studio and build
```

# Usage
```
$  ./cusolver_Xtrtri_example
```

Sample example output:

```
Generating random diagonal dominant matrix...
Initializing required CUDA and cuSOLVER miscelaneous variables...
Allocating required device memory...
Copying input data to the device...
Quering required device and host workspace size...
Allocating required device workspace...
Allocating required host workspace...
Computing the inverse of a upper triangular matrix...
Copying information back to the host...
Checking returned information...
Verifying results...
Check: PASSED
Destroying CUDA and cuSOLVER miscelaneous variables...
Freeing memory...
Done...
```
