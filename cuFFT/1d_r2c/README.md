# cuFFT 1D FFT R2C example

## Description

In this example a one-dimensional real-to-complex transform is applied to the input data.

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
- [cufftExecR2C API](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftexecr2c-cufftexecd2z)

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

# Usage 1
```
$  ./bin/1d_r2c_example
```

Sample example output (batch_size=1):

```
Input array:
0.000000
1.000000
2.000000
3.000000
4.000000
5.000000
6.000000
7.000000
=====
Output array:
28.000000 + 0.000000j
-4.000000 + 9.656855j
-4.000000 + 4.000000j
-4.000000 + 1.656854j
-4.000000 + 0.000000j
=====
```

Sample example output (batch_size=2):

```
Input array:
0.000000
1.000000
2.000000
3.000000
4.000000
5.000000
6.000000
7.000000
8.000000
9.000000
10.000000
11.000000
12.000000
13.000000
14.000000
15.000000
=====
Output array:
120.000000 + 0.000000j
-8.000000 + 40.218716j
-8.000000 + 19.313707j
-7.999999 + 11.972845j
-8.000000 + 8.000000j
-8.000000 + 5.345429j
-8.000000 + 3.313708j
-8.000000 + 1.591299j
-8.000000 + 0.000000j
=====
```