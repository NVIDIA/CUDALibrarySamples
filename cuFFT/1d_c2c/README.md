# cuFFT 1D FFT C2C example

## Description

In this example a one-dimensional complex-to-complex transform is applied to the input data. Afterwards an inverse transform is performed on the computed frequency domain representation.

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
- [cufftExecC2C API](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftexecc2c-cufftexecz2z)


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
$  ./bin/1d_c2c_example
```

Sample example output (batch_size=1):

```
Input array:
0.000000 + 0.000000j
1.000000 + -1.000000j
2.000000 + -2.000000j
3.000000 + -3.000000j
4.000000 + -4.000000j
5.000000 + -5.000000j
6.000000 + -6.000000j
7.000000 + -7.000000j
=====
Output array:
0.000000 + 0.000000j
8.000001 + -8.000000j
16.000000 + -16.000000j
24.000002 + -24.000004j
32.000000 + -32.000000j
40.000000 + -40.000000j
48.000000 + -48.000000j
56.000000 + -55.999996j
=====
```

Sample example output (batch_size=2):

```
Input array:
0.000000 + 0.000000j
1.000000 + -1.000000j
2.000000 + -2.000000j
3.000000 + -3.000000j
4.000000 + -4.000000j
5.000000 + -5.000000j
6.000000 + -6.000000j
7.000000 + -7.000000j
8.000000 + -8.000000j
9.000000 + -9.000000j
10.000000 + -10.000000j
11.000000 + -11.000000j
12.000000 + -12.000000j
13.000000 + -13.000000j
14.000000 + -14.000000j
15.000000 + -15.000000j
=====
Output array:
0.000004 + 0.000000j
16.000015 + -16.000004j
32.000004 + -32.000004j
48.000004 + -48.000004j
64.000000 + -64.000000j
80.000000 + -80.000008j
96.000000 + -96.000015j
112.000015 + -112.000000j
128.000000 + -128.000000j
144.000000 + -144.000000j
160.000000 + -160.000000j
176.000000 + -176.000000j
192.000000 + -192.000000j
208.000000 + -208.000000j
224.000000 + -223.999985j
239.999985 + -240.000000j
=====
```