# cuRAND Host APIs - `curandCreateGenerator[Host] - XORWOW - Uniform`

## Description

This code demonstrates a usage of cuRAND `curandCreateGenerator[Host]` to generate uniform XORWOW pseudorandom generated numbers

See documentation for further details.

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
- [curandCreateGeneratorHost API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g35b6e9396d5b54b52ba9053496ad4ff4)
- [curandCreateGenerator API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g56ff2b3cf7e28849f73a1e22022bcbfd)
- [curandSetPseudoRandomGeneratorSeed API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gbcd2982aa3d53571b8ad12d8188b139b)
- [curandGenerateUniform API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g5df92a7293dc6b2e61ea481a2069ebc2)

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
$ Open curand_examples.sln project in Visual Studio and build
```

# Usage
```
$  ./curand_xorwow_uniform_example
```

Sample example output:

```
Host
0.145468
0.820181
0.550399
0.294830
0.914733
0.868979
0.321921
0.782857
0.011302
0.285450
=====
Device
0.145468
0.820181
0.550399
0.294830
0.914733
0.868979
0.321921
0.782857
0.011302
0.285450
=====
```
