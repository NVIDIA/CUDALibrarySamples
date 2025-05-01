# cuBLAS Emulation With Extension APIs - `cublasGemmEx`

## Description

This code demonstrates a usage of cuBLAS `GemmEx` function to compute a matrix-matrix product using floating point emulation via the BF16x9 method

```
A = |  1.0 |  5.0 |  9.0 | 13.0 |
    |  2.0 |  6.0 | 10.0 | 14.0 |
    |  3.0 |  7.0 | 11.0 | 15.0 |
    |  4.0 |  8.0 | 12.0 | 16.0 |

B = |  1.0 |  2.0 |  3.0 |  4.0 |
    |  5.0 |  6.0 |  7.0 |  8.0 |
    |  9.0 | 10.0 | 11.0 | 12.0 |
    | 13.0 | 14.0 | 15.0 | 16.0 |
```

See the [floating point emulation](https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation) documentation for further details.

## Supported SM Architectures

All GPUs supported by CUDA Toolkit (https://developer.nvidia.com/cuda-gpus)

While the API is supported on all GPUs supported by the CUDA toolkit, emulated algorithms are implemented on a subset of architectures, see [here](https://docs.nvidia.com/cuda/cublas/#floating-point-emulation-support-overview) for more details.

## Supported OSes

Linux  
Windows

## Supported CPU Architecture

x86_64  
ppc64le  
arm64-sbsa

## CUDA APIs involved
- [cublasGemmEx API](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex)
- [cublasSetEmulationStrategy API](https://docs.nvidia.com/cuda/cublas/index.html#cublassetemulationstrategy)

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
$ Open cublas_examples.sln project in Visual Studio and build
```

# Usage
```
$  ./cublas_GemmEx_example
```

Sample example output:

```
A
1.00 5.00 9.00 13.00
2.00 6.00 10.00 14.00
3.00 7.00 11.00 15.00
4.00 8.00 12.00 16.00
=====
B
1.00 2.00 3.00 4.00
5.00 6.00 7.00 8.00
9.00 10.00 11.00 12.00
13.00 14.00 15.00 16.00
=====
C (fp32)
276.00 304.00 332.00 360.00
304.00 336.00 368.00 400.00
332.00 368.00 404.00 440.00
360.00 400.00 440.00 480.00
=====
C (bf16x9)
276.00 304.00 332.00 360.00
304.00 336.00 368.00 400.00
332.00 368.00 404.00 440.00
360.00 400.00 440.00 480.00
=====
```
