# cuBLAS Emulation With Level-3 APIs - `cublasZgemm` (Dynamic Mantissa Control)

## Description

This code demonstrates a usage of cuBLAS `ZGEMM` function to compute a matrix-matrix product using floating point emulation via the automatic dynamic precision framework.

```
A = |  1.0 + 1.0j |  3.0 - 3.0j |
    |  2.0 - 2.0j |  4.0 + 4.0j |

B = |  5.0 + 5.0j |  7.0 - 7.0j |
    |  6.0 - 6.0j |  8.0 + 8.0j |
```

Dynamic mantissa control leverages an automatic dynamic precision framework to determine how many emulated mantissa bits should be retained to have the same or better accuracy than native FP64.

See the [floating point emulation](https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation) documentation for further details.

## Supported SM Architectures

All GPUs supported by CUDA Toolkit (https://developer.nvidia.com/cuda-gpus)

While the API is supported on all GPUs supported by the CUDA toolkit, emulated algorithms are implemented on a subset of architectures, see [here](https://docs.nvidia.com/cuda/cublas/#floating-point-emulation-support-overview) for more details.

## Supported OSes

Linux  
Windows

## Supported CPU Architecture

x86_64  
arm64-sbsa

## CUDA APIs involved
- [cublasZgemm API](https://docs.nvidia.com/cuda/cublas/index.html#cublaszgemm)
- [cublasSetEmulationStrategy API](https://docs.nvidia.com/cuda/cublas/index.html#cublassetemulationstrategy)
- [cublasSetFixedPointEmulationMantissaControl API](https://docs.nvidia.com/cuda/cublas/index.html#cublassetfixedpointemulationmantissacontrol)

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers (r580+)
- CUDA Toolkit 13.0 Update 2 or newer
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
$  ./cublas_zgemm_example
```

Sample example output:

```
A
1.00+1.00i 3.00-3.00i
2.00-2.00i 4.00+4.00i
=====
B
5.00+5.00i 7.00-7.00i
6.00-6.00i 8.00+8.00i
=====
C
-26.00-26.00i 62.00-62.00i
68.00-68.00i 36.00+36.00i
=====
```
