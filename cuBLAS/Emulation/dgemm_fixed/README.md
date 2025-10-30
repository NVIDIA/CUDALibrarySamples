# cuBLAS Emulation With Level-3 APIs - `cublasDgemm` (Fixed Mode)

## Description

This code demonstrates a usage of cuBLAS `DGEMM` function to compute a matrix-matrix product using floating point emulation via fixed mantissa control.

```
A = |  1.0 |  3.0 |
    |  2.0 |  4.0 |

B = |  5.0 |  7.0 |
    |  6.0 |  8.0 |
```

A fixed mantissa control allows users to specify how many mantissa bits will be retained by the emulation algorithm.

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
- [cublasDgemm API](https://docs.nvidia.com/cuda/cublas/index.html#cublasdgemm)
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
$  ./cublas_dgemm_fixed_example
```

Sample example output:

```
A
1.00 3.00
2.00 4.00
=====
B
5.00 7.00
6.00 8.00
=====
C
23.00 31.00
34.00 46.00
=====
```
