# cuBLAS Extension APIs - `cublasGemmBatchedEx`

## Description

This code demonstrates a usage of cuBLAS `GemmBatchedEx` function to compute batches of matrix-matrix products

```
A = | 1.0 | 2.0 | 5.0 | 6.0 |
    | 3.0 | 4.0 | 7.0 | 8.0 |

B = | 5.0 | 6.0 |  9.0 | 10.0 |
    | 7.0 | 8.0 | 11.0 | 12.0 |
```

This function is an extension of `cublas\<t>gemmBatched` that performs the matrix-matrix multiplication of a batch of matrices and allows the user to individually specify the data types for each of the A, B and C matrix arrays, the precision of computation and the GEMM algorithm to be run. Like `cublas\<t>gemmBatched`, the batch is considered to be "uniform", i.e. all instances have the same dimensions (m, n, k), leading dimensions (lda, ldb, ldc) and transpositions (transa, transb) for their respective A, B and C matrices. The address of the input matrices and the output matrix of each instance of the batch are read from arrays of pointers passed to the function by the caller. Supported combinations of arguments are listed further down in this section.

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
- [cublasGemmBatchedEx API](https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmBatchedEx)

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
$  ./cublas_GemmBatchedEx_example
```

Sample example output:

```
A[0]
1 2
3 4
=====
A[1]
5 6
7 8
=====
B[0]
5 6
7 8
=====
B[1]
9 10
11 12
=====
C[0]
19 43
22 50
=====
C[1]
111 151
122 166
=====
```
