# cuBLAS Level-3 APIs - `cublas<t>gemmGroupedBatched`

## Description

This code demonstrates a usage of cuBLAS `gemmGroupedBatched` function to compute groups of batched matrix-matrix products

```
Group 0:
A = | 1.0 | 2.0 | 5.0 | 6.0 |
    | 3.0 | 4.0 | 7.0 | 8.0 |

B = | 5.0 | 6.0 |  9.0 | 10.0 |
    | 7.0 | 8.0 | 11.0 | 12.0 |

Group 1:
A = | 1.0 | 2.0 | 3.0 |
    | 4.0 | 5.0 | 6.0 |
    | 7.0 | 8.0 | 9.0 |

B = |  4.0 |  5.0 |  6.0 |
    |  7.0 |  8.0 |  9.0 |
    | 10.0 | 11.0 | 12.0 |
```

This function performs the matrix-matrix multiplication on groups of matrices. A given group is considered to be “uniform”, i.e. all instances have the same dimensions (m, n, k), leading dimensions (lda, ldb, ldc) and transpositions (transa, transb) for their respective A, B and C matrices. However, the dimensions, leading dimensions, transpositions, and scaling factors (alpha, beta) may vary between groups. The address of the input matrices and the output matrix of each instance of the batch are read from arrays of pointers passed to the function by the caller.

See documentation for further details.

## Supported SM Architectures

All GPUs supported by CUDA Toolkit (https://developer.nvidia.com/cuda-gpus)  

## Supported OSes

Linux  
Windows

## Supported CPU Architecture

x86_64  
arm64-sbsa

## CUDA APIs involved
- [cublas\<t>gemmGroupedBatched API](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmgroupedbatched)

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
$  ./cublas_gemmGroupedBatched_example
```

Sample example output:

```
Group 0:
A[0]
1.00 2.00
3.00 4.00
=====
B[0]
5.00 6.00
7.00 8.00
=====
A[1]
5.00 6.00
7.00 8.00
=====
B[1]
9.00 10.00
11.00 12.00
=====

Group 1:
A[0]
1.00 2.00 3.00
4.00 5.00 6.00
7.00 8.00 9.00
=====
B[0]
4.00 5.00 6.00
7.00 8.00 9.00
10.00 11.00 12.00
=====

Group 0:
C[0]
19.00 22.00
43.00 50.00
=====
C[1]
111.00 122.00
151.00 166.00
=====

Group 1:
C[0]
48.00 54.00 60.00
111.00 126.00 141.00
174.00 198.00 222.00
=====
```
