# cuBLAS Level-3 APIs - `cublas<t>trsmBatched`

## Description

This code demonstrates a usage of cuBLAS `trsmBatched` function to compute a batch of triangular linear systems with multiple right-hand-sides

```
A = | 1.0 | 2.0 |
    | 3.0 | 4.0 |

B = | 5.0 | 6.0 |
    | 7.0 | 8.0 |
```

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
- [cublas\<t>trsmBatched API](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-trsmBatched)

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
$  ./cublas_trsmBatched_example
```

Sample example output:

```
A[0]
1.00 2.00 
3.00 4.00 
=====
A[1]
5.00 6.00 
7.00 8.00 
=====
B[0] (in)
5.00 6.00 
7.00 8.00 
=====
B[1] (in)
9.00 10.00 
11.00 12.00 
=====
B[0] (out)
1.50 2.00 
1.75 2.00 
=====
B[1] (out)
0.15 0.20 
1.38 1.50 
=====

```
