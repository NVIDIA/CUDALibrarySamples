# cuBLAS Level-1 APIs - `cublas<t>dot`

## Description

This code demonstrates a usage of cuBLAS `dot` and `dotc` function to apply the dot product to vector _x_ and _y_

```
A = | 1.0 | 2.0 | 3.0 | 4.0 |
B = | 5.0 | 6.0 | 7.0 | 8.0 |
```

For `dotc`
```
A = | 1.1 + 1.2j | 2.3 + 2.4j | 3.5 + 3.6j | 4.7 + 4.8j |
B = | 5.1 + 5.2j | 6.3 + 6.4j | 7.5 + 7.6j | 8.7 + 8.8j |
```

This function computes the dot product of vectors _x_ and _y_.

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
- [cublas\<t>dot  API](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-dot)
- [cublas\<t>dotc  API](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-dot)

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

# Usage 1
```
$  ./cublas_dot_example
```

Sample example output:

```
A
1.00 2.00 3.00 4.00
=====
B
5.00 6.00 7.00 8.00
=====
Result
70.00
```

# Usage 2
```
$  ./cublas_dotc_example
```

Sample example output:

```
A
1.10 + 1.20j 2.30 + 2.40j 3.50 + 3.60j 4.70 + 4.80j
=====
B
5.10 + 5.20j 6.30 + 6.40j 7.50 + 7.60j 8.70 + 8.80j
=====
Result
178.44+-1.60j
```
