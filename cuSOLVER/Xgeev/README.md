# cuSOLVER Standard Non-Hermitian Dense Eigenvalue solver example

## Description

This chapter provides three examples that demonstrate the usage of cuSOLVER geev for computing the eigenvalues _&lambda;<sub>k</sub>_ and corresponding eigenvectors _x<sub>k</sub>_ of a non-Hermitian matrix _**A**_ such that

_**A** x<sub>k</sub> = &lambda;<sub>k</sub> x<sub>k</sub>_.


Example 1 and Example 2 compute the eigenvalues and right eigenvectors of the real 3x3 matrix
```
A = | 1.0 | 2.0 | -3.0 |
    | 7.0 | 4.0 | -2.0 |
    | 4.0 | 2.0 |  1.0 |.
```
The matrix has one real eigenvalue and two complex eigenvalues that appear as conjugate pair. Example 1 computes the eigenpairs using double precision for all types. Example 2 demonstrates the usage of the mixed real-complex interface.

Example 3 computes the eigenvalues and right eigenvectors of the complex 3x3 matrix
```
A = | 2.00 + 1.00j | -1.00 + 0.00j | 1.00 + 2.00j |
    | 2.00 + 1.00j | -3.00 + 1.00j | 2.00 + 3.00j |
    | 1.00 + 2.00j | -1.00 + 2.00j | 0.00 + 1.00j |.
```


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
- [cusolverDnXgeev_bufferSize API](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxgeev)
- [cusolverDnXgeev API](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxgeev)

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
$ Open cusolver_examples.sln project in Visual Studio and build
```

# Example 1
```
$  ./cusolver_geev_example1
```

Sample example output:
```
A = (matlab base -1) 
1.00 2.00 -3.00 
7.00 4.00 -2.00 
4.00 2.00 1.00 
=====
info = 0 : Xgeev converged
eigenvalues = (matlab base -1) 
WR[1] + 1i* WI[1] = -9.209319E-01 + 1i*0.000000E+00
WR[2] + 1i* WI[2] = 3.460466E+00 + 1i*2.323549E+00
WR[3] + 1i* WI[3] = 3.460466E+00 + 1i*-2.323549E+00
=====
VR = (matlab base -1) 
0.52 0.10 0.19 
-0.83 0.76 0.00 
-0.22 0.57 -0.23 
=====
|A*VR - VR*diag(W)| = 4.055924E-15 
```

# Example 2
```
$  ./cusolver_geev_example2
```

Sample example output:
```
A = (matlab base -1) 
1.00 2.00 -3.00 
7.00 4.00 -2.00 
4.00 2.00 1.00 
=====
info = 0 : Xgeev converged
eigenvalues = (matlab base -1) 
-0.92 + 0.00j 
3.46 + 2.32j 
3.46 + -2.32j 
=====
VR = 
0.52 0.10 0.19 
-0.83 0.76 0.00 
-0.22 0.57 -0.23 
=====
```

# Example 3
```
$  ./cusolver_geev_example3
```

Sample example output:

```
A = (matlab base -1) 
2.00 + 1.00j -1.00 + 0.00j 1.00 + 2.00j 
2.00 + 1.00j -3.00 + 1.00j 2.00 + 3.00j 
1.00 + 2.00j -1.00 + 2.00j 0.00 + 1.00j 
=====
info = 0 : Xgeev converged
eigenvalues = (matlab base -1) 
0.14 + 3.80j 
0.22 + 0.85j 
-1.37 + -1.66j 
=====
VR = 
0.33 + -0.20j -0.13 + -0.57j -0.33 + -0.01j 
0.67 + 0.00j 0.47 + 0.20j -0.36 + 0.51j 
0.64 + 0.01j 0.62 + 0.00j 0.71 + 0.00j 
=====
|A*VR - VR*diag(W)| = 2.970989E-15 
```