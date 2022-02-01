# Biconjugate Gradient Stabilized Method (BiCGStab)

## Description

This sample describes how to use the cuSPARSE and cuBLAS libraries to implement the Incomplete-LU preconditioned iterative method **BiCGStab**.

[cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/index.html)
[cuSPARSE Documentation](https://docs.nvidia.com/cuda/cusparse/index.html)

#### Introduction

The solution of large sparse linear systems is an important problem in computational mechanics, atmospheric modeling, geophysics, biology, circuit simulation, and many other applications in the field of computational science and engineering. In general, these linear systems can be solved using direct or preconditioned iterative methods. Although the direct methods are often more reliable, they usually have large memory requirements and do not scale well on massively parallel computer platforms.

The iterative methods are more amenable to parallelism and therefore can be used to solve larger problems. Currently, the most popular iterative schemes belong to the Krylov subspace family of methods. They include Bi-Conjugate Gradient Stabilized (BiCGStab) and Conjugate Gradient (CG) iterative methods for non-symmetric and symmetric positive definite (s.p.d.) linear systems. We describe the BiCGStab method in more detail in the next section.

In practice, we typically use a variety of preconditioning techniques to improve the convergence of the iterative methods. In this sample, we focus on the Incomplete-LU which is one of the most popular of these preconditioning techniques. It computes an incomplete factorization of the coefficient matrix and requires a solution of lower and upper triangular linear systems in every iteration of the iterative method.

In order to implement the preconditioned BiCGStab, we use the sparse matrix-vector multiplication and the sparse triangular solve implemented in the cuSPARSE library. We point out that the parallelism available in these iterative methods depends highly on the sparsity pattern of the coefficient matrix at hand.

Notice that in every iteration of the incomplete-LU preconditioned BiCGStab iterative method, we need to perform two sparse matrix-vector multiplications and four triangular solves. The corresponding BiCGStab code using the cuSPARSE and cuBLAS libraries in the C programming language is shown below.

#### Preconditioned BiCGStab

<center>
<img src="BiCGStab.png" alt="drawing" width="500"/>

the pdf version is also available [here](./BigCGStab.pdf)

the code contains the line references to the above algorithm
</center>

## Building

* Linux
    ```bash
    make
    ```

* Windows/Linux
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```
    On Windows, instead of running the last build step, open the Visual Studio Solution that was created and build.

## Support

* **Supported SM Architectures:** SM 3.5, SM 3.7, SM 5.0, SM 5.2, SM 5.3, SM 6.0, SM 6.1, SM 6.2, SM 7.0, SM 7.2, SM 7.5, SM 8.0, SM 8.6
* **Supported OSes:** Linux, Windows, QNX, Android
* **Supported CPU Architectures**: x86_64, ppc64le, arm64
* **Supported Compilers**: gcc, clang, Intel icc, IBM xlc, Microsoft msvc, Nvidia HPC SDK nvc
* **Language**: `C99`

## Prerequisites

* [CUDA 11.3 toolkit](https://developer.nvidia.com/cuda-downloads) (or above) and compatible driver (see [CUDA Driver Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)).
* [CMake 3.9](https://cmake.org/download/) or above on Windows


## Usage
Input files -> https://sparse.tamu.edu/

```bash
$ wget https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/FEM_3D_thermal2.tar.gz
$ tar xf FEM_3D_thermal2.tar.gz
$ ./bicgstab_example FEM_3D_thermal2/FEM_3D_thermal2.mtx
```

Sample example output:

```
matrix name: FEM_3D_thermal2/FEM_3D_thermal2.mtx
num. rows:   147900
num. cols:   147900
nnz:         3489300
structure:   unsymmetric

Matrix parsing...
Testing BiCGStab
A(-1,-1) is missing
U(-1,-1) is zero
BiCGStab loop:
  Initial Residual: Norm 2.913525e+00' threshold 2.913525e-10
  Iteration = 1; Error Norm = 2.913525e+00
  Iteration = 2; Error Norm = 3.013470e-02
  Iteration = 3; Error Norm = 1.234002e-01
  Iteration = 4; Error Norm = 2.893113e-03
  Iteration = 5; Error Norm = 9.122037e-04
  Iteration = 6; Error Norm = 3.418270e-04
  Iteration = 7; Error Norm = 2.657349e-04
  Iteration = 8; Error Norm = 1.042150e-04
  Iteration = 9; Error Norm = 7.515743e-05
  Iteration = 10; Error Norm = 3.131400e-05
  Iteration = 11; Error Norm = 2.206263e-05
  Iteration = 12; Error Norm = 9.736421e-06
  Iteration = 13; Error Norm = 6.742005e-06
  Iteration = 14; Error Norm = 3.137997e-06
  Iteration = 15; Error Norm = 2.136315e-06
  Iteration = 16; Error Norm = 1.043664e-06
  Iteration = 17; Error Norm = 6.987928e-07
  Iteration = 18; Error Norm = 3.564072e-07
  Iteration = 19; Error Norm = 2.349326e-07
  Iteration = 20; Error Norm = 1.243880e-07
Check Solution
Final error norm = 8.084749e-08
```
