# Preconditioned Biconjugate Gradient Stabilized Method (BiCGStab)

## Description

This sample describes how to use the cuSPARSE and cuBLAS libraries to implement the Incomplete-LU preconditioned iterative method **BiCGStab**.

[cuSPARSE Documentation](https://docs.nvidia.com/cuda/cusparse/index.html)

[cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/index.html)

#### Introduction

The solution of large sparse linear systems is an important problem in computational mechanics, atmospheric modeling, geophysics, biology, circuit simulation, and many other applications in the field of computational science and engineering. In general, these linear systems can be solved using direct or preconditioned iterative methods. Although the direct methods are often more reliable, they usually have large memory requirements and do not scale well on massively parallel computer platforms.

The iterative methods are more amenable to parallelism and therefore can be used to solve larger problems. Currently, the most popular iterative schemes belong to the Krylov subspace family of methods. They include Bi-Conjugate Gradient Stabilized (BiCGStab) and Conjugate Gradient (CG) iterative methods for non-symmetric and symmetric positive definite (s.p.d.) linear systems. We describe the BiCGStab method in more detail in the next section.

In practice, we typically use a variety of preconditioning techniques to improve the convergence of the iterative methods. In this sample, we focus on the Incomplete-LU preconditioning which is one of the most popular of these preconditioning techniques. It computes an incomplete factorization of the coefficient matrix and requires a solution of lower and upper triangular linear systems in every iteration of the iterative method.

In order to implement the preconditioned BiCGStab, we use the sparse matrix-vector multiplication and the sparse triangular solve implemented in the cuSPARSE library. We point out that the parallelism available in these iterative methods depends highly on the sparsity pattern of the coefficient matrix at hand.

Notice that in every iteration of the incomplete-LU preconditioned BiCGStab iterative method, we need to perform two sparse matrix-vector multiplications and four triangular solves. The corresponding BiCGStab code using the cuSPARSE and cuBLAS libraries in the C programming language is shown below.

#### Preconditioned BiCGStab

<center>
<img src="BiCGStab.png" alt="drawing" width="500"/>

the pdf version is also available [here](./BigCGStab.pdf)

the code contains the line references to the above algorithm
</center>

## Building

* Command line
    ```bash
    gcc -I<cuda_toolkit_path>/include bicgstab_example.c -o bicgstab_example -lcudart -lcusparse -lcublas
    ```

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

* **Supported SM Architectures:** SM 3.5, SM 3.7, SM 5.0, SM 5.2, SM 5.3, SM 6.0, SM 6.1, SM 6.2, SM 7.0, SM 7.2, SM 7.5, SM 8.0, SM 8.6, SM 8.9, SM 9.0
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
$ wget https://suitesparse-collection-website.herokuapp.com/MM/Wissgott/parabolic_fem.tar.gz
$ tar xf parabolic_fem.tar.gz
$ ./bicgstab_example parabolic_fem/parabolic_fem.mtx
```

Sample example output:

```
matrix name: parabolic_fem/parabolic_fem.mtx
num. rows:   525825
num. cols:   525825
nnz:         4200450
structure:   symmetric

Matrix parsing...
Testing BiCGStab
BiCGStab loop:
  Initial Residual: Norm 2.170672e+02' threshold 2.170672e-08
  Iteration = 1; Error Norm = 2.170672e+02
  Iteration = 2; Error Norm = 2.361113e+01
  Iteration = 3; Error Norm = 1.319608e+02
  Iteration = 4; Error Norm = 3.670778e+01
  Iteration = 5; Error Norm = 3.488150e+01
  Iteration = 6; Error Norm = 1.719376e+01
  Iteration = 7; Error Norm = 1.463979e+01
  Iteration = 8; Error Norm = 8.805470e+00
  Iteration = 9; Error Norm = 7.035572e+00
  Iteration = 10; Error Norm = 4.633854e+00
  Iteration = 11; Error Norm = 3.568345e+00
  Iteration = 12; Error Norm = 2.466218e+00
  Iteration = 13; Error Norm = 1.862108e+00
  Iteration = 14; Error Norm = 1.324990e+00
  Iteration = 15; Error Norm = 9.914012e-01
  Iteration = 16; Error Norm = 7.191100e-01
  Iteration = 17; Error Norm = 5.366529e-01
  Iteration = 18; Error Norm = 3.945207e-01
  Iteration = 19; Error Norm = 2.947479e-01
  Iteration = 20; Error Norm = 2.188237e-01
Check Solution
Final error norm = 1.639804e-01
```
