# cuSOLVER Library - APIs Examples

## Description

This folder demonstrates cuSOLVER APIs usage.

[cuSOLVER API Documentation](https://docs.nvidia.com/cuda/cusolver/index.html)

## cuSOLVER Samples

##### MutliGPU LU Decomposition example

* [cuSOLVER MgGetrs](MgGetrs/)

    The sample solves linear system by *LU Decomposition*, with partial pivoting (`getrf` and `getrs`). See example for detailed description.

##### MutliGPU Cholesky Decomposition example

* [cuSOLVER MgPotrf](MgPotrf/)

    The sample solves linear system by *Cholesky factorization* (`potrf` and `potrs`). See example for detailed description.

##### MutliGPU Symmetric Eigenvalue solver examples

* [cuSOLVER MgSyevd](MgSyevd/)

    The sample provides three examples to demonstrate *multiGPU standard symmetric eigenvalue* solver. See example for detailed description.

##### 64-bit Singular Value Decomposition example

* [cuSOLVER Xgesvd](Xgesvd/)

    The sample computes *singular value decomposition*, using 64-bit APIs. See example for detailed description.

##### 64-bit Singular Value + Polar Decomposition example

* [cuSOLVER Xgesvdp](Xgesvdp/)

    The sample computes *singular value decomposition*, in combination with polar decomposition, using 64-bit APIs. See example for detailed description.

##### 64-bit Rank-K Singular Value Decomposition example

* [cuSOLVER Xgesvdr](Xgesvdr/)

    The sample computes approximated rank-k *singular value decomposition*, using 64-bit APIs. See example for detailed description.

##### 64-bit LU Decomposition example

* [cuSOLVER Xgetrf](Xgetrf/)

    The sample solves linear system by *LU Decomposition*, with partial pivoting (`getrf` and `getrs`), using 64-bit APIs. See example for detailed description.

##### 64-bit Standard Symmetric Dense Eigenvalue solver example

* [cuSOLVER Xsyevd](Xsyevd/)

    The sample demonstrates *standard symmetric eigenvalue* solver, using 64-bit APIs. See example for detailed description.

##### 64-bit Standard Symmetric Dense Eigenvalue solver example

* [cuSOLVER Xsyevdx](Xsyevdx/)

    The sample demonstrates *standard symmetric eigenvalue* solver, using 64-bit APIs. See example for detailed description.

##### Batched Sparse QR Factorizaion examples

* [cuSOLVER csrqr](csrqr/)

    The sample provides two examples to demonstrate batched *sparse qr factorization*. See example for detailed description.

##### Iterative Refinement solver example

* [cuSOLVER gesv](gesv/)

    The sample demonstrates *iterative refinement solver example* for solving linear systems with multiple right hand sides. See example for detailed description.

##### Singular Value Decomposition example

* [cuSOLVER gesvd](gesvd/)

    The sample computes *singular value decomposition*. See example for detailed description.

##### Singular Value Decomposition example

* [cuSOLVER gesvdaStridedBatched](gesvdaStridedBatched/)

    The sample computes a strided, batched approximate *singular value decomposition*. See example for detailed description.

##### Singular Value Decomposition (via Jacobi method) example

* [cuSOLVER gesvdj](gesvdj/)

    The sample computes *singular value decomposition*, via Jacobi method. See example for detailed description.

##### Batched Singular Value Decomposition (via Jacobi method) example

* [cuSOLVER gesvdjBatched](gesvdjBatched/)

    The sample computes batched *singular value decomposition*, via Jacobi method. See example for detailed description.

##### LU Decomposition example

* [cuSOLVER getrf](getrf/)

    The sample solves linear system by *LU Decomposition*, with partial pivoting (`getrf` and `getrs`). See example for detailed description.

##### Orthgonalization example

* [cuSOLVER orgqr](orgqr/)

    The sample computes complete *orthgonalization* (`geqrf` and `orgqr`). See example for detailed description.

##### QR Factorization example

* [cuSOLVER ormqr](ormqr/)

    The sample computes dense *QR orthgonalization* (`geqrf`, `ormqr`, and `trsm`). See example for detailed description.

##### Batched Cholesky Decomposition example

* [cuSOLVER potrfBatched](potrfBatched/)

    The sample solves linear system by *Cholesky factorization* (`potrfBatched` and `potrsBatched`). See example for detailed description.

##### Standard Symmetric Dense Eigenvalue solver example

* [cuSOLVER syevd](syevd/)

    The sample demonstrates *standard symmetric eigenvalue* solver. See example for detailed description.

##### Standard Symmetric Dense Eigenvalue solver (via Jacobi method) example

* [cuSOLVER syevj](syevj/)

    The sample demonstrates *standard symmetric eigenvalue* solver, via Jacobi method. See example for detailed description.

##### Batched Standard Symmetric Dense Eigenvaluesolver (via Jacobi method)  example

* [cuSOLVER syevjBatched](syevjBatched/)

    The sample demonstrates batched *standard symmetric eigenvalue* solver, via Jacobi method. See example for detailed description.

##### Generalized Symmetric-Definite Dense Eigenvalue solver example

* [cuSOLVER sygvd](sygvd/)

    The sample demonstrates *generalized symmetric-definite dense eigenvalue* solver. See example for detailed description.

##### Generalized Symmetric-Definite Dense Eigenvalue solver (via Jacobi method) example

* [cuSOLVER sygvj](sygvj/)

    The sample demonstrates *generalized symmetric-definite dense eigenvalue* solver, (via Jacobi method). See example for detailed description.

##### Triangular Matrix Inversion Computation example

* [cuSOLVER trtri](trtri/)

    The sample preforms *triangular matrix mnversion*. See example for detailed description.