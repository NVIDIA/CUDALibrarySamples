# cuSPARSE Library - Generic APIs Examples

## Description

This folder demonstrates cuSPARSE Generic APIs usage.

[cuSPARSE Generic APIs Documentation](https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-api-reference)

## cuSPARSE Samples

- [Vector - Vector Operations](#vector-vector-operations)
- [Matrix - Vector Operations](#matrix-vector-operations)
- [Matrix - Matrix Operations](#matrix-matrix-operations)
- [Conversion](#conversion)
- [Legacy APIs](#legacy-apis)
- [Optimizations](#optimizations)
- [Sparse Iterative Methods](#sparse-iterative-methods)

---

#### Vector - Vector Operations

* [cusparseAxpby](axpby/)

    The sample demonstrates *Sparse Vector - Dense Vector scaling and sum*

* [cusparseGather](gather/)

    The sample demonstrates *Dense Vector to Sparse Vector element gathering*

* [cusparseRot](rot/)

    The sample demonstrates *Sparse Vector - Dense Vector Givens rotation*

* [cusparseScatter](scatter/)

    The sample demonstrates *Sparse Vector to Dense Vector element scattering*

* [cusparseSpVV](spvv/)

    The sample demonstrates *Sparse Vector - Dense Vector dot product*

#### Matrix - Vector Operations

* [cusparseSpMV CSR](spmv_csr/)

    The sample demonstrates *Sparse Matrix - Dense Vector multiplication*, where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format

* [cusparseSpMV COO](spmv_coo/)

    The sample demonstrates *Sparse Matrix - Dense Vector multiplication*, where the sparse matrix is represented in COO (Coordinate) storage format

* [cusparseSpSV CSR](spsv_csr/)

    The sample demonstrates *Sparse triangular solver with single right-hand side*, where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format

* [cusparseSpSV COO](spsv_coo/)

    The sample demonstrates *Sparse triangular solver with single right-hand side*, where the sparse matrix is represented in COO (Coordinate) storage format

#### Matrix - Matrix Operations

* [cusparseSpMM CSR](spmm_csr/)

    The sample demonstrates *Sparse Matrix - Dense Matrix multiplication = Dense Matrix*, where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format

* [cusparseSpMM CSR - Batched](spmm_csr_batched/)

    The sample demonstrates *Batched Sparse Matrix - Dense Matrix multiplication = Dense Matrix* , where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format

* [cusparseSpMMOp CSR](spmm_csr_op/)

    The sample demonstrates *Sparse Matrix - Dense Matrix multiplication = Dense Matrix* *with Custom Operators*, where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format

* [cusparseSpMM COO](spmm_coo/)

    The sample demonstrates *Sparse Matrix - Dense Matrix multiplication = Dense Matrix*, where the sparse matrix is represented in COO (Coordinate) storage format

* [cusparseSpMM COO - Batched](spmm_coo_batched/)

    The sample demonstrates *Batched Sparse Matrix - Dense Matrix multiplication = Dense Matrix*, where the sparse matrix is represented in COO (Coordinate) storage format

* [cusparseSpMM SDDMM](sddmm_csr/)

    The sample demonstrates *Dense Matrix - Dense Matrix multiplication = Sparse Matrix*, where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format

* [cusparseSpMM Blocked-ELL](spmm_blockedell/)

    The sample demonstrates *Sparse Matrix - Dense Matrix multiplication = Dense Matrix*, where the sparse matrix is represented in Blocked-ELL storage format

* [cusparseSpMM SDDMM - Batched](sddmm_csr_batched/)
    The sample demonstrates *Batched Dense matrix - Dense matrix multiplication = Sparse matrix*, where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format

* [cusparseSpGEMM](spgemm/)

    The sample demonstrates *Sparse Matrix - Sparse Matrix multiplication = Sparse Matrix*, where all operands are sparse matrices represented in CSR (Compressed Sparse Row) storage format

* [cusparseSpGEMM - Memory Optmization](spgemm_mem/)

    The sample demonstrates *Sparse Matrix - Sparse Matrix multiplication = Sparse Matrix*, where all operands are sparse matrices represented in CSR (Compressed Sparse Row) storage format and the memory usage is optimzed

* [cusparseSpGEMM_reuse](spgemm_reuse/)

    The sample demonstrates *Sparse Matrix - Sparse Matrix multiplication = Sparse Matrix*, where all operands are sparse matrices represented in CSR (Compressed Sparse Row) storage format and the structure of the output matrix can be reused multiple times

* [cusparseSpSM CSR](spsm_csr/)

    The sample demonstrates *Sparse triangular solver with multiple right-hand sides*, where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format

* [cusparseSpSM COO](spsm_coo/)

    The sample demonstrates *Sparse triangular solver with multiple right-hand sides*, where the sparse matrix is represented in COO (Coordinate) storage format

#### Conversion

* [cusparseSparseToDense](sparse2dense_csr/)

    The sample demonstrates *Sparse Matrix to Dense Matrix conversion*, where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format

* [cusparseDenseToSparse](dense2sparse_csr/)

    The sample demonstrates *Dense Matrix to Sparse Matrix conversion*, where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format

* [cusparseDenseToSparse (Blocked-ELL)](dense2sparse_blockedell/)

    The sample demonstrates *Dense Matrix to Sparse Matrix conversion*, where the sparse matrix is represented in Blocked-Ellpack storage format

#### Legacy APIs

* [cusparseXcoosortByRow](coosort/)

    The sample demonstrates how to sort a COO format matrix

* [cusparseSgpsvInterleavedBatch](gpsvInterleavedBatch/)

    The sample demonstrates how to solves two penta-diagonal systems with NOT interleaved format

#### Optimizations

* [CUDA Graph Capture](graph_capture/)

    The sample demonstrates how to optimize *Sparse vector - Dense vector dot product* (`cusparseSpVV`) by exploiting *CUDA Graph Capture functionality*

* [Hardware Memory Compression](compression/)

    The sample demonstrates how to optimize *Sparse vector - Dense vector scaling and sum* (`cusparseAxpby`) by exploiting NVIDIA Ampere architecture *Hardware Memory Compression*

#### Sparse Iterative Methods

* [Preconditioned CG](cg/)

    The sample describes how to use the cuSPARSE and cuBLAS libraries to implement the Incomplete-Cholesky preconditioned iterative *Conjugate Gradient (CG)*

* [Preconditioned BiCGStab](bicgstab/)

    The sample describes how to use the cuSPARSE and cuBLAS libraries to implement the Incomplete-LU preconditioned iterative *Biconjugate Gradient Stabilized Method (BiCGStab)*
