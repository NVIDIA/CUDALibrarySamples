# cuSPARSE Library - Generic APIs Examples

## Description

This folder demonstrates cuSPARSE Generic APIs usage.

[cuSPARSE Generic APIs Documentation](https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-api-reference)

## cuSPARSE Samples

##### Vector - Vector Operations

* [cusparseAxpby](axpby/)

    The sample demonstrates *sparse vector - dense vector scaling and sum*

* [cusparseGather](gather/)

    The sample demonstrates *dense vector to sparse vector element gathering*

* [cusparseRot](rot/)

    The sample demonstrates *sparse vector - dense vector Givens rotation*

* [cusparseScatter](scatter/)

    The sample demonstrates *sparse vector to dense vector element scattering*

* [cusparseSpVV](spvv/)

    The sample demonstrates *sparse vector - dense vector dot product*

##### Matrix - Vector Operations

* [cusparseSpMV CSR](spmm_csr/)

    The sample demonstrates *sparse matrix - dense vector multiplication*, where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format

* [cusparseSpMV COO](spmv_coo/)

    The sample demonstrates *sparse matrix - dense vector multiplication*, where the sparse matrix is represented in COO (Coordinate) storage format

##### Matrix - Matrix Operations

* [cusparseSpMM CSR](spmm_csr/)

    The sample demonstrates *sparse matrix - dense matrix multiplication*, where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format

* [cusparseSpMM COO](spmm_coo/)

    The sample demonstrates *sparse matrix - dense matrix multiplication*, where the sparse matrix is represented in COO (Coordinate) storage format

* [cusparseSpGEMM](spgemm/)

    The sample demonstrates *sparse matrix - sparse matrix multiplication*, where all operands are sparse matrices represented in CSR (Compressed Sparse Row) storage format

##### Conversion

* [cusparseSparseToDense](sparse2dense/)

    The sample demonstrates *sparse matrix to dense matrix conversion*, where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format

* [cusparseDenseToSparse](dense2sparse/)

    The sample demonstrates *dense matrix to sparse matrix conversion*, where the sparse matrix is represented in CSR (Compressed Sparse Row) storage format

##### Optimizations

* [CUDA Graph Capture](graph_capture/)

    The sample demonstrates how to optimize *sparse vector - dense vector dot product* (`cusparseSpVV`) by exploting *CUDA Graph Capture functionality*

* [Hardware Memory Compression](compression/)

    The sample demonstrates how to optimize *sparse vector - dense vector scaling and sum* (`cusparseAxpby`) by exploting NVIDIA Ampere architecture *Hardware Memory Compression*
