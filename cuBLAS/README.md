# cuBLAS Library - APIs Examples

## Description

This folder demonstrates cuBLAS APIs usage.

[cuBLAS API Documentation](https://docs.nvidia.com/cuda/cublas/index.html)

## cuBLAS Samples

##### cuBLAS Level 1

* [cuBLAS amax](Level-1/amax/)

    The sample finds the (smallest) index of the element of the maximum magnitude.

* [cuBLAS amin](Level-1/amin/)

    The sample finds the (smallest) index of the element of the minimum magnitude.

* [cuBLAS asum](Level-1/asum/)

    The sample computes the sum of the absolute values of the elements of vector _x_.

* [cuBLAS axpy](Level-1/axpy/)

    The sample computes a vector-scalar product and adds the result to a vector.

* [cuBLAS copy](Level-1/copy/)

    The sample copies the vector _x_ into the vector _y_.

* [cuBLAS dot](Level-1/dot/)

    The sample applies the dot product to vector _x_ and _y_.

* [cuBLAS nrm2](Level-1/nrm2/)

    The sample computes the Euclidean norm of a vector.

* [cuBLAS rot](Level-1/rot/)

    The sample applies the Givens rotation matrix to vector _x_ and _y_.

* [cuBLAS rotg](Level-1/rotg/)

    The sample applies the Givens rotation matrix to vector _x_ and _y_.

* [cuBLAS rotm](Level-1/rotm/)

    The sample applies the modified Givens rotation matrix to vector _x_ and _y_.

* [cuBLAS rotmg](Level-1/rotmg/)

    The sample applies the modified Givens rotation matrix to vector _x_ and _y_.

* [cuBLAS scal](Level-1/scal/)

    The sample computes the product of a vector by a scalar.

* [cuBLAS swap](Level-1/swap/)

    The sample interchanges the elements of vector _x_ and _y_.

##### cuBLAS Level 2

* [cuBLAS gbmv](Level-2/gbmv/)

    The sample performs a banded matrix-vector multiplication.

* [cuBLAS gemv](Level-2/gemv/)

    The sample performs a matrix-vector multiplication.

* [cuBLAS ger](Level-2/ger/)

    The sample performs a rank-1 update .

* [cuBLAS sbmv](Level-2/sbmv/)

    The sample performs a symmetric banded matrix-vector multiplication.

* [cuBLAS spmv](Level-2/spmv/)

    The sample performs a performs the symmetric packed matrix-vector multiplication.

* [cuBLAS spr](Level-2/spr/)

    The sample performs a packed symmetric rank-1 update.

* [cuBLAS spr2](Level-2/spr2/)

    The sample performs a packed symmetric rank-2 update.

* [cuBLAS symv](Level-2/symv/)

    The sample performs a symmetric matrix-vector multiplication.

* [cuBLAS syr](Level-2/syr/)

    The sample performs a symmetric rank-1 update.

* [cuBLAS syr2](Level-2/syr2/)

    The sample performs a symmetric rank-2 update.

* [cuBLAS tbmv](Level-2/tbmv/)

    The sample performs a triangular banded matrix-vector multiplication.

* [cuBLAS tbsv](Level-2/tbsv/)

    The sample solves a triangular banded linear system with a single right-hand-side.

* [cuBLAS tpmv](Level-2/tpmv/)

    The sample performs a triangular packed matrix-vector multiplication.

* [cuBLAS tpsv](Level-2/tpsv/)

    The sample solves a packed triangular linear system with a single right-hand-side.

* [cuBLAS trmv](Level-2/trmv/)

    The sample performs a triangular matrix-vector multiplication.

* [cuBLAS trsv](Level-2/trsv/)

    The sample solves a triangular linear system with a single right-hand-side.

* [cuBLAS hemv](Level-2/hemv/)

    The sample performs a Hermitian matrix-vector multiplication.

* [cuBLAS hbmv](Level-2/hbmv/)

    The sample performs a Hermitian banded matrix-vector multiplication.

* [cuBLAS hpmv](Level-2/hpmv/)

    The sample performs a Hermitian packed matrix-vector multiplication.

* [cuBLAS her](Level-2/her/)

    The sample performs a Hermitian rank-1 update.

* [cuBLAS her2](Level-2/her2/)

    The sample performs a Hermitian rank-2 update.

* [cuBLAS hpr](Level-2/hpr/)

    The sample performs a packed Hermitian rank-1 update.

* [cuBLAS hpr2](Level-2/hpr2/)

    The sample performs a packed Hermitian rank-2 update.

##### cuBLAS Level 3

* [cuBLAS gemm](Level-3/gemm/)

    The sample computes a matrix-matrix product with general matrices.

* [cuBLAS gemm3m](Level-3/gemm3m/)

    The sample computes matrix-matrix product with general matrices, using the Gauss complexity reduction algorithm.

* [cuBLAS gemmBatched](Level-3/gemmBatched/)

    The sample computes batches of matrix-matrix product with general matrices.

* [cuBLAS gemmStridedBatched](Level-3/gemmStridedBatched/)

    The sample computes strided batches of matrix-matrix product with general matrices.

* [cuBLAS hemm](Level-3/hemm/)

    The sample computes a Hermitian matrix-matrix product.

* [cuBLAS herk](Level-3/herk/)

    The sample computes a Hermitian rank-k update.

* [cuBLAS her2k](Level-3/her2k/)

    The sample computes a Hermitian rank-2k update.

* [cuBLAS herkx](Level-3/herkx/)

    The sample computes a variation of Hermitian rank-2k update.

* [cuBLAS symm](Level-3/symm/)

    The sample computes a symmetric matrix-matrix product.

* [cuBLAS syrk](Level-3/syrk/)

    The sample computes a symmetric rank-k update.

* [cuBLAS syrk](Level-3/syr2k/)

    The sample computes a symmetric rank-2k update.

* [cuBLAS syrk](Level-3/syrkx/)

    The sample computes a variation of symmetric rank-2k update.

* [cuBLAS trmm](Level-3/trmm/)

    The sample computes a triangular matrix-matrix product.

* [cuBLAS trsm](Level-3/trsm/)

    The sample computes a triangular linear system with multiple right-hand-sides.

* [cuBLAS trsmBatched](Level-3/trsmBatched/)

    The sample computes batched triangular linear systems with multiple right-hand-sides.

##### cuBLAS Extensions

* [cuBLAS geam](Extensions/geam/)

    The sample computes a matrix-matrix addition/transposition.

* [cuBLAS dgmm](Extensions/dgmm/)

    The sample computes a matrix-matrix multiplication.

* [cuBLAS tpttr](Extensions/tpttr/)

    The sample computes a conversion from the triangular packed format to the triangular format.

* [cuBLAS trttp](Extensions/trttp/)

    The sample computes a conversion from the triangular format to the triangular packed format.

* [cuBLAS AxpyEx](Extensions/AxpyEx/)

    The sample computes a vector-scalar product and adds the result to a vector.

* [cuBLAS Cherk3mEx](Extensions/Cherk3mEx/)

    The sample computes a Hermitian rank-k update, using the Gauss complexity reduction algorithm.

* [cuBLAS CherkEx](Extensions/CherkEx/)

    The sample computes a Hermitian rank-k update.

* [cuBLAS Csyrk3mEx](Extensions/Csyrk3mEx/)

    The sample computes a symmetric rank-k update, using the Gauss complexity reduction algorithm.

* [cuBLAS CsyrkEx](Extensions/CsyrkEx/)

    The sample computes a symmetric rank-k update.

* [cuBLAS DotEx](Extensions/DotEx/)

    The sample applies the dot product to vector _x_ and _y_.

* [cuBLAS GemmEx](Extensions/GemmEx/)

    The sample computes a matrix-matrix product with general matrices.

* [cuBLAS GemmBatchedEx](Extensions/GemmBatchedEx/)

    The sample computes batches of matrix-matrix product with general matrices.

* [cuBLAS GemmStridedBatchedEx](Extensions/GemmStridedBatchedEx/)

    The sample computes strided batches of matrix-matrix product with general matrices.

* [cuBLAS Nrm2Ex](Extensions/Nrm2Ex/)

    The sample computes the Euclidean norm of a vector.

* [cuBLAS RotEx](Extensions/RotEx/)

    The sample applies the Givens rotation matrix to vector _x_ and _y_.

* [cuBLAS ScalEx](Extensions/ScalEx/)

    The sample computes the product of a vector by a scalar.
