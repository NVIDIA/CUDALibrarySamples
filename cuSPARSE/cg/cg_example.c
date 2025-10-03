/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>  // fopen
#include <stdlib.h> // EXIT_FAILURE
#include <string.h> // strtok
#include <assert.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#if defined(NDEBUG)
#   define PRINT_INFO(var)
#else
#   define PRINT_INFO(var) printf("  " #var ": %f\n", var);
#endif

typedef struct VecStruct {
    cusparseDnVecDescr_t vec;
    double*              ptr;
} Vec;

//==============================================================================

/// A 5-point Laplacian on a g x g grid with Dirichlet boundary conditions.
/// This code allocates. The caller must free.
void make_laplace_matrix(int * n_out,
                         int **row_offsets_out, 
                         int **columns_out, 
                         double **values_out) {
    int grid = 700; // grid resolution

    int n = grid * grid;
    *n_out = n;
    // vertices have 5 neighbors, 
    // but each vertex on the boundary loses 1. corners lose 2.
    int nnz = 5 * n - 4 * grid;

    printf("Creating 5-point time-dependent diffusion matrix.\n"
           " grid size: %d x %d\n"
           " matrix rows:   %d\n"
           " matrix cols:   %d\n"
           " nnz:         %d\n",
           grid, grid, n, n, nnz);

    int* row_offsets = *row_offsets_out = (int*)malloc((n + 1) * sizeof(int));
    int* columns     = *columns_out     = (int*)malloc(nnz * sizeof(int));
    double* values   = *values_out      = (double*)malloc(nnz * sizeof(double));
    assert(row_offsets);
    assert(columns);
    assert(values);

    // The Laplacian stencil looks like [-1;-1,4,-1;-1].
    // ICHOL doesn't work great with that stencil.
    // ICHOL is better suited when there's some more mass on the diagonal.
    double mass = 0.04;

    int it = 0; // next unused index into `columns`/`values`

#define INSERT(u,v, x)                    \
    if(0<=(u) && (u)<grid &&              \
       0<=(v) && (v)<grid)                \
    {                                     \
        columns[it] = ((u) * grid + (v)); \
        values[it] = x;                   \
        ++it;                             \
    }

    int row = 0;
    row_offsets[row] = 0;
    for (int i = 0; i < grid; ++i) {
        for (int j = 0; j < grid; ++j)
        {
            INSERT(i - 1, j    , -1.0);
            INSERT(i    , j - 1, -1.0);
            INSERT(i    , j    ,  4.0 + mass);
            INSERT(i    , j + 1, -1.0);
            INSERT(i + 1, j    , -1.0);
            row_offsets[++row] = it;
        }
    }
    assert(it == nnz);
#undef INSERT
}

//==============================================================================

int gpu_CG(cublasHandle_t       cublasHandle,
           cusparseHandle_t     cusparseHandle,
           int                  m,
           cusparseSpMatDescr_t matA,
           cusparseSpMatDescr_t matL,
           Vec                  d_B,
           Vec                  d_X,
           Vec                  d_R,
           Vec                  d_R_aux,
           Vec                  d_P,
           Vec                  d_T,
           Vec                  d_tmp,
           void*                d_bufferMV,
           int                  maxIterations,
           double               tolerance) {
    const double zero      = 0.0;
    const double one       = 1.0;
    const double minus_one = -1.0;
    //--------------------------------------------------------------------------
    // ### 1 ### R0 = b - A * X0 (using initial guess in X)
    //    (a) copy b in R0
    CHECK_CUDA( cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) )
    //    (b) compute R = -A * X0 + R
    CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &minus_one, matA, d_X.vec, &one, d_R.vec,
                                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                 d_bufferMV) )
    //--------------------------------------------------------------------------
    // ### 2 ### R_i_aux = L^-1 L^-T R_i
    size_t              bufferSizeL, bufferSizeLT;
    void*               d_bufferL, *d_bufferLT;
    cusparseSpSVDescr_t spsvDescrL, spsvDescrLT;
    //    (a) L^-1 tmp => R_i_aux    (triangular solver)
    CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsvDescrL) )
    CHECK_CUSPARSE( cusparseSpSV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL) )
    CHECK_CUDA( cudaMalloc(&d_bufferL, bufferSizeL) )
    CHECK_CUSPARSE( cusparseSpSV_analysis(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL) )
    CHECK_CUDA( cudaMemset(d_tmp.ptr, 0x0, m * sizeof(double)) )
    CHECK_CUSPARSE( cusparseSpSV_solve(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL) )

    //    (b) L^-T R_i => tmp    (triangular solver)
    CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsvDescrLT) )
    CHECK_CUSPARSE( cusparseSpSV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                        &one, matL, d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, &bufferSizeLT) )
    CHECK_CUDA( cudaMalloc(&d_bufferLT, bufferSizeLT) )
    CHECK_CUSPARSE( cusparseSpSV_analysis(
                        cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                        &one, matL, d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, d_bufferLT) )
    CHECK_CUDA( cudaMemset(d_R_aux.ptr, 0x0, m * sizeof(double)) )
    CHECK_CUSPARSE( cusparseSpSV_solve(
                        cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                        &one, matL, d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT) )
    //--------------------------------------------------------------------------
    // ### 3 ### P0 = R0_aux
    CHECK_CUDA( cudaMemcpy(d_P.ptr, d_R_aux.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) )
    //--------------------------------------------------------------------------
    // nrm_R0 = ||R||
    double nrm_R;
    CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) )
    double threshold = tolerance * nrm_R;
    printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
    //--------------------------------------------------------------------------
    double delta;
    CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R_aux.ptr, 1, &delta) )
    //--------------------------------------------------------------------------
    // ### 4 ### repeat until convergence based on max iterations and
    //           and relative residual
    for (int i = 0; i < maxIterations; i++) {
        printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
        //----------------------------------------------------------------------
        // ### 5 ### alpha = (R_i, R_aux_i) / (A * P_i, P_i)
        //     (a) T  = A * P_i
        CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                     matA, d_P.vec, &zero, d_T.vec,
                                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                     d_bufferMV) )
        //     (b) denominator = (T, P_i)
        double denominator;
        CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_T.ptr, 1, d_P.ptr, 1,
                                 &denominator) )
        //     (c) alpha = delta / denominator
        double alpha = delta / denominator;
        PRINT_INFO(delta)
        PRINT_INFO(denominator)
        PRINT_INFO(alpha)
        //----------------------------------------------------------------------
        // ### 6 ###  X_i+1 = X_i + alpha * P
        //    (a) X_i+1 = -alpha * T + X_i
        CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &alpha, d_P.ptr, 1,
                                  d_X.ptr, 1) )
        //----------------------------------------------------------------------
        // ### 7 ###  R_i+1 = R_i - alpha * (A * P)
        //    (a) R_i+1 = -alpha * T + R_i
        double minus_alpha = -alpha;
        CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &minus_alpha, d_T.ptr, 1,
                                  d_R.ptr, 1) )
        //----------------------------------------------------------------------
        // ### 8 ###  check ||R_i+1|| < threshold
        CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) )
        PRINT_INFO(nrm_R)
        if (nrm_R < threshold)
            break;
        //----------------------------------------------------------------------
        // ### 9 ### R_aux_i+1 = L^-1 L^-T R_i+1
        //    (a) L^-1 R_i+1 => tmp    (triangular solver)
        CHECK_CUDA( cudaMemset(d_tmp.ptr,   0x0, m * sizeof(double)) )
        CHECK_CUDA( cudaMemset(d_R_aux.ptr, 0x0, m * sizeof(double)) )
        CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matL, d_R.vec, d_tmp.vec,
                                           CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrL) )
        //    (b) L^-T tmp => R_aux_i+1    (triangular solver)
        CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_TRANSPOSE,
                                           &one, matL, d_tmp.vec,
                                           d_R_aux.vec, CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrLT) )
        //----------------------------------------------------------------------
        // ### 10 ### beta = (R_i+1, R_aux_i+1) / (R_i, R_aux_i)
        //    (a) delta_new => (R_i+1, R_aux_i+1)
        double delta_new;
        CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R_aux.ptr, 1,
                                 &delta_new) )
        //    (b) beta => delta_new / delta
        double beta = delta_new / delta;
        PRINT_INFO(delta_new)
        PRINT_INFO(beta)
        delta       = delta_new;
        //----------------------------------------------------------------------
        // ### 11 ###  P_i+1 = R_aux_i+1 + beta * P_i
        //    (a) P = beta * P
        CHECK_CUBLAS(cublasDscal(cublasHandle, m, &beta, d_P.ptr, 1))
        //    (b) P = R_aux + P
        CHECK_CUBLAS(
            cublasDaxpy(cublasHandle, m, &one, d_R_aux.ptr, 1, d_P.ptr, 1))
    }
    //--------------------------------------------------------------------------
    printf("Check Solution\n"); // ||R = b - A * X||
    //    (a) copy b in R
    CHECK_CUDA( cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) )
    // R = -A * X + R
    CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one,
                                 matA, d_X.vec, &one, d_R.vec, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV) )
    // check ||R||
    CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) )
    printf("Final error norm = %e\n", nrm_R);
    //--------------------------------------------------------------------------
    CHECK_CUSPARSE( cusparseSpSV_destroyDescr(spsvDescrL) )
    CHECK_CUSPARSE( cusparseSpSV_destroyDescr(spsvDescrLT) )
    CHECK_CUDA( cudaFree(d_bufferL) )
    CHECK_CUDA( cudaFree(d_bufferLT) )
    return EXIT_SUCCESS;
}

//==============================================================================
//==============================================================================

int main(int argc, char** argv) {
    const int    maxIterations = 10000;
    const double tolerance     = 1e-8f;
    if (argc != 1) {
        printf("Wrong number of command line arguments. cg_example accepts no arguments.\n");
        return EXIT_FAILURE;
    }
    int     base        = 0;
    int     m           = -1;
    int*    h_A_rows    = NULL;
    int*    h_A_columns = NULL;
    double* h_A_values  = NULL;
    make_laplace_matrix(&m, &h_A_rows, &h_A_columns, &h_A_values);
    int num_offsets = m + 1;
    int nnz = h_A_rows[m];
    double* h_X = (double*)malloc(m * sizeof(double));

    printf("Testing CG\n");
    for (int i = 0; i < m; i++)
        h_X[i] = 1.0;
    //--------------------------------------------------------------------------
    // ### Device memory management ###
    int*    d_A_rows, *d_A_columns;
    double* d_A_values, *d_L_values;
    Vec     d_B, d_X, d_R, d_R_aux, d_P, d_T, d_tmp;

    // allocate device memory for CSR matrices
    CHECK_CUDA( cudaMalloc((void**) &d_A_rows,    num_offsets * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_A_columns, nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_A_values,  nnz * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_L_values,  nnz * sizeof(double)) )

    CHECK_CUDA( cudaMalloc((void**) &d_B.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_X.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_R.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_R_aux.ptr, m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_P.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_T.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_tmp.ptr,   m * sizeof(double)) )

    // copy the CSR matrices and vectors into device memory
    CHECK_CUDA( cudaMemcpy(d_A_rows, h_A_rows, num_offsets * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_A_columns, h_A_columns, nnz *  sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_A_values, h_A_values, nnz * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_L_values, h_A_values, nnz * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_X.ptr, h_X, m * sizeof(double),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // ### cuSPARSE Handle and descriptors initialization ###
    // create the test matrix on the host
    cublasHandle_t   cublasHandle   = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUBLAS( cublasCreate(&cublasHandle) )
    CHECK_CUSPARSE( cusparseCreate(&cusparseHandle) )
    // Create dense vectors
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_B.vec,     m, d_B.ptr, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_X.vec,     m, d_X.ptr, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_R.vec,     m, d_R.ptr, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_R_aux.vec, m, d_R_aux.ptr,
                                        CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_P.vec,   m, d_P.ptr,   CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_T.vec,   m, d_T.ptr,   CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_tmp.vec, m, d_tmp.ptr, CUDA_R_64F) )

    cusparseIndexBase_t  baseIdx = CUSPARSE_INDEX_BASE_ZERO;
    cusparseSpMatDescr_t matA, matL;
    int*                 d_L_rows      = d_A_rows;
    int*                 d_L_columns   = d_A_columns;
    cusparseFillMode_t   fill_lower    = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t   diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    // A
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, m, m, nnz, d_A_rows,
                                      d_A_columns, d_A_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      baseIdx, CUDA_R_64F) )
    // L
    CHECK_CUSPARSE( cusparseCreateCsr(&matL, m, m, nnz, d_L_rows,
                                      d_L_columns, d_L_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      baseIdx, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matL,
                                              CUSPARSE_SPMAT_FILL_MODE,
                                              &fill_lower, sizeof(fill_lower)) )
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matL,
                                              CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_non_unit,
                                              sizeof(diag_non_unit)) )
    //--------------------------------------------------------------------------
    // ### Preparation ### b = A * X
    const double alpha = 0.75;
    size_t       bufferSizeMV;
    void*        d_bufferMV;
    double       beta = 0.0;
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, d_X.vec, &beta, d_B.vec, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV) )
    CHECK_CUDA( cudaMalloc(&d_bufferMV, bufferSizeMV) )

    CHECK_CUSPARSE( cusparseSpMV(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, d_X.vec, &beta, d_B.vec, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV) )
    // X0 = 0
    CHECK_CUDA( cudaMemset(d_X.ptr, 0x0, m * sizeof(double)) )
    //--------------------------------------------------------------------------
    // Perform Incomplete-Cholesky factorization of A (csric0) -> L, L^T
    cusparseMatDescr_t descrM;
    csric02Info_t      infoM        = NULL;
    int                bufferSizeIC = 0;
    void*              d_bufferIC;
    CHECK_CUSPARSE( cusparseCreateMatDescr(&descrM) )
    CHECK_CUSPARSE( cusparseSetMatIndexBase(descrM, baseIdx) )
    CHECK_CUSPARSE( cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL) )
    CHECK_CUSPARSE( cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER) )
    CHECK_CUSPARSE( cusparseSetMatDiagType(descrM,
                                           CUSPARSE_DIAG_TYPE_NON_UNIT) )
    CHECK_CUSPARSE( cusparseCreateCsric02Info(&infoM) )

    CHECK_CUSPARSE( cusparseDcsric02_bufferSize(
                        cusparseHandle, m, nnz, descrM, d_L_values,
                        d_A_rows, d_A_columns, infoM, &bufferSizeIC) )
    CHECK_CUDA( cudaMalloc(&d_bufferIC, bufferSizeIC) )
    CHECK_CUSPARSE( cusparseDcsric02_analysis(
                        cusparseHandle, m, nnz, descrM, d_L_values,
                        d_A_rows, d_A_columns, infoM,
                        CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_bufferIC) )
    int structural_zero;
    CHECK_CUSPARSE( cusparseXcsric02_zeroPivot(cusparseHandle, infoM,
                                               &structural_zero) )
    // M = L * L^T
    CHECK_CUSPARSE( cusparseDcsric02(
                        cusparseHandle, m, nnz, descrM, d_L_values,
                        d_A_rows, d_A_columns, infoM,
                        CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_bufferIC) )
    // Find numerical zero
    int numerical_zero;
    CHECK_CUSPARSE( cusparseXcsric02_zeroPivot(cusparseHandle, infoM,
                                               &numerical_zero) )

    CHECK_CUSPARSE( cusparseDestroyCsric02Info(infoM) )
    CHECK_CUSPARSE( cusparseDestroyMatDescr(descrM) )
    CHECK_CUDA( cudaFree(d_bufferIC) )
    //--------------------------------------------------------------------------
    // ### Run CG computation ###
    printf("CG loop:\n");
    gpu_CG(cublasHandle, cusparseHandle, m,
           matA, matL, d_B, d_X, d_R, d_R_aux, d_P, d_T,
           d_tmp, d_bufferMV, maxIterations, tolerance);
    //--------------------------------------------------------------------------
    // ### Free resources ###
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_B.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_X.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_R.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_R_aux.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_P.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_T.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_tmp.vec) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matL) )
    CHECK_CUSPARSE( cusparseDestroy(cusparseHandle) )
    CHECK_CUBLAS( cublasDestroy(cublasHandle) )

    free(h_A_rows);
    free(h_A_columns);
    free(h_A_values);
    free(h_X);

    CHECK_CUDA( cudaFree(d_X.ptr) )
    CHECK_CUDA( cudaFree(d_B.ptr) )
    CHECK_CUDA( cudaFree(d_R.ptr) )
    CHECK_CUDA( cudaFree(d_R_aux.ptr) )
    CHECK_CUDA( cudaFree(d_P.ptr) )
    CHECK_CUDA( cudaFree(d_T.ptr) )
    CHECK_CUDA( cudaFree(d_tmp.ptr) )
    CHECK_CUDA( cudaFree(d_A_values) )
    CHECK_CUDA( cudaFree(d_A_columns) )
    CHECK_CUDA( cudaFree(d_A_rows) )
    CHECK_CUDA( cudaFree(d_L_values) )
    CHECK_CUDA( cudaFree(d_bufferMV) )
    return EXIT_SUCCESS;
}