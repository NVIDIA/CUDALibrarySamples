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
#if CUDART_VERSION < 12000
// The CUDA 11 compatibility path deliberately uses csrsv2 instead of SpSV.
#define DISABLE_CUSPARSE_DEPRECATED
#endif
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

// Keep the triangular solve setup outside the BiCGStab iteration. CUDA 11's
// SpSV analysis can hang in find_colors_ker under concurrent GPU workloads;
// csrsv2's level analysis avoids that kernel. CUDA 12 removed csrsv2.
typedef struct TriangularSolveStruct {
    void* buffer;
#if CUSPARSE_VER_MAJOR < 12
    cusparseMatDescr_t matrix;
    csrsv2Info_t       info;
    int               rows, nnz;
    int*              row_offsets;
    int*              columns;
    double*           values;
#else
    cusparseSpMatDescr_t matrix;
    cusparseSpSVDescr_t  info;
#endif
} TriangularSolve;

int create_triangular_solve(cusparseHandle_t handle,
                            cusparseSpMatDescr_t matrix, Vec x, Vec y,
                            TriangularSolve* solve) {
#if CUSPARSE_VER_MAJOR < 12
    int64_t rows, columns, nnz;
    cusparseIndexType_t row_type, column_type;
    cusparseIndexBase_t base;
    cudaDataType value_type;
    cusparseFillMode_t fill;
    cusparseDiagType_t diagonal;
    // This sample uses 32-bit CSR indices and double values throughout.
    CHECK_CUSPARSE( cusparseCsrGet(matrix, &rows, &columns, &nnz,
                        (void**) &solve->row_offsets, (void**) &solve->columns,
                        (void**) &solve->values, &row_type, &column_type,
                        &base, &value_type) )
    solve->rows = (int)rows;
    solve->nnz  = (int)nnz;
    CHECK_CUSPARSE( cusparseSpMatGetAttribute(matrix, CUSPARSE_SPMAT_FILL_MODE,
                                              &fill, sizeof(fill)) )
    CHECK_CUSPARSE( cusparseSpMatGetAttribute(matrix, CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diagonal, sizeof(diagonal)) )
    CHECK_CUSPARSE( cusparseCreateMatDescr(&solve->matrix) )
    CHECK_CUSPARSE( cusparseSetMatIndexBase(solve->matrix, base) )
    CHECK_CUSPARSE( cusparseSetMatFillMode(solve->matrix, fill) )
    CHECK_CUSPARSE( cusparseSetMatDiagType(solve->matrix, diagonal) )
    CHECK_CUSPARSE( cusparseCreateCsrsv2Info(&solve->info) )
    int buffer_size;
    CHECK_CUSPARSE( cusparseDcsrsv2_bufferSize(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, solve->rows,
                        solve->nnz, solve->matrix, solve->values,
                        solve->row_offsets, solve->columns, solve->info,
                        &buffer_size) )
    CHECK_CUDA( cudaMalloc(&solve->buffer, buffer_size) )
    CHECK_CUSPARSE( cusparseDcsrsv2_analysis(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, solve->rows,
                        solve->nnz, solve->matrix, solve->values,
                        solve->row_offsets, solve->columns, solve->info,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, solve->buffer) )
#else
    const double one = 1.0;
    size_t buffer_size;
    solve->matrix = matrix;
    CHECK_CUSPARSE( cusparseSpSV_createDescr(&solve->info) )
    CHECK_CUSPARSE( cusparseSpSV_bufferSize(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matrix,
                        x.vec, y.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                        solve->info, &buffer_size) )
    CHECK_CUDA( cudaMalloc(&solve->buffer, buffer_size) )
    CHECK_CUSPARSE( cusparseSpSV_analysis(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matrix,
                        x.vec, y.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                        solve->info, solve->buffer) )
#endif
    return EXIT_SUCCESS;
}

int triangular_solve(cusparseHandle_t handle, TriangularSolve* solve,
                     Vec x, Vec y) {
    const double one = 1.0;
#if CUSPARSE_VER_MAJOR < 12
    CHECK_CUSPARSE( cusparseDcsrsv2_solve(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, solve->rows,
                        solve->nnz, &one, solve->matrix, solve->values,
                        solve->row_offsets, solve->columns, solve->info,
                        x.ptr, y.ptr, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                        solve->buffer) )
#else
    CHECK_CUSPARSE( cusparseSpSV_solve(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &one, solve->matrix,
                        x.vec, y.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                        solve->info) )
#endif
    return EXIT_SUCCESS;
}

int destroy_triangular_solve(TriangularSolve* solve) {
#if CUSPARSE_VER_MAJOR < 12
    CHECK_CUSPARSE( cusparseDestroyCsrsv2Info(solve->info) )
    CHECK_CUSPARSE( cusparseDestroyMatDescr(solve->matrix) )
#else
    CHECK_CUSPARSE( cusparseSpSV_destroyDescr(solve->info) )
#endif
    CHECK_CUDA( cudaFree(solve->buffer) )
    return EXIT_SUCCESS;
}

//==============================================================================

/// This code allocates. The caller must free.
void make_test_matrix(int * n_out,
                      int **row_offsets_out, 
                      int **columns_out, 
                      double **values_out) {
    int grid = 700; // grid resolution

    int n = grid * grid;
    *n_out = n;
    // vertices have 5 neighbors, 
    // but each vertex on the boundary loses 1. corners lose 2.
    int nnz = 5 * n - 4 * grid;

    printf("Creating 5-point time-dependent advection-diffusion matrix.\n"
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
    
    double mass = 0.3; // extra diagonal/mass term    
    double ux = 0.3;   // advection velocity
    double uy = 0.2;

    int it = 0; // next unused index into `columns`/`values`

#define INSERT(u, v, x)                   \
    if(0<=(u) && (u)<grid &&              \
       0<=(v) && (v)<grid)                \
    {                                     \
        columns[it] = ((u) * grid + (v)); \
        values[it] = (x);                 \
        ++it;                             \
    }

    int row = 0;
    row_offsets[row] = 0;
    for (int i = 0; i < grid; ++i) {
        for (int j = 0; j < grid; ++j)
        {
            // Upwinding, so 'ux' and 'uy' only affect the -i and -j directions.
            INSERT(i - 1, j    , -1.0 - ux);
            INSERT(i    , j - 1, -1.0 - uy);
            INSERT(i    , j    ,  4.0 + mass + ux + uy);
            INSERT(i    , j + 1, -1.0);
            INSERT(i + 1, j    , -1.0);
            row_offsets[++row] = it;
        }
    }
    assert(it == nnz);
#undef INSERT
}

//==============================================================================

int gpu_BiCGStab(cublasHandle_t       cublasHandle,
                 cusparseHandle_t     cusparseHandle,
                 int                  m,
                 cusparseSpMatDescr_t matA,
                 cusparseSpMatDescr_t matM_lower,
                 cusparseSpMatDescr_t matM_upper,
                 Vec                  d_B,
                 Vec                  d_X,
                 Vec                  d_R0,
                 Vec                  d_R,
                 Vec                  d_P,
                 Vec                  d_P_aux,
                 Vec                  d_S,
                 Vec                  d_S_aux,
                 Vec                  d_V,
                 Vec                  d_T,
                 Vec                  d_tmp,
                 void*                d_bufferMV,
                 int                  maxIterations,
                 double               tolerance) {
    const double zero      = 0.0;
    const double one       = 1.0;
    const double minus_one = -1.0;
    //--------------------------------------------------------------------------
    TriangularSolve lower, upper;
    if (create_triangular_solve(cusparseHandle, matM_lower, d_P, d_tmp,
                                &lower) != EXIT_SUCCESS ||
        create_triangular_solve(cusparseHandle, matM_upper, d_tmp, d_P_aux,
                                &upper) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    //--------------------------------------------------------------------------
    // ### 1 ### R0 = b - A * X0 (using initial guess in X)
    //    (a) copy b in R0
    CHECK_CUDA( cudaMemcpy(d_R0.ptr, d_B.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) )
    //    (b) compute R = -A * X0 + R
    CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &minus_one, matA, d_X.vec, &one, d_R0.vec,
                                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                 d_bufferMV) )
    //--------------------------------------------------------------------------
    double alpha, delta, delta_prev, omega;
    CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_R0.ptr, 1, d_R0.ptr, 1,
                             &delta) )
    delta_prev = delta;
    // R = R0
    CHECK_CUDA( cudaMemcpy(d_R.ptr, d_R0.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) )
    //--------------------------------------------------------------------------
    // nrm_R0 = ||R||
    double nrm_R;
    CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_R0.ptr, 1, &nrm_R) )
    double threshold = tolerance * nrm_R;
    printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
    //--------------------------------------------------------------------------
    // ### 2 ### repeat until convergence based on max iterations and
    //           and relative residual
    for (int i = 1; i <= maxIterations; i++) {
        printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
        //----------------------------------------------------------------------
        // ### 4, 7 ### P_i = R_i
        if (i == 1) {
            CHECK_CUDA(cudaMemcpy(d_P.ptr, d_R.ptr, m * sizeof(double),
                                  cudaMemcpyDeviceToDevice))
        }
        else {
            //------------------------------------------------------------------
            // ### 6 ### beta = (delta_i / delta_i-1) * (alpha / omega_i-1)
            //    (a) delta_i = (R'_0, R_i-1)
            CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_R0.ptr, 1, d_R.ptr, 1,
                                     &delta) )
            //    (b) beta = (delta_i / delta_i-1) * (alpha / omega_i-1);
            double beta = (delta / delta_prev) * (alpha / omega);
            delta_prev  = delta;
            //------------------------------------------------------------------
            // ### 7 ### P = R + beta * (P - omega * V)
            //    (a) P = - omega * V + P
            double minus_omega = -omega;
            CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &minus_omega, d_V.ptr, 1,
                                      d_P.ptr, 1) )
            //    (b) P = beta * P
            CHECK_CUBLAS( cublasDscal(cublasHandle, m, &beta, d_P.ptr, 1) )
            //    (c) P = R + P
            CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &one, d_R.ptr, 1,
                                      d_P.ptr, 1) )
        }
        //----------------------------------------------------------------------
        // ### 9 ### P_aux = M_U^-1 M_L^-1 P_i
        //    (a) M_L^-1 P_i => tmp    (triangular solver)
        CHECK_CUDA( cudaMemset(d_tmp.ptr,   0x0, m * sizeof(double)) )
        CHECK_CUDA( cudaMemset(d_P_aux.ptr, 0x0, m * sizeof(double)) )
        if (triangular_solve(cusparseHandle, &lower, d_P, d_tmp) !=
            EXIT_SUCCESS)
            return EXIT_FAILURE;
        //    (b) M_U^-1 tmp => P_aux    (triangular solver)
        if (triangular_solve(cusparseHandle, &upper, d_tmp, d_P_aux) !=
            EXIT_SUCCESS)
            return EXIT_FAILURE;
        //----------------------------------------------------------------------
        // ### 10 ### alpha = (R'0, R_i-1) / (R'0, A * P_aux)
        //    (a) V = A * P_aux
        CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                     matA, d_P_aux.vec, &zero, d_V.vec,
                                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                     d_bufferMV) )
        //    (b) denominator = R'0 * V
        double denominator;
        CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_R0.ptr, 1, d_V.ptr, 1,
                                 &denominator) )
        alpha = delta / denominator;
        PRINT_INFO(delta)
        PRINT_INFO(alpha)
        //----------------------------------------------------------------------
        // ### 11 ###  X_i = X_i-1 + alpha * P_aux
        CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &alpha, d_P_aux.ptr, 1,
                                  d_X.ptr, 1) )
        //----------------------------------------------------------------------
        // ### 12 ###  S = R_i-1 - alpha * (A * P_aux)
        //    (a) S = R_i-1
        CHECK_CUDA( cudaMemcpy(d_S.ptr, d_R.ptr, m * sizeof(double),
                               cudaMemcpyDeviceToDevice) )
        //    (b) S = -alpha * V + R_i-1
        double minus_alpha = -alpha;
        CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &minus_alpha, d_V.ptr, 1,
                                  d_S.ptr, 1) )
        //----------------------------------------------------------------------
        // ### 13 ###  check ||S|| < threshold
        double nrm_S;
        CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_S.ptr, 1, &nrm_S) )
        PRINT_INFO(nrm_S)
        if (nrm_S < threshold)
            break;
        //----------------------------------------------------------------------
        // ### 14 ### S_aux = M_U^-1 M_L^-1 S
        //    (a) M_L^-1 S => tmp    (triangular solver)
        cudaMemset(d_tmp.ptr, 0x0, m * sizeof(double));
        cudaMemset(d_S_aux.ptr, 0x0, m * sizeof(double));
        if (triangular_solve(cusparseHandle, &lower, d_S, d_tmp) !=
            EXIT_SUCCESS)
            return EXIT_FAILURE;
        //    (b) M_U^-1 tmp => S_aux    (triangular solver)
        if (triangular_solve(cusparseHandle, &upper, d_tmp, d_S_aux) !=
            EXIT_SUCCESS)
            return EXIT_FAILURE;
        //----------------------------------------------------------------------
        // ### 15 ### omega = (A * S_aux, s) / (A * S_aux, A * S_aux)
        //    (a) T = A * S_aux
        CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                     matA, d_S_aux.vec, &zero, d_T.vec,
                                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                     d_bufferMV) )
        //    (b) omega_num = (A * S_aux, s)
        double omega_num, omega_den;
        CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_T.ptr, 1, d_S.ptr, 1,
                                 &omega_num) )
        //    (c) omega_den = (A * S_aux, A * S_aux)
        CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_T.ptr, 1, d_T.ptr, 1,
                                 &omega_den) )
        //    (d) omega = omega_num / omega_den
        omega = omega_num / omega_den;
        PRINT_INFO(omega)
        // ---------------------------------------------------------------------
        // ### 16 ### omega = X_i = X_i-1 + alpha * P_aux + omega * S_aux
        //    (a) X_i has been updated with h = X_i-1 + alpha * P_aux
        //        X_i = omega * S_aux + X_i
        CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &omega, d_S_aux.ptr, 1,
                                  d_X.ptr, 1) )
        // ---------------------------------------------------------------------
        // ### 17 ###  R_i+1 = S - omega * (A * S_aux)
        //    (a) copy S in R
        CHECK_CUDA( cudaMemcpy(d_R.ptr, d_S.ptr, m * sizeof(double),
                               cudaMemcpyDeviceToDevice) )
        //    (a) R_i+1 = -omega * T + R
        double minus_omega = -omega;
        CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &minus_omega, d_T.ptr, 1,
                                  d_R.ptr, 1) )
       // ---------------------------------------------------------------------
        // ### 18 ###  check ||R_i|| < threshold
        CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) )
        PRINT_INFO(nrm_R)
        if (nrm_R < threshold)
            break;
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
    if (destroy_triangular_solve(&lower) != EXIT_SUCCESS ||
        destroy_triangular_solve(&upper) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}

//==============================================================================
//==============================================================================

int main(int argc, char** argv) {
    const int    maxIterations = 100;
    const double tolerance     = 0.0000000001;
    if (argc != 1) {
        printf("Wrong number of command line arguments. bicgstab_example accepts no arguments.\n");
        return EXIT_FAILURE;
    }
    int     base        = 0;
    int     m           = -1;
    int*    h_A_rows    = NULL;
    int*    h_A_columns = NULL;
    double* h_A_values  = NULL;
    make_test_matrix(&m, &h_A_rows, &h_A_columns, &h_A_values);
    int num_offsets = m + 1;
    int nnz = h_A_rows[m];
    double* h_X = (double*)malloc(m * sizeof(double));

    printf("Testing BiCGStab\n");
    for (int i = 0; i < m; i++)
        h_X[i] = 1.0;
    //--------------------------------------------------------------------------
    // ### Device memory management ###
    int*    d_A_rows, *d_A_columns;
    double* d_A_values, *d_M_values;
    Vec     d_B, d_X, d_R, d_R0, d_P, d_P_aux, d_S, d_S_aux, d_V, d_T, d_tmp;

    // allocate device memory for CSR matrices
    CHECK_CUDA( cudaMalloc((void**) &d_A_rows,    num_offsets * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_A_columns, nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_A_values,  nnz * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_M_values,  nnz * sizeof(double)) )

    CHECK_CUDA( cudaMalloc((void**) &d_B.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_X.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_R.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_R0.ptr,    m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_P.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_P_aux.ptr, m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_S.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_S_aux.ptr, m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_V.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_T.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_tmp.ptr,   m * sizeof(double)) )

    // copy the CSR matrices and vectors into device memory
    CHECK_CUDA( cudaMemcpy(d_A_rows, h_A_rows, num_offsets * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_A_columns, h_A_columns, nnz *  sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_A_values, h_A_values, nnz * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_M_values, h_A_values, nnz * sizeof(double),
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
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_R0.vec,    m, d_R0.ptr, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_P.vec,     m, d_P.ptr, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_P_aux.vec, m, d_P_aux.ptr,
                                        CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_S.vec,     m, d_S.ptr, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_S_aux.vec, m, d_S_aux.ptr,
                                        CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_V.vec,   m, d_V.ptr,   CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_T.vec,   m, d_T.ptr,   CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_tmp.vec, m, d_tmp.ptr, CUDA_R_64F) )

    cusparseIndexBase_t  baseIdx = CUSPARSE_INDEX_BASE_ZERO;
    // IMPORTANT: Upper/Lower triangular decompositions of A
    //            (matM_lower, matM_upper) must use two distinct descriptors
    cusparseSpMatDescr_t matA, matM_lower, matM_upper;
    cusparseMatDescr_t   matLU;
    int*                 d_M_rows      = d_A_rows;
    int*                 d_M_columns   = d_A_columns;
    cusparseFillMode_t   fill_lower    = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t   diag_unit     = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t   fill_upper    = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t   diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    // A
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, m, m, nnz, d_A_rows,
                                      d_A_columns, d_A_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      baseIdx, CUDA_R_64F) )
    // M_lower
    CHECK_CUSPARSE( cusparseCreateCsr(&matM_lower, m, m, nnz, d_M_rows,
                                      d_M_columns, d_M_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      baseIdx, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matM_lower,
                                              CUSPARSE_SPMAT_FILL_MODE,
                                              &fill_lower, sizeof(fill_lower)) )
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matM_lower,
                                              CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_unit, sizeof(diag_unit)) )
    // M_upper
    CHECK_CUSPARSE( cusparseCreateCsr(&matM_upper, m, m, nnz, d_M_rows,
                                      d_M_columns, d_M_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      baseIdx, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matM_upper,
                                              CUSPARSE_SPMAT_FILL_MODE,
                                              &fill_upper, sizeof(fill_upper)) )
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matM_upper,
                                              CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_non_unit,
                                              sizeof(diag_non_unit)) )
    //--------------------------------------------------------------------------
    // ### Preparation ### b = A * X
    const double alpha = 0.75;
    size_t bufferSizeMV;
    void*  d_bufferMV;
    double beta = 0.0;
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
    // Perform Incomplete-LU factorization of A (csrilu0) -> M_lower, M_upper
    csrilu02Info_t infoM        = NULL;
    int            bufferSizeLU = 0;
    void*          d_bufferLU;
    CHECK_CUSPARSE( cusparseCreateMatDescr(&matLU) )
    CHECK_CUSPARSE( cusparseSetMatType(matLU, CUSPARSE_MATRIX_TYPE_GENERAL) )
    CHECK_CUSPARSE( cusparseSetMatIndexBase(matLU, baseIdx) )
    CHECK_CUSPARSE( cusparseCreateCsrilu02Info(&infoM) )

    CHECK_CUSPARSE( cusparseDcsrilu02_bufferSize(
                        cusparseHandle, m, nnz, matLU, d_M_values,
                        d_A_rows, d_A_columns, infoM, &bufferSizeLU) )
    CHECK_CUDA( cudaMalloc(&d_bufferLU, bufferSizeLU) )
    CHECK_CUSPARSE( cusparseDcsrilu02_analysis(
                        cusparseHandle, m, nnz, matLU, d_M_values,
                        d_A_rows, d_A_columns, infoM,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU) )
    int structural_zero;
    CHECK_CUSPARSE( cusparseXcsrilu02_zeroPivot(cusparseHandle, infoM,
                                                &structural_zero) )
    // M = L * U
    CHECK_CUSPARSE( cusparseDcsrilu02(
                        cusparseHandle, m, nnz, matLU, d_M_values,
                        d_A_rows, d_A_columns, infoM,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU) )
    // Find numerical zero
    int numerical_zero;
    CHECK_CUSPARSE( cusparseXcsrilu02_zeroPivot(cusparseHandle, infoM,
                                                &numerical_zero) )

    CHECK_CUSPARSE( cusparseDestroyCsrilu02Info(infoM) )
    CHECK_CUSPARSE( cusparseDestroyMatDescr(matLU) )
    CHECK_CUDA( cudaFree(d_bufferLU) )
    //--------------------------------------------------------------------------
    // ### Run BiCGStab computation ###
    printf("BiCGStab loop:\n");
    if (gpu_BiCGStab(cublasHandle, cusparseHandle, m,
                    matA, matM_lower, matM_upper,
                    d_B, d_X, d_R0, d_R, d_P, d_P_aux, d_S, d_S_aux, d_V, d_T,
                    d_tmp, d_bufferMV, maxIterations, tolerance) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    //--------------------------------------------------------------------------
    // ### Free resources ###
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_B.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_X.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_R.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_R0.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_P.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_P_aux.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_S.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_S_aux.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_V.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_T.vec) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(d_tmp.vec) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matM_lower) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matM_upper) )
    CHECK_CUSPARSE( cusparseDestroy(cusparseHandle) )
    CHECK_CUBLAS( cublasDestroy(cublasHandle) )

    free(h_A_rows);
    free(h_A_columns);
    free(h_A_values);
    free(h_X);

    CHECK_CUDA( cudaFree(d_X.ptr) )
    CHECK_CUDA( cudaFree(d_B.ptr) )
    CHECK_CUDA( cudaFree(d_R.ptr) )
    CHECK_CUDA( cudaFree(d_R0.ptr) )
    CHECK_CUDA( cudaFree(d_P.ptr) )
    CHECK_CUDA( cudaFree(d_P_aux.ptr) )
    CHECK_CUDA( cudaFree(d_S.ptr) )
    CHECK_CUDA( cudaFree(d_S_aux.ptr) )
    CHECK_CUDA( cudaFree(d_V.ptr) )
    CHECK_CUDA( cudaFree(d_T.ptr) )
    CHECK_CUDA( cudaFree(d_tmp.ptr) )
    CHECK_CUDA( cudaFree(d_A_values) )
    CHECK_CUDA( cudaFree(d_A_columns) )
    CHECK_CUDA( cudaFree(d_A_rows) )
    CHECK_CUDA( cudaFree(d_M_values) )
    CHECK_CUDA( cudaFree(d_bufferMV) )
    return EXIT_SUCCESS;
}
