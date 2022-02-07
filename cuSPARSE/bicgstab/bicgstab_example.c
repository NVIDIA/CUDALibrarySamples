/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>  // fopen
#include <stdlib.h> // EXIT_FAILURE
#include <string.h> // strtok

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

void mtx_header(const char* file_path,
                int*        num_lines,
                int*        num_rows,
                int*        num_cols,
                int*        nnz,
                int*        is_symmetric) {
    char buffer[256];
    FILE* file = fopen(file_path, "r");
    if (file == NULL) {
        printf("Error: unable to open the file %s\n", file_path);
        exit(EXIT_FAILURE);
    }
    fgets(buffer, 256, file); // skip comments
    char *token = strtok(buffer, " ");
    if (strcmp(token, "%%MatrixMarket") != 0) {
        printf("Unsupported file format. Only MTX format is supported");
        exit(EXIT_FAILURE);
    }
    strtok(NULL, " "); // skip word
    strtok(NULL, " "); // skip word
    token = strtok(NULL, " "); // check data type
    if (strcmp(token, "real") != 0) {
        printf("Only real (double) matrices are supported");
        exit(EXIT_FAILURE);
    }
    token = strtok(NULL, " \n"); // symmetric, unsymmetric
    *is_symmetric = (strcmp(token, "symmetric") == 0);
    while (fgetc(file) == '%')
        fgets(buffer, 256, file); // skip % comments
    fseek(file, -1, SEEK_CUR);
    fscanf(file, "%d %d %d", num_rows, num_cols, num_lines);
    *nnz = (*is_symmetric) ? *num_lines * 2 : *num_lines;
    fclose(file);
}

void mtx_parsing(const char* file_path,
                 int         num_lines,
                 int         num_rows,
                 int         nnz,
                 int*        rows_offsets,
                 int*        columns,
                 double*     values,
                 int         base) {
    typedef struct IdxType {
        int    row, col;
        double val;
    } Idx;
    int sort_by_row(const void *a, const void *b) {
        return ((Idx*) a)->row - ((Idx*) b)->row;
    }
    char buffer[256];
    FILE* file = fopen(file_path, "r");
    while (fgetc(file) == '%')
        fgets(buffer, 256, file); // skip comments
    fgets(buffer, 256, file);     // skip num row, cols, nnz

    Idx* idx_tmp = (Idx*) malloc(nnz * sizeof(Idx));
    for (int i = 0; i < num_lines; i++) {
        int    row, column;
        double value;
        fscanf(file, "%d %d %lf ", &row, &column, &value);
        row         -= (1 - base);
        column      -= (1 - base);
        idx_tmp[i].row = row;
        idx_tmp[i].col = column;
        idx_tmp[i].val = value;
        if (nnz != num_lines) { // is stored symmetric
            idx_tmp[i + num_lines].row = column;
            idx_tmp[i + num_lines].col = row;
            idx_tmp[i + num_lines].val = value;
        }
    }
    qsort(idx_tmp, nnz, sizeof(Idx), sort_by_row); // sort by row
    memset(rows_offsets, 0x0, (num_rows + 1) * sizeof(int));
    for (int i = 0; i < nnz; i++)
        rows_offsets[idx_tmp[i].row + 1]++;
    // prefix-scan
    for (int i = 1; i <= num_rows; i++)
        rows_offsets[i] = rows_offsets[i] + rows_offsets[i - 1];
    for (int i = 0; i < nnz; i++) {
        columns[i] = idx_tmp[i].col;
        values[i]  = idx_tmp[i].val;
    }
    fclose(file);
    free(idx_tmp);
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
    // Create opaque data structures that holds analysis data between calls
    double              coeff_tmp;
    size_t              bufferSizeL, bufferSizeU;
    void*               d_bufferL, *d_bufferU;
    cusparseSpSVDescr_t spsvDescrL, spsvDescrU;
    CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsvDescrL) )
    CHECK_CUSPARSE( cusparseSpSV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_lower, d_P.vec, d_tmp.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL) )
    CHECK_CUDA( cudaMalloc(&d_bufferL, bufferSizeL) )
    CHECK_CUSPARSE( cusparseSpSV_analysis(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_lower, d_P.vec, d_tmp.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL) )

    // Calculate UPPER buffersize
    CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsvDescrU) )
    CHECK_CUSPARSE( cusparseSpSV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_upper, d_tmp.vec, d_P_aux.vec,
                        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU,
                        &bufferSizeU) )
    CHECK_CUDA( cudaMalloc(&d_bufferU, bufferSizeU) )
    CHECK_CUSPARSE( cusparseSpSV_analysis(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_upper, d_tmp.vec, d_P_aux.vec,
                        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU,
                        d_bufferU) )
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
        CHECK_CUDA( cudaMemcpy(d_P.ptr, d_R.ptr, m * sizeof(double),
                               cudaMemcpyDeviceToDevice) )
        if (i > 1) {
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
        CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_lower, d_P.vec, d_tmp.vec,
                                           CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrL) )
        //    (b) M_U^-1 tmp => P_aux    (triangular solver)
        CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_upper, d_tmp.vec,
                                           d_P_aux.vec, CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrU) )
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
        CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_lower, d_S.vec, d_tmp.vec,
                                           CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrL) )
        //    (b) M_U^-1 tmp => S_aux    (triangular solver)
        CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_upper, d_tmp.vec,
                                           d_S_aux.vec, CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrU))
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
    CHECK_CUSPARSE( cusparseSpSV_destroyDescr(spsvDescrL) )
    CHECK_CUSPARSE( cusparseSpSV_destroyDescr(spsvDescrU) )
    CHECK_CUDA( cudaFree(d_bufferL) )
    CHECK_CUDA( cudaFree(d_bufferU) )
    return EXIT_SUCCESS;
}

//==============================================================================
//==============================================================================

int main(int argc, char** argv) {
    const int    maxIterations = 20;
    const double tolerance     = 0.0000000001;
    printf("Usage: bicgstab_example <matrix.mtx>\n");
    if (argc != 2) {
        printf("Wrong parameter: bicgstab_example <matrix.mtx>\n");
        return EXIT_FAILURE;
    }
    int base = 0;
    int num_rows, num_cols, nnz, num_lines, is_symmetric;
    mtx_header(argv[1], &num_lines, &num_rows, &num_cols, &nnz, &is_symmetric);
    printf("\nmatrix name: %s\n"
           "num. rows:   %d\n"
           "num. cols:   %d\n"
           "nnz:         %d\n"
           "structure:   %s\n\n",
           argv[1], num_rows, num_cols, nnz,
           (is_symmetric) ? "symmetric" : "unsymmetric");
    if (num_rows != num_cols) {
        printf("the input matrix must be square\n");
        return EXIT_FAILURE;
    }
    if (!is_symmetric) {
        printf("the input matrix must be symmetric\n");
        return EXIT_FAILURE;
    }
    int     m           = num_rows;
    int     num_offsets = m + 1;
    int*    h_A_rows    = (int*)    malloc(num_offsets * sizeof(int));
    int*    h_A_columns = (int*)    malloc(nnz * sizeof(int));
    double* h_A_values  = (double*) malloc(nnz * sizeof(double));
    double* h_X         = (double*) malloc(m * sizeof(double));
    printf("Matrix parsing...\n");
    mtx_parsing(argv[1], num_lines, num_rows, nnz, h_A_rows,
                h_A_columns, h_A_values, base);
    printf("Testing BiCGStab\n");
    for (int i = 0; i < num_rows; i++)
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
    gpu_BiCGStab(cublasHandle, cusparseHandle, m,
                 matA, matM_lower, matM_upper,
                 d_B, d_X, d_R0, d_R, d_P, d_P_aux, d_S, d_S_aux, d_V, d_T,
                 d_tmp, d_bufferMV, maxIterations, tolerance);
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
