/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#ifndef CUSPARSE_ENABLE_EXPERIMENTAL_API
#   define CUSPARSE_ENABLE_EXPERIMENTAL_API
#endif
#include <cusparse.h> // cusparseSpMVGeneric
#include <stdio.h>    // printf
#include <stdlib.h>   // EXIT_FAILURE

#define CHECK_CUDA(func)                                                       \
    {                                                                          \
        cudaError_t status = (func);                                           \
        if (status != cudaSuccess) {                                           \
            printf("CUDA API failed at line %d with error: %s (%d)\n",         \
                   __LINE__, cudaGetErrorString(status), status);              \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    }

#define CHECK_CUSPARSE(func)                                                   \
    {                                                                          \
        cusparseStatus_t status = (func);                                      \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n",     \
                   __LINE__, cusparseGetErrorString(status), status);          \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    }

int main(void) {
    // Host problem definition
    const int A_num_rows = 4;
    const int A_num_cols = 4;
    const int A_nnz      = 7;
    // Only the lower triangle is stored in CSR format:
    //          [ 1  .  .  . ]
    // stored = [ 2  3  .  . ]
    //          [ 0  4  5  . ]
    //          [ 0  0  6  7 ]
    //
    // With CUSPARSE_SYMMETRIC and CUSPARSE_STORE_LOWER, cusparseSpMVGeneric
    // interprets the stored entries as the following full symmetric matrix:
    //     [ 1  2  0  0 ]
    // A = [ 2  3  4  0 ]
    //     [ 0  4  5  6 ]
    //     [ 0  0  6  7 ]
    int   hA_csrOffsets[] = {0, 1, 3, 5, 7};
    int   hA_columns[]    = {0, 0, 1, 1, 2, 2, 3};
    float hA_values[]     = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float hX[]            = {1.0f, 2.0f, 3.0f, 4.0f};
    float hY[]            = {1.0f, 2.0f, 3.0f, 4.0f};
    float hZ[]            = {0.0f, 0.0f, 0.0f, 0.0f};
    float hZ_result[]     = {7.0f, 24.0f, 53.0f, 54.0f};
    float alpha           = 1.0f;
    float beta            = 2.0f;

    // Device memory management
    int *  dA_csrOffsets, *dA_columns;
    float *dA_values, *dX, *dY, *dZ;
    CHECK_CUDA(
        cudaMalloc((void**) &dA_csrOffsets, (A_num_rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**) &dA_values, A_nnz * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void**) &dX, A_num_cols * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void**) &dY, A_num_rows * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void**) &dZ, A_num_rows * sizeof(float)))

    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                          (A_num_rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(
        cudaMemcpy(dX, hX, A_num_cols * sizeof(float), cudaMemcpyHostToDevice))
    CHECK_CUDA(
        cudaMemcpy(dY, hY, A_num_rows * sizeof(float), cudaMemcpyHostToDevice))
    CHECK_CUDA(
        cudaMemcpy(dZ, hZ, A_num_rows * sizeof(float), cudaMemcpyHostToDevice))

    // cuSPARSE descriptors
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY, vecZ;
    cusparseSpMVDescr_t  spmvDescr = NULL;
    void*                dBuffer   = NULL;
    size_t               bufferSize;

    CHECK_CUSPARSE(cusparseCreate(&handle))
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                     dA_csrOffsets, dA_columns, dA_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    cusparseSymmetryType_t symmetry = CUSPARSE_SYMMETRIC;
    cusparseSpMatStorage_t storage  = CUSPARSE_STORE_LOWER;
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_SYMMETRY_TYPE,
                                             &symmetry, sizeof(symmetry)))
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_STORAGE,
                                             &storage, sizeof(storage)))
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecZ, A_num_rows, dZ, CUDA_R_32F))

    // Configure the descriptor. Unset parameters use their default values.
    CHECK_CUSPARSE(cusparseSpMV_createDescr(&spmvDescr))
    cusparseOperation_t operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpMVAlg_t   algorithm = CUSPARSE_SPMV_ALG_DEFAULT;
    CHECK_CUSPARSE(cusparseSpMVDescrSetParam(spmvDescr, CUSPARSE_PARAM_OP,
                                             &operation, sizeof(operation)))
    CHECK_CUSPARSE(cusparseSpMVDescrSetParam(spmvDescr, CUSPARSE_PARAM_ALG,
                                             &algorithm, sizeof(algorithm)))

    // Query, allocate, and attach the external workspace.
    CHECK_CUSPARSE(cusparseSpMVGeneric_bufferSize(handle, &alpha, matA, vecX,
                                                  &beta, vecY, vecZ, CUDA_R_32F,
                                                  spmvDescr, &bufferSize))
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
    }
    CHECK_CUSPARSE(cusparseSpMVDescrSetParam(spmvDescr, CUSPARSE_PARAM_BUFFER,
                                             &dBuffer, sizeof(dBuffer)))

    // Preprocessing is optional and can improve repeated SpMV calls.
    CHECK_CUSPARSE(cusparseSpMVGeneric_preprocess(
        handle, &alpha, matA, vecX, &beta, vecY, vecZ, CUDA_R_32F, spmvDescr))

    // Z = alpha * A * X + beta * Y. vecY remains unchanged.
    CHECK_CUSPARSE(cusparseSpMVGeneric(handle, &alpha, matA, vecX, &beta, vecY,
                                       vecZ, CUDA_R_32F, spmvDescr))

    CHECK_CUDA(
        cudaMemcpy(hZ, dZ, A_num_rows * sizeof(float), cudaMemcpyDeviceToHost))
    float hY_after[4];
    CHECK_CUDA(cudaMemcpy(hY_after, dY, A_num_rows * sizeof(float),
                          cudaMemcpyDeviceToHost))

    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        if (hZ[i] != hZ_result[i] || hY_after[i] != hY[i]) {
            correct = 0;
            break;
        }
    }
    printf("spmv_symmetric_example test %s\n",
           correct ? "PASSED" : "FAILED");

    CHECK_CUSPARSE(cusparseSpMV_destroyDescr(spmvDescr))
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecZ))
    CHECK_CUSPARSE(cusparseDestroy(handle))
    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(dA_csrOffsets))
    CHECK_CUDA(cudaFree(dA_columns))
    CHECK_CUDA(cudaFree(dA_values))
    CHECK_CUDA(cudaFree(dX))
    CHECK_CUDA(cudaFree(dY))
    CHECK_CUDA(cudaFree(dZ))

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
