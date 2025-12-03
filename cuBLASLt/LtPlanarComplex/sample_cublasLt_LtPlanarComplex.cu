/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <cublasLt.h>
#include <cuda_runtime.h>

#include "sample_cublasLt_LtPlanarComplex.h"
#include "helpers.h"

/// Use cublasLtMatmul to perform tensor-op Cgemm using planar complex memory layout and half-precision inputs.
///
/// For better performance data order transforms should be offline as much as possible.
///
/// transa, transb assumed N; alpha, beta are host pointers, tensor ops allowed, alpha assumed 1, beta assumed 0,
/// stream assumed 0
/// outputs can be either single or half precision, half precision is used in this example
void LtPlanarCgemm(cublasLtHandle_t ltHandle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const __half *A_real,
                   const __half *A_imag,
                   int lda,
                   const __half *B_real,
                   const __half *B_imag,
                   int ldb,
                   __half *C_real,
                   __half *C_imag,
                   int ldc) {
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cuComplex alpha = {1, 0}, beta = {0, 0};

    // cublasLt expects offests in bytes
    int64_t AplaneOffset = (A_imag - A_real) * sizeof(A_real[0]);
    int64_t BplaneOffset = (B_imag - B_real) * sizeof(B_real[0]);
    int64_t CplaneOffset = (C_imag - C_real) * sizeof(C_real[0]);

    checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_C_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // ---------------------------------------------------------------------------------------------
    // create descriptors for planar complex matrices

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_C_16F, transa == CUBLAS_OP_N ? m : k,
                                                 transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &AplaneOffset,
                                                       sizeof(AplaneOffset)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_C_16F, transb == CUBLAS_OP_N ? k : n,
                                                 transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &BplaneOffset,
                                                       sizeof(BplaneOffset)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_C_16F, m, n, ldc));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &CplaneOffset,
                                                       sizeof(CplaneOffset)));

    // ---------------------------------------------------------------------------------------------
    // Launch computation

    checkCublasStatus(cublasLtMatmul(ltHandle, matmulDesc, &alpha, A_real, Adesc, B_real, Bdesc, &beta, C_real, Cdesc,
                                     C_real, Cdesc, NULL, NULL, 0, 0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc) checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
}