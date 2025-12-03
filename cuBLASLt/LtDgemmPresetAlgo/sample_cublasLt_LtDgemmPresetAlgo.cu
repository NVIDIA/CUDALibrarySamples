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

#include "sample_cublasLt_LtDgemmPresetAlgo.h"
#include "helpers.h"

/// Sample wrapper executing double precision gemm with a predefined algorithm using cublasLtMatmul, nearly a drop-in
/// replacement for cublasDgemm, with addition of the workspace to support split-K algorithms
///
/// pointer mode is always host, to change it configure the appropriate matmul descriptor attribute
/// matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to change
/// this configure appropriate attribute in the preference handle
///
/// NOTE: this sample may not work on all architectures or all problem sizes
void LtDgemmPresetAlgo(cublasLtHandle_t ltHandle,
                       cublasOperation_t transa,
                       cublasOperation_t transb,
                       int m,
                       int n,
                       int k,
                       const double *alpha, /* host pointer */
                       const double *A,
                       int lda,
                       const double *B,
                       int ldb,
                       const double *beta, /* host pointer */
                       double *C,
                       int ldc,
                       void *workspace,
                       size_t workspaceSize,
                       cudaStream_t stream) {
    cublasLtMatmulDescOpaque_t operationDesc = {};
    cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
    cublasLtMatmulAlgo_t algo = {};

    const int32_t algoId = 10;
    const cublasLtMatmulTile_t tileId = CUBLASLT_MATMUL_TILE_16x16;                    // 5
    const cublasLtReductionScheme_t reductionMode = CUBLASLT_REDUCTION_SCHEME_INPLACE; // 1
    const int32_t splitKFactor = 256;

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_64F, CUDA_R_64F));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    checkCublasStatus(cublasLtMatrixLayoutInit(&Adesc, CUDA_R_64F, transa == CUBLAS_OP_N ? m : k,
                                               transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutInit(&Bdesc, CUDA_R_64F, transb == CUBLAS_OP_N ? k : n,
                                               transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_64F, m, n, ldc));

    checkCublasStatus(cublasLtMatmulAlgoInit(ltHandle,           //
                                             CUBLAS_COMPUTE_64F, // compute
                                             CUDA_R_64F,         // scale
                                             CUDA_R_64F,         // A
                                             CUDA_R_64F,         // B
                                             CUDA_R_64F,         // C
                                             CUDA_R_64F,         // D
                                             algoId, &algo));

    checkCublasStatus(
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId)));
    checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionMode,
                                                           sizeof(reductionMode)));
    checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKFactor,
                                                           sizeof(splitKFactor)));

    checkCublasStatus(cublasLtMatmul(ltHandle, &operationDesc, alpha, A, &Adesc, B, &Bdesc, beta, C, &Cdesc, C, &Cdesc,
                                     &algo, workspace, workspaceSize, stream));
}