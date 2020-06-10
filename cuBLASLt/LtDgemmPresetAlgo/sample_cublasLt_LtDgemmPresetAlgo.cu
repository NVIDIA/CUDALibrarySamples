/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
    const cublasLtMatmulTile_t tileId = CUBLASLT_MATMUL_TILE_16x16; // 5
    const cublasLtReductionScheme_t reductionMode = CUBLASLT_REDUCTION_SCHEME_INPLACE; // 1
    const int32_t splitKFactor = 256;

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_64F, CUDA_R_64F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    checkCublasStatus(cublasLtMatrixLayoutInit(&Adesc, CUDA_R_64F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutInit(&Bdesc, CUDA_R_64F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_64F, m, n, ldc));

    checkCublasStatus(cublasLtMatmulAlgoInit(ltHandle,  //
                                             CUBLAS_COMPUTE_64F,   // compute
                                             CUDA_R_64F,   // scale
                                             CUDA_R_64F,   // A
                                             CUDA_R_64F,   // B
                                             CUDA_R_64F,   // C
                                             CUDA_R_64F,   // D
                                             algoId,
                                             &algo));

    checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId)));
    checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionMode, sizeof(reductionMode)));
    checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKFactor, sizeof(splitKFactor)));

    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     &operationDesc,
                                     alpha,
                                     A,
                                     &Adesc,
                                     B,
                                     &Bdesc,
                                     beta,
                                     C,
                                     &Cdesc,
                                     C,
                                     &Cdesc,
                                     &algo,
                                     workspace,
                                     workspaceSize,
                                     stream));
}