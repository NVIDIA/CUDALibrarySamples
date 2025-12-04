/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "sample_cublasLt_LtHSHgemmGroupedSimple.h"
#include "helpers.h"

/// (!) EXPERIMENTAL: This sample implements an experimental grouped gemm
/// feature and may be changed or removed in the future. Refer to the
/// documentation for more details and limitations.
///
/// Sample wrapper executing a grouped mixed precision gemm with cublasLtMatmul
///
/// pointer mode is always device, to change it configure the appropriate matmul
/// descriptor attribute matmul is not using cublas handle's configuration of
/// math mode, here tensor ops are implicitly allowed
///
void LtHSHgemmGroupedSimple(cublasLtHandle_t ltHandle,
                            cublasOperation_t transa,
                            cublasOperation_t transb,
                            const void *mArrayDev,
                            int64_t avgM,
                            const void *nArrayDev,
                            int64_t avgN,
                            const void *kArrayDev,
                            int64_t avgK,
                            const float *const *alphaArrayDev,
                            const __half *const *A,
                            const void *ldaArrayDev,
                            const __half *const *B,
                            const void *ldbArrayDev,
                            const float *const *betaArrayDev,
                            const __half *const *C,
                            const void *ldcArrayDev,
                            __half *const *D,
                            const void *lddArrayDev,
                            int batchCount,
                            void *workspace,
                            size_t workspaceSize) {
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtPointerMode_t pointerMode = CUBLASLT_POINTER_MODE_DEVICE;
    int64_t alphaBatchStride = 1;
    int64_t betaBatchStride = 1;
    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for
    // details about defaults; here we need to set the transforms for A and B
    // and pointer mode and alpha/beta batch stride
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointerMode,
                                                     sizeof(pointerMode)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE,
                                                     &alphaBatchStride, sizeof(alphaBatchStride)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE,
                                                     &betaBatchStride, sizeof(betaBatchStride)));

    // create matrix descriptors
    checkCublasStatus(cublasLtGroupedMatrixLayoutCreate(&Adesc, CUDA_R_16F, batchCount,
                                                        transa == CUBLAS_OP_N ? mArrayDev : kArrayDev,
                                                        transa == CUBLAS_OP_N ? kArrayDev : mArrayDev, ldaArrayDev));
    checkCublasStatus(cublasLtGroupedMatrixLayoutCreate(&Bdesc, CUDA_R_16F, batchCount,
                                                        transb == CUBLAS_OP_N ? kArrayDev : nArrayDev,
                                                        transb == CUBLAS_OP_N ? nArrayDev : kArrayDev, ldbArrayDev));
    checkCublasStatus(
        cublasLtGroupedMatrixLayoutCreate(&Cdesc, CUDA_R_16F, batchCount, mArrayDev, nArrayDev, ldcArrayDev));
    checkCublasStatus(
        cublasLtGroupedMatrixLayoutCreate(&Ddesc, CUDA_R_16F, batchCount, mArrayDev, nArrayDev, lddArrayDev));

    // Since M, N, K are located on the device, it is highly recommended to
    // provide average values for them to improve the heuristics result. Without
    // this, the functionality will still work, but the heuristics result may
    // not be optimal.
    cublasLtMatmulPreference_t preference = NULL;
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS,
                                                           &avgM, sizeof(avgM)));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS,
                                                           &avgN, sizeof(avgN)));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM, &avgK, sizeof(avgK)));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &workspaceSize, sizeof(workspaceSize)));
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
                                                     &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    // in this simplified example we take advantage of cublasLtMatmul shortcut
    // notation with algo=NULL which will force matmul to get the basic
    // heuristic result internally. Downsides of this approach are that there is
    // no way to configure search preferences (e.g. disallow tensor operations
    // or some reduction schemes) and no way to store the algo for later use
    checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alphaArrayDev, A, Adesc, B, Bdesc, betaArrayDev, C, Cdesc,
                                     (void *)D, Ddesc, &heuristicResult.algo, workspace, workspaceSize, 0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (Ddesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}
