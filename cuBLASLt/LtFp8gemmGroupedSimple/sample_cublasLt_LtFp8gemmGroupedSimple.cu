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

#include <cublasLt.h>

#include "helpers.h"
#include "sample_cublasLt_LtFp8gemmGroupedSimple.h"

/// (!) EXPERIMENTAL: This sample implements an experimental grouped gemm
/// feature and may be changed or removed in the future. Refer to the
/// documentation for more details and limitations.
///
/// Sample wrapper executing grouped FP8 gemm with cublasLtMatmul, with per-batch scalar scaling.
/// Output is bfloat16, alpha/beta are host pointers shared across all groups.
///
/// matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed
void LtFp8gemmGroupedSimple(cublasLtHandle_t ltHandle,
                            cublasOperation_t transa,
                            cublasOperation_t transb,
                            const void *mArrayDev,
                            int64_t avgM,
                            const void *nArrayDev,
                            int64_t avgN,
                            const void *kArrayDev,
                            int64_t avgK,
                            const float *alpha,
                            const float *const *a_scale,
                            const __nv_fp8_e4m3 *const *A,
                            const void *ldaArrayDev,
                            const float *const *b_scale,
                            const __nv_fp8_e4m3 *const *B,
                            const void *ldbArrayDev,
                            const float *beta,
                            const __nv_bfloat16 *const *C,
                            const void *ldcArrayDev,
                            __nv_bfloat16 *const *D,
                            const void *lddArrayDev,
                            int batchCount,
                            void *workspace,
                            size_t workspaceSize,
                            cublasLtMatmulMatrixScale_t AScaleMode,
                            cublasLtMatmulMatrixScale_t BScaleMode) {
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;

    // create operation descriptor; see cublasLtMatmulDescAttributes_t for
    // details about defaults; here we need to set the transforms for A and B
    // pointer mode is host (default) - alpha/beta are shared across all groups
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // set per-batch scalar scaling mode (each group has its own scalar scale)
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &AScaleMode,
                                                     sizeof(AScaleMode)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &BScaleMode,
                                                     sizeof(BScaleMode)));

    // set scaling factors (per-batch scalar scaling for A, B only - C and D are bf16)
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));

    // create grouped matrix descriptors
    checkCublasStatus(cublasLtGroupedMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, batchCount,
                                                        transa == CUBLAS_OP_N ? mArrayDev : kArrayDev,
                                                        transa == CUBLAS_OP_N ? kArrayDev : mArrayDev, ldaArrayDev));
    checkCublasStatus(cublasLtGroupedMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, batchCount,
                                                        transb == CUBLAS_OP_N ? kArrayDev : nArrayDev,
                                                        transb == CUBLAS_OP_N ? nArrayDev : kArrayDev, ldbArrayDev));
    checkCublasStatus(
        cublasLtGroupedMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, batchCount, mArrayDev, nArrayDev, ldcArrayDev));
    checkCublasStatus(
        cublasLtGroupedMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, batchCount, mArrayDev, nArrayDev, lddArrayDev));

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

    checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, (void *)D,
                                     Ddesc, &heuristicResult.algo, workspace, workspaceSize, 0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (Ddesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}
