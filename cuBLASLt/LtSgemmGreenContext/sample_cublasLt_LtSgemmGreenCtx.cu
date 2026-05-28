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

#include "sample_cublasLt_LtSgemmGreenCtx.h"
#include "helpers.h"

void LtSgemmGCtx(cublasLtHandle_t ltHandle,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 int m,
                 int n,
                 int k,
                 const float *alpha, /* host pointer */
                 const float *A,
                 int lda,
                 const float *B,
                 int ldb,
                 const float *beta, /* host pointer */
                 float *C,
                 int ldc,
                 void *workspace,
                 size_t workspaceSize,
                 unsigned int minGreenContextSmCount,
                 cudaStream_t primaryContextStream) {

    cudaExecutionContext_t greenCtx = 0;
    cudaStream_t stream = 0;
    {
        int device = 0;
        cudaDevResource input;
        cudaDevResource smPartition;
        unsigned int nbGroups = 1;
        cudaDevResourceDesc_t smPartitionDesc;
        checkCudaStatus(cudaDeviceGetDevResource(device, &input, cudaDevResourceTypeSm));
        checkCudaStatus(
            cudaDevSmResourceSplitByCount(&smPartition, &nbGroups, &input, NULL, 0, minGreenContextSmCount));
        checkCudaStatus(cudaDevResourceGenerateDesc(&smPartitionDesc, &smPartition, 1));
        checkCudaStatus(cudaGreenCtxCreate(&greenCtx, smPartitionDesc, device, 0));
        checkCudaStatus(cudaExecutionCtxStreamCreate(&stream, greenCtx, 0, 0));
        // Note: streams created on the green context are non-blocking and need explicit synchronization.
        cudaEvent_t ev;
        checkCudaStatus(cudaEventCreate(&ev));
        checkCudaStatus(cudaEventRecord(ev, primary_context_stream));
        checkCudaStatus(cudaStreamWaitEvent(stream, ev, 0));
        checkCudaStatus(cudaEventDestroy(ev));
    }

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation descriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k,
                                                 transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n,
                                                 transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul with the green context resources in mind passed
    // via the stream.
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristicForStream(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc,
                                                              preference, 1, &heuristicResult, &returnedResults,
                                                              stream));

    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc,
                                     &heuristicResult.algo, workspace, workspaceSize, stream));

    checkCudaStatus(cudaStreamSynchronize(stream));
    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    if (stream) checkCudaStatus(cudaStreamDestroy(stream));
    if (greenCtx) checkCudaStatus(cudaExecutionCtxDestroy(greenCtx));
}
