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

    // ---------------------------------------------------------------------------------------------
    // create descriptors for planar complex matrices

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_C_16F, m, k, lda));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &AplaneOffset, sizeof(AplaneOffset)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_C_16F, k, n, ldb));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &BplaneOffset, sizeof(BplaneOffset)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_C_16F, m, n, ldc));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &CplaneOffset, sizeof(CplaneOffset)));

    // ---------------------------------------------------------------------------------------------
    // Launch computation

    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     matmulDesc,
                                     &alpha,
                                     A_real,
                                     Adesc,
                                     B_real,
                                     Bdesc,
                                     &beta,
                                     C_real,
                                     Cdesc,
                                     C_real,
                                      Cdesc,
                                     NULL,
                                     NULL,
                                     0,
                                     0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc) checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
}
