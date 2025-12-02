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

/// Sample wrapper executing mixed precision grouped gemm with cublasLtMatmul
///
/// pointer mode is always device, to change it configure the appropriate matmul descriptor attribute
/// matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed
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
                            size_t workspaceSize);
