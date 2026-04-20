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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

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
                            const float *alpha, /* host pointer */
                            const float *const *a_scale, /* device pointer array */
                            const __nv_fp8_e4m3 *const *A,
                            const void *ldaArrayDev,
                            const float *const *b_scale, /* device pointer array */
                            const __nv_fp8_e4m3 *const *B,
                            const void *ldbArrayDev,
                            const float *beta, /* host pointer */
                            const __nv_bfloat16 *const *C,
                            const void *ldcArrayDev,
                            __nv_bfloat16 *const *D,
                            const void *lddArrayDev,
                            int batchCount,
                            void *workspace,
                            size_t workspaceSize,
                            cublasLtMatmulMatrixScale_t AScaleMode,
                            cublasLtMatmulMatrixScale_t BScaleMode);
