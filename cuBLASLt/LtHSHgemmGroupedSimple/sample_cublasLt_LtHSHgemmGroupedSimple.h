/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
