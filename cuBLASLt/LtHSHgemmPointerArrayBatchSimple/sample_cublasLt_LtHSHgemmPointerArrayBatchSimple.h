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

/// Sample wrapper executing mixed precision gemm with cublasLtMatmul, nearly a drop-in replacement for cublasGemmEx,
/// with addition of the workspace to support split-K algorithms
///
/// pointer mode is always host, to change it configure the appropriate matmul descriptor attribute
/// matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed
void LtHSHgemmPointerArrayBatchSimple(cublasLtHandle_t ltHandle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb,
                                      int m,
                                      int n,
                                      int k,
                                      const float *alpha, /* host pointer */
                                      const __half *const *A,
                                      int lda,
                                      const __half *const *B,
                                      int ldb,
                                      const float *beta, /* host pointer */
                                      const __half *const *C,
                                      int ldc,
                                      __half *const *D,
                                      int ldd,
                                      int batchCount,
                                      void *workspace,
                                      size_t workspaceSize);
