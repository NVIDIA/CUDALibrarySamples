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

/// Use cublasLtMatmul to perform tensor-op Igemm with memory order transforms on all buffers
///
/// For better performance data order transforms should be offline as much as possible.
///
/// transa, transb assumed N; alpha, beta are host pointers, tensor ops allowed, alpha assumed 1, beta assumed 0,
/// stream assumed 0
void LtIgemmTensor(cublasLtHandle_t ltHandle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const int8_t *A,
                   int lda,
                   const int8_t *B,
                   int ldb,
                   int32_t *C,
                   int ldc);