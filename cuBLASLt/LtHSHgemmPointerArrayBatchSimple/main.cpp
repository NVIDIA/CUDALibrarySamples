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


#include "sample_cublasLt_LtHSHgemmPointerArrayBatchSimple.h"
#include "helpers.h"

int main() {
    TestBench<__half, __half, float> props(CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, 2.0f, 0.0f, 4 * 1024 * 1024 * 2, 2, true, true);

    props.run([&props] {
        LtHSHgemmPointerArrayBatchSimple(props.ltHandle,
                                    props.transa,
                                    props.transb,
                                    props.m,
                                    props.n,
                                    props.k,
                                    &props.alpha,
                                    props.APtrArrayDev,
                                    props.lda,
                                    props.BPtrArrayDev,
                                    props.ldb,
                                    &props.beta,
                                    props.CPtrArrayDev,
                                    props.ldc,
                                    props.DPtrArrayDev,
                                    props.ldd,
                                    props.N,
                                    props.workspace,
                                    props.workspaceSize);
    });

    return 0;
}