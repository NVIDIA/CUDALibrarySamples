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

#include "sample_cublasLt_LtHSHgemmGroupedSimple.h"
#include "helpers.h"

int main() {
    TestBench<__half, __half, float> props(CUBLAS_OP_N, CUBLAS_OP_N, 64, 96, 128, 2.0f, 0.0f, 4 * 1024 * 1024 * 2, 2,
                                           false, true, true);

    props.run([&props] {
        LtHSHgemmGroupedSimple(props.ltHandle, props.transa, props.transb, props.mArrayDev, props.avgM, props.nArrayDev,
                               props.avgN, props.kArrayDev, props.avgK, props.alphaArrayDev, props.APtrArrayDev,
                               props.ldaArrayDev, props.BPtrArrayDev, props.ldbArrayDev, props.betaArrayDev,
                               props.CPtrArrayDev, props.ldcArrayDev, props.DPtrArrayDev, props.lddArrayDev, props.N,
                               props.workspace, props.workspaceSize);
    });

    return 0;
}
