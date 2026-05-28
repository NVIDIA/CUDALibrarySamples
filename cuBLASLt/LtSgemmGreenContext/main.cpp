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

#include "sample_cublasLt_LtSgemmGreenCtx.h"
#include "helpers.h"

int main() {
    // user defined green context SM count, has to be lower or equal than device SMs
    const unsigned int greenContextSmCount = 16;

    TestBench<float> props(CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, 2.0f, 0.0f);

    props.run([&props] {
        LtSgemmGCtx(props.ltHandle, props.transa, props.transb, props.m, props.n, props.k, &props.alpha, props.Adev,
                    props.lda, props.Bdev, props.ldb, &props.beta, props.Cdev, props.ldc, props.workspace,
                    props.workspaceSize, greenContextSmCount, props.stream);
    });

    return 0;
}
