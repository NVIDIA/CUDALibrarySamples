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


#include <cuComplex.h>

#include "sample_cublasLt_LtPlanarComplex.h"
#include "helpers.h"

int main() {
    TestBench<__half, __half, cuComplex, __half> props(CUBLAS_OP_N, CUBLAS_OP_N, 16, 16, 16, {1.0f, 0}, {0.0f, 0}, 0, 2);

    // planar layout is ordered with imaginary first, to prove that this is arbitrary

    // real and imaginary pointers are arbitrary. pointers are converted to pointer(64bit)+offset(int64_t) (negative here) in the example function
    props.run([&props] {
        LtPlanarCgemm(props.ltHandle,
                props.transa,
                props.transb,
                props.m,
                props.n,
                props.k,
                props.Adev+props.m*props.k, // see comment above, real part follows after imaginary in this example; 
                props.Adev,
                props.lda,
                props.Bdev+props.n*props.k,
                props.Bdev,
                props.ldb,
                props.Cdev+props.n*props.k,
                props.Cdev,
                props.ldc);
    });

    return 0;
}