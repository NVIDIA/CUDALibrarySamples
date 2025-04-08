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

#include <cuda_fp8.h>
#include <cuda_bf16.h>

#include "sample_cublasLt_LtMxfp8Matmul.h"
#include "helpers.h"

int main() {
    TestBench<__nv_fp8_e4m3, __nv_fp8_e4m3, float, __nv_fp8_e8m0, __nv_fp8_e8m0, __nv_bfloat16> props(
        64, 128, 256, 2.0f, 1.0f, 32ULL * 1024 * 1024, 1,
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);

    props.run([&props] {
        LtMxfp8Matmul(props.ltHandle,
                    props.m,
                    props.n,
                    props.k,
                    &props.alpha,
                    props.AscaleDev,
                    props.Adev,
                    props.k,
                    props.BscaleDev,
                    props.Bdev,
                    props.k,
                    &props.beta,
                    props.CscaleDev,
                    props.Cdev,
                    props.m,
                    props.Ddev,
                    props.m,
                    props.DOutscaleDev,
                    props.workspace,
                    props.workspaceSize,
                    props.AScaleMode,
                    props.BScaleMode,
                    props.CScaleMode,
                    props.DOutScaleMode);
    });

    return 0;
}