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
