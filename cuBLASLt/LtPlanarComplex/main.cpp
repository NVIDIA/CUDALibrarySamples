/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <vector>

#include <cuda_runtime_api.h>
#include <cublasLt.h>

#include "sample_cublasLt_LtPlanarComplex.h"
#include "helpers.h"

int main() {
    TestBench<__half, __half, cuComplex> props(16, 16, 16, {1.0f, 0}, {0.0f, 0}, 0, 2);

    // planar layout is ordered with imaginary first, to prove that this is arbitrary

    // real and imaginary pointers are arbitrary. pointers are converted to pointer(64bit)+offset(int64_t) (negative here) in the example function
    props.run([&props] {
        LtPlanarCgemm(props.ltHandle,
                props.m,
                props.n,
                props.k,
                props.Adev+props.m*props.k, // see comment above, real part follows after imaginary in this example; 
                props.Adev,
                props.m,
                props.Bdev+props.n*props.k,
                props.Bdev,
                props.k,
                props.Cdev+props.n*props.k,
                props.Cdev,
                props.m);
    });

    return 0;
}