/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>

#include "utils.cuh"


int main()
try
{
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cudaDataType_t              const typeA       = CUDA_R_32F;
    cudaDataType_t              const typeB       = CUDA_R_32F;
    cudaDataType_t              const typeC       = CUDA_R_32F;
    cutensorComputeDescriptor_t const descCompute = CUTENSOR_COMPUTE_DESC_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)1.3f;;
    floatTypeCompute gamma = (floatTypeCompute)1.2f;

    /**********************
     * Computing: D_{a,b,c} = alpha * A_{b,a,c} + beta * B_{c,a,b} + gamma * C_{a,b,c}
     **********************/

    std::vector<int> modeC{'a','b','c'};
    std::vector<int> modeA{'c','b','a'};
    std::vector<int> modeB{'c','a','b'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    extent['a'] = 400;
    extent['b'] = 200;
    extent['c'] = 300;
    std::vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentB;
    for (auto mode : modeB)
        extentB.push_back(extent[mode]);
    std::vector<int64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);

    /**********************
     * Allocating data
     **********************/

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsB = 1;
    for (auto mode : modeB)
        elementsB *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeB = sizeof(floatTypeB) * elementsB;
    size_t sizeC = sizeof(floatTypeC) * elementsC;
    printf("Total memory: %.2f GiB\n", (sizeA + sizeB + sizeC)/1024./1024./1024);

    auto A_d = cuda_alloc     <floatTypeA>(elementsA); 
    auto B_d = cuda_alloc     <floatTypeB>(elementsB); 
    auto C_d = cuda_alloc     <floatTypeC>(elementsC); 
    auto D_d = cuda_alloc     <floatTypeC>(elementsC); 

    auto A   = cuda_host_alloc<floatTypeA>(elementsA); 
    auto B   = cuda_host_alloc<floatTypeB>(elementsB); 
    auto C   = cuda_host_alloc<floatTypeC>(elementsC); 

    const uint32_t kAlignment = 256; // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d.get()) % kAlignment == 0);
    assert(uintptr_t(B_d.get()) % kAlignment == 0);
    assert(uintptr_t(C_d.get()) % kAlignment == 0);
    assert(uintptr_t(D_d.get()) % kAlignment == 0);

    /*******************
     * Initialize data
     *******************/

    std::generate(A.get(),A.get()+elementsA, randomgen<floatTypeA>() );
    std::generate(B.get(),B.get()+elementsB, randomgen<floatTypeB>() );
    std::generate(C.get(),C.get()+elementsC, randomgen<floatTypeC>() );

    handle_error(cudaMemcpy2DAsync(C_d.get(), sizeC, C.get(), sizeC, sizeC, 1, cudaMemcpyHostToDevice, nullptr));
    handle_error(cudaMemcpy2DAsync(D_d.get(), sizeC, C.get(), sizeC, sizeC, 1, cudaMemcpyHostToDevice, nullptr));
    handle_error(cudaMemcpy2DAsync(A_d.get(), sizeA, A.get(), sizeA, sizeA, 1, cudaMemcpyHostToDevice, nullptr));
    handle_error(cudaMemcpy2DAsync(B_d.get(), sizeB, B.get(), sizeB, sizeB, 1, cudaMemcpyHostToDevice, nullptr));

    /*************************
     * Memcpy perf 
     *************************/

    double minTimeMEMCPY = 1e100;
    handle_error(cudaDeviceSynchronize());
    GPUTimer timer;
    handle_error(cudaMemcpy2DAsync(D_d.get(), sizeC, C_d.get(), sizeC, sizeC, 1, cudaMemcpyDeviceToDevice, nullptr));
    minTimeMEMCPY = timer.seconds(); // timer synchronizes.

    /*************************
     * cuTENSOR
     *************************/

    cutensorHandle_t handle;
    handle_error(cutensorCreate(&handle));
    auto guardHandle = finally( [&handle]() { cutensorDestroy(handle); } );

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t  descA;
    handle_error(cutensorCreateTensorDescriptor(handle,
                                                &descA, nmodeA, extentA.data(),
                                                nullptr /* stride */,
                                                typeA,
                                                kAlignment));
    auto guardDescA = finally( [&descA]() { cutensorDestroyTensorDescriptor(descA); } );

    cutensorTensorDescriptor_t  descB;
    handle_error(cutensorCreateTensorDescriptor(handle,
                                                &descB, nmodeB, extentB.data(),
                                                nullptr /* stride */,
                                                typeB,
                                                kAlignment));
    auto guardDescB = finally( [&descB]() { cutensorDestroyTensorDescriptor(descB); } );

    cutensorTensorDescriptor_t  descC;
    handle_error(cutensorCreateTensorDescriptor(handle,
                                                &descC, nmodeC, extentC.data(),
                                                nullptr /* stride */,
                                                typeC,
                                                kAlignment));
    auto guardDescC = finally( [&descC]() { cutensorDestroyTensorDescriptor(descC); } );

    /*******************************
     * Create Elementwise Trinary Descriptor
     *******************************/

    cutensorOperationDescriptor_t desc;
    handle_error(cutensorCreateElementwiseTrinary(handle, 
                                                  &desc,
                                                  descA, modeA.data(), /* unary operator A */ CUTENSOR_OP_IDENTITY,
                                                  descB, modeB.data(), /* unary operator B */ CUTENSOR_OP_IDENTITY,
                                                  descC, modeC.data(), /* unary operator C */ CUTENSOR_OP_IDENTITY,
                                                  descC, modeC.data(),
                                                  /* binary operator AC  */ CUTENSOR_OP_ADD,
                                                  /* binary operator ABC */ CUTENSOR_OP_ADD,
                                                  descCompute));
    auto guardDesc = finally( [desc]() { cutensorDestroyOperationDescriptor(desc); } );

    /**************************
    * Set the algorithm to use
    ***************************/

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t  planPref;
    handle_error(cutensorCreatePlanPreference(handle,
                                              &planPref,
                                              algo,
                                              CUTENSOR_JIT_MODE_NONE));
    auto guardPlanPref = finally( [&planPref]() { cutensorDestroyPlanPreference(planPref); } );

    /**************************
     * Create Plan
     **************************/

    cutensorPlan_t  plan;
    handle_error(cutensorCreatePlan(handle,
                                    &plan,
                                    desc,
                                    planPref,
                                    0 /*workspaceSizeEstimate*/));
    auto guardPlan = finally( [&plan]() { cutensorDestroyPlan(plan); } );

    /**********************
     * Run
     **********************/

    cudaStream_t stream;
    handle_error(cudaStreamCreate(&stream));
    auto guardStream = finally( [&stream](){ cudaStreamDestroy(stream); } );

    double minTimeCUTENSOR = 1e100;
    for (int i = 0; i < 3; i++)
    {
        handle_error(cudaMemcpy2DAsync(D_d.get(), sizeC, C.get(), sizeC, sizeC, 1, cudaMemcpyHostToDevice, stream));
        timer.start(stream);
        handle_error(cutensorElementwiseTrinaryExecute(handle, plan,
                                                (void*)&alpha, A_d.get(),
                                                (void*)&beta , B_d.get(),
                                                (void*)&gamma, C_d.get(),
                                                               D_d.get(), stream));
        auto time = timer.seconds(stream);
        minTimeCUTENSOR = (minTimeCUTENSOR < time)? minTimeCUTENSOR : time;
    }

    /*************************/

    double transferedBytes = sizeC;
    transferedBytes += ((float) alpha != 0.f) ? sizeA : 0;
    transferedBytes += ((float) beta != 0.f) ? sizeB : 0;
    transferedBytes += ((float) gamma != 0.f) ? sizeC : 0;
    transferedBytes /= 1e9;
    printf("cuTensor: %.2f GB/s\n", transferedBytes/ minTimeCUTENSOR);
    printf("memcpy: %.2f GB/s\n", 2 * sizeC / minTimeMEMCPY / 1e9 );

    return EXIT_SUCCESS;
}
catch ( std::exception &ex )
{
    std::cerr << "Exception caught! Exiting." << std::endl;
    std::cerr << ex.what() << std::endl;
    return EXIT_FAILURE;
}
catch ( ... )
{
    std::cerr << "Unknown exception caught! Exiting." << std::endl;
    return EXIT_FAILURE;
}

