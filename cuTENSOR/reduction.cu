/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <iostream>
#include <unordered_map>
#include <stdexcept>
#include <memory>
#include <vector>

#include <cutensor.h>

#include "utils.cuh"


int main()
try
{
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.f;

    /**********************
     * Computing (partial) reduction : C_{m,v} = alpha * A_{m,h,k,v} + beta * C_{m,v}
     *********************/

    std::vector<int32_t> modeA{'m','h','k','v'};
    std::vector<int32_t> modeC{'m','v'};
    int32_t nmodeA = modeA.size();
    int32_t nmodeC = modeC.size();

    std::unordered_map<int32_t, int64_t> extent;
    extent['m'] = 196;
    extent['v'] = 64;
    extent['h'] = 256;
    extent['k'] = 64;

    std::vector<int64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);

    /**********************
     * Allocating data
     *********************/

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeC = sizeof(floatTypeC) * elementsC;
    printf("Total memory: %.2f GiB\n",(sizeA + sizeC)/1024./1024./1024);

    auto A_d = cuda_alloc     <floatTypeA>(elementsA); 
    auto C_d = cuda_alloc     <floatTypeC>(elementsC); 
    auto A   = cuda_host_alloc<floatTypeA>(elementsA); 
    auto C   = cuda_host_alloc<floatTypeC>(elementsC); 

    const uint32_t kAlignment = 256; // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d.get()) % kAlignment == 0);
    assert(uintptr_t(C_d.get()) % kAlignment == 0);

    /*******************
     * Initialize data
     *******************/

    std::generate(A.get(),A.get()+elementsA, randomgen<floatTypeA>() );
    std::generate(C.get(),C.get()+elementsC, randomgen<floatTypeC>() );

    handle_error(cudaMemcpy(C_d.get(),C.get(),sizeC,cudaMemcpyHostToDevice));
    handle_error(cudaMemcpy(A_d.get(),A.get(),sizeA,cudaMemcpyHostToDevice));

    /*************************
     * cuTENSOR
     *************************/ 

    cutensorHandle_t handle;
    handle_error(cutensorCreate(&handle)); 
    auto guardHandle = finally( [&handle]() { cutensorDestroy(handle); } );

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t descA;
    handle_error(cutensorCreateTensorDescriptor(handle,
                 &descA,
                 nmodeA,
                 extentA.data(),
                 NULL /* stride */,
                 typeA, kAlignment));
    auto guardDescA = finally( [&descA]() { cutensorDestroyTensorDescriptor(descA); } );

    cutensorTensorDescriptor_t descC;
    handle_error(cutensorCreateTensorDescriptor(handle,
                 &descC,
                 nmodeC,
                 extentC.data(),
                 NULL /* stride */,
                 typeC, kAlignment));
    auto guardDescC = finally( [&descC]() { cutensorDestroyTensorDescriptor(descC); } );

    const cutensorOperator_t opReduce = CUTENSOR_OP_ADD;

    /*******************************
     * Create Reduction Descriptor
     *******************************/

    cutensorOperationDescriptor_t desc;
    handle_error(cutensorCreateReduction(
                 handle, &desc,
                 descA, modeA.data(), CUTENSOR_OP_IDENTITY,
                 descC, modeC.data(), CUTENSOR_OP_IDENTITY,
                 descC, modeC.data(),
                 opReduce, descCompute));
    auto guardDesc = finally( [&desc]() { cutensorDestroyOperationDescriptor(desc); } );

    /**************************
    * Set the algorithm to use
    ***************************/

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t planPref;
    handle_error(cutensorCreatePlanPreference(
                               handle,
                               &planPref,
                               algo,
                               CUTENSOR_JIT_MODE_NONE));
    auto guardPlanPref = finally( [&planPref]() { cutensorDestroyPlanPreference(planPref); } );

    /**********************
     * Query workspace estimate
     **********************/

    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    handle_error(cutensorEstimateWorkspaceSize(handle,
                                          desc,
                                          planPref,
                                          workspacePref,
                                          &workspaceSizeEstimate));

    /**************************
     * Create Contraction Plan
     **************************/

    cutensorPlan_t plan;
    handle_error(cutensorCreatePlan(handle,
                 &plan,
                 desc,
                 planPref,
                 workspaceSizeEstimate));
    auto guardPlan = finally( [&plan]() { cutensorDestroyPlan(plan); } );

    /**************************
     * Optional: Query information about the created plan
     **************************/

    // query actually used workspace
    uint64_t actualWorkspaceSize = 0;
    handle_error(cutensorPlanGetAttribute(handle,
        plan,
        CUTENSOR_PLAN_REQUIRED_WORKSPACE,
        &actualWorkspaceSize,
        sizeof(actualWorkspaceSize)));

    // At this point the user knows exactly how much memory is need by the operation and
    // only the smaller actual workspace needs to be allocated
    assert(actualWorkspaceSize <= workspaceSizeEstimate);

    auto work = cuda_alloc<char>(actualWorkspaceSize);
    assert(uintptr_t(work.get()) % 128 == 0); // workspace must be aligned to 128 byte-boundary

    /**********************
     * Run
     **********************/

    cudaStream_t stream;
    handle_error(cudaStreamCreate(&stream));
    auto guardStream = finally( [&stream]() { cudaStreamDestroy(stream); });

    double minTimeCUTENSOR = 1e100;
    for(int i=0; i < 3; ++i)
    {
        handle_error(cudaMemcpyAsync(C_d.get(),C.get(), sizeC, cudaMemcpyHostToDevice,stream));
        GPUTimer timer(stream);
        handle_error(cutensorReduce(handle, plan,
                (const void*)&alpha, A_d.get(),
                (const void*)&beta,  C_d.get(), 
                                     C_d.get(), work.get(), actualWorkspaceSize, stream));
        auto time = timer.seconds(stream); // Synchronise and measure time.
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
    }

    /*************************/

    double transferedBytes = sizeC + sizeA;
    transferedBytes += ((float) beta != 0.f) ? sizeC : 0;
    transferedBytes /= 1e9;
    printf("cuTensor: %.2f GB/s\n", transferedBytes / minTimeCUTENSOR);

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

