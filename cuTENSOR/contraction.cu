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

#include <assert.h>
#include <stdlib.h>

#include <vector>
#include <iostream>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cutensor.h>

#include "utils.cuh"

int main()
try
{
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;

    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

    /**********************
     * Computing: C_{m,u,n,v} = alpha * A_{m,h,k,n} B_{u,k,v,h} + beta * C_{m,u,n,v}
     **********************/

    std::vector<int> modeC{'m','u','n','v'};
    std::vector<int> modeA{'m','h','k','n'};
    std::vector<int> modeB{'u','k','v','h'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    extent['m'] = 96;
    extent['n'] = 96;
    extent['u'] = 96;
    extent['v'] = 64;
    extent['h'] = 64;
    extent['k'] = 64;

    double gflops = (2.0 * extent['m'] * extent['n'] * extent['u'] * extent['v'] * extent['k'] * extent['h']) /1e9;

    std::vector<int64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentB;
    for (auto mode : modeB)
        extentB.push_back(extent[mode]);

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
   
    auto A   = cuda_host_alloc<floatTypeA>(elementsA); 
    auto B   = cuda_host_alloc<floatTypeB>(elementsB); 
    auto C   = cuda_host_alloc<floatTypeC>(elementsC); 

    /*******************
     * Initialize data
     *******************/

    // Random data.
    std::generate(A.get(),A.get()+elementsA, randomgen<floatTypeA>() );
    std::generate(B.get(),B.get()+elementsB, randomgen<floatTypeB>() );
    std::generate(C.get(),C.get()+elementsC, randomgen<floatTypeC>() );

    handle_error(cudaMemcpy(A_d.get(), A.get(), sizeA, cudaMemcpyHostToDevice));
    handle_error(cudaMemcpy(B_d.get(), B.get(), sizeB, cudaMemcpyHostToDevice));
    handle_error(cudaMemcpy(C_d.get(), C.get(), sizeC, cudaMemcpyHostToDevice));

    const uint32_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d.get()) % kAlignment == 0);
    assert(uintptr_t(B_d.get()) % kAlignment == 0);
    assert(uintptr_t(C_d.get()) % kAlignment == 0);

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
                 NULL,/*stride*/
                 typeA, kAlignment));
    auto guardDescA = finally( [&descA]() { cutensorDestroyTensorDescriptor(descA); } );

    cutensorTensorDescriptor_t descB;
    handle_error(cutensorCreateTensorDescriptor(handle,
                 &descB,
                 nmodeB,
                 extentB.data(),
                 NULL,/*stride*/
                 typeB, kAlignment));
    auto guardDescB = finally( [&descB]() { cutensorDestroyTensorDescriptor(descB); } );

    cutensorTensorDescriptor_t descC;
    handle_error(cutensorCreateTensorDescriptor(handle,
                 &descC,
                 nmodeC,
                 extentC.data(),
                 NULL,/*stride*/
                 typeC, kAlignment));
    auto guardDescC = finally( [&descC]() { cutensorDestroyTensorDescriptor(descC); } );

    /*******************************
     * Create Contraction Descriptor
     *******************************/

    cutensorOperationDescriptor_t desc;
    handle_error(cutensorCreateContraction(handle, 
                 &desc,
                 descA, modeA.data(), /* unary operator A*/CUTENSOR_OP_IDENTITY,
                 descB, modeB.data(), /* unary operator B*/CUTENSOR_OP_IDENTITY,
                 descC, modeC.data(), /* unary operator C*/CUTENSOR_OP_IDENTITY,
                 descC, modeC.data(),
                 descCompute));
    auto guardDesc = finally( [&desc]() { cutensorDestroyOperationDescriptor(desc); } );

    /*****************************
     * Optional (but recommended): ensure that the scalar type is correct.
     *****************************/

    cudaDataType_t scalarType;
    handle_error(cutensorOperationDescriptorGetAttribute(handle,
        desc,
        CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
        (void*)&scalarType,
        sizeof(scalarType)));

    assert(scalarType == CUDA_R_32F);
    typedef float floatTypeCompute;
    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.f;

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
    auto guardStream = finally( [&stream]() { cudaStreamDestroy(stream); } );

    double minTimeCUTENSOR = 1e100;
    for (int i=0; i < 3; ++i)
    {
        handle_error(cudaMemcpy(C_d.get(), C.get(), sizeC, cudaMemcpyHostToDevice));
        handle_error(cudaDeviceSynchronize());

        // Set up timing
        GPUTimer timer(stream);

        handle_error(cutensorContract(handle,
                               plan,
                               (void*) &alpha, A_d.get(), B_d.get(),
                               (void*) &beta,  C_d.get(), C_d.get(), 
                               work.get(), actualWorkspaceSize, stream));

        // Synchronize and measure timing
        auto time = timer.seconds(stream);
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
    }

    /*************************/

    double transferedBytes = sizeC + sizeA + sizeB;
    transferedBytes += ((float) beta != 0.f) ? sizeC : 0;
    transferedBytes /= 1e9;
    printf("cuTensor: %.2f GFLOPs/s %.2f GB/s\n", gflops / minTimeCUTENSOR, transferedBytes/ minTimeCUTENSOR);

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

