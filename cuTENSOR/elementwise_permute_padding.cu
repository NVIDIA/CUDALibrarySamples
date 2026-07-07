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
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cudaDataType_t              const typeA       = CUDA_R_32F;
    cudaDataType_t              const typeC       = CUDA_R_32F;
    cutensorComputeDescriptor_t const descCompute = CUTENSOR_COMPUTE_DESC_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.0f;

    /**********************
     * This example illustrates the use case where an input tensor A (in host memory) is
     * permuted from an NCHW data layout to NHWC while moving the data from host to device
     * memory C. It also adds padding for 'w' and 'h' modes:
     *
     * C_{c',w',h,n} = alpha * A_{w,h,c,n}
     **********************/

    std::vector<int> modeC{'c','w','h','n'};
    std::vector<int> modeA{'w','h','c','n'};
    int nmodeA = modeA.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    extent['h'] = 128;
    extent['w'] = 32;
    extent['c'] = 128;
    extent['n'] = 128;

    std::unordered_map<int, int> paddingLeft;
    paddingLeft['h'] = 1;
    paddingLeft['w'] = 1;
    paddingLeft['c'] = 0;
    paddingLeft['n'] = 0;

    std::unordered_map<int, int> paddingRight;
    paddingLeft['h'] = 1;
    paddingLeft['w'] = 1;
    paddingLeft['c'] = 0;
    paddingLeft['n'] = 0;

    floatTypeC paddingValue = 0.0f;

    std::vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentC;
    std::vector<int> paddingLeftVec;
    std::vector<int> paddingRightVec;
    for (auto mode : modeC)
    {
        extentC.push_back(extent[mode]);
        paddingLeftVec.push_back(paddingLeft[mode]);
        paddingRightVec.push_back(paddingRight[mode]);
    }


    /**********************
     * Allocating data
     **********************/

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode] + paddingLeft[mode] + paddingRight[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeC = sizeof(floatTypeC) * elementsC;
    printf("Total memory: %.2f GiB\n", (sizeA + sizeC)/1024./1024./1024);


    auto A_d = cuda_alloc     <floatTypeA>(elementsA); 
    auto C_d = cuda_alloc     <floatTypeC>(elementsC);

    auto A   = cuda_host_alloc<floatTypeA>(elementsA);
    auto C   = cuda_host_alloc<floatTypeC>(elementsC);

    uint32_t const kAlignment = 128;  // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d.get()) % kAlignment == 0);
    assert(uintptr_t(C_d.get()) % kAlignment == 0);


    /*******************
     * Initialize data
     *******************/

    std::generate(A.get(),A.get()+elementsA, randomgen<floatTypeA>() );
    handle_error(cudaMemcpy2DAsync(A_d.get(), sizeA, A.get(), sizeA, sizeA, 1, cudaMemcpyHostToDevice, nullptr));

    /*************************
     * CUTENSOR
     *************************/

    cutensorHandle_t handle;
    handle_error(cutensorCreate(&handle));
    auto guardHandle = finally( [&handle]() { cutensorDestroy(handle); } );

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t  descA;
    handle_error(cutensorCreateTensorDescriptor(handle,
                                                &descA,
                                                nmodeA,
                                                extentA.data(),
                                                nullptr /* stride */,
                                                typeA,
                                                kAlignment));
    auto guardDescA = finally( [&descA]() { cutensorDestroyTensorDescriptor(descA); } );

    cutensorTensorDescriptor_t  descC;
    handle_error(cutensorCreateTensorDescriptor(handle,
                                                &descC,
                                                nmodeC,
                                                extentC.data(),
                                                nullptr /* stride */,
                                                typeC,
                                                kAlignment));
    auto guardDescC = finally( [&descC]() { cutensorDestroyTensorDescriptor(descC); } );

    /*******************************
     * Create Permutation Descriptor
     *******************************/

    cutensorOperationDescriptor_t  desc;
    handle_error(cutensorCreatePermutation(handle,
                                           &desc,
                                           descA,
                                           modeA.data(),
                                           CUTENSOR_OP_IDENTITY,
                                           descC,
                                           modeC.data(),
                                           descCompute));
    auto guardDesc = finally( [&desc]() { cutensorDestroyOperationDescriptor(desc); } );

    /*******************************
     * Set Padding Information
     *******************************/
    handle_error(cutensorOperationDescriptorSetAttribute(handle,
                                                         desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT,
                                                         paddingLeftVec.data(),
                                                         sizeof(int) * nmodeC));

    handle_error(cutensorOperationDescriptorSetAttribute(handle,
                                                         desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT,
                                                         paddingRightVec.data(),
                                                         sizeof(int) * nmodeC));

    handle_error(cutensorOperationDescriptorSetAttribute(handle,
                                                         desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE,
                                                         &paddingValue,
                                                         sizeof(paddingValue)));

    /*****************************
     * Optional (but recommended): ensure that the scalar type is correct.
     *****************************/

    cudaDataType_t scalarType;
    handle_error(cutensorOperationDescriptorGetAttribute(handle, desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                         (void*)&scalarType,
                                                         sizeof(scalarType)));

    assert(scalarType == CUDA_R_32F);


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
                                    0 /* workspaceSizeLimit */));
    auto guardPlan = finally( [&plan]() { cutensorDestroyPlan(plan); } );

    /**********************
     * Run
     **********************/

    double minTimeCUTENSOR = 1e100;
    for (int i = 0; i < 3; i++)
    {
        GPUTimer timer;

        handle_error(cutensorPermute(handle,
                        plan,
                        &alpha, A_d.get(), C_d.get(), nullptr /* stream */));

        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
    }

    /*************************/

    double transferedBytes = 2.0 * sizeC;
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

