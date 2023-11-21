/*  
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 * 
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name(s) of the copyright holder(s) nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
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

#define HANDLE_ERROR(x)                                                   \
{ auto const __err = x;                                                   \
  if( __err != CUTENSOR_STATUS_SUCCESS )                                  \
  { printf("Error: %d %s\n", __LINE__, cutensorGetErrorString(__err)); exit(-1); } \
};

#define HANDLE_CUDA_ERROR(x)                                          \
{ auto const __err = x;                                               \
  if( __err != cudaSuccess )                                          \
  { printf("Error: %d %s\n", __LINE__, cudaGetErrorString(__err)); exit(-1); } \
};

struct GPUTimer
{
    GPUTimer() 
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, nullptr);
    }

    ~GPUTimer() 
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() 
    {
        cudaEventRecord(start_, nullptr);
    }

    float seconds() 
    {
        cudaEventRecord(stop_, nullptr);
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return static_cast<float>(time * 1e-3);
    }

private:
    cudaEvent_t start_, stop_;
};

int main()
{
    typedef float floatTypeA;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cutensorDataType_t          const typeA       = CUTENSOR_R_32F;
    cutensorDataType_t          const typeC       = CUTENSOR_R_32F;
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

    void *A_d, *C_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &A_d, sizeA));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &C_d, sizeC));

    uint32_t const kAlignment = 128;  // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(C_d) % kAlignment == 0);

    floatTypeA *A, *C;
    HANDLE_CUDA_ERROR(cudaMallocHost((void**) &A, sizeof(floatTypeA) * elementsA));
    HANDLE_CUDA_ERROR(cudaMallocHost((void**) &C, sizeof(floatTypeC) * elementsC));

    if (A == nullptr || C == nullptr)
    {
        printf("Error: Host allocation of A or C.\n");
        return -1;
    }

    /*******************
     * Initialize data
     *******************/

    for (size_t i = 0; i < elementsA; i++)
    {
        A[i] = (((float) rand())/RAND_MAX)*100;
    }
    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(A_d, sizeA, A, sizeA, sizeA, 1, cudaMemcpyDefault, nullptr));

    /*************************
     * CUTENSOR
     *************************/

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t  descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descA,
                                                nmodeA,
                                                extentA.data(),
                                                nullptr /* stride */,
                                                typeA,
                                                kAlignment));

    cutensorTensorDescriptor_t  descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descC,
                                                nmodeC,
                                                extentC.data(),
                                                nullptr /* stride */,
                                                typeC,
                                                kAlignment));

    /*******************************
     * Create Permutation Descriptor
     *******************************/

    cutensorOperationDescriptor_t  desc;
    HANDLE_ERROR(cutensorCreatePermutation(handle,
                                           &desc,
                                           descA,
                                           modeA.data(),
                                           CUTENSOR_OP_IDENTITY,
                                           descC,
                                           modeC.data(),
                                           descCompute));

    /*******************************
     * Set Padding Information
     *******************************/
    HANDLE_ERROR(cutensorOperationDescriptorSetAttribute(handle,
                                                         desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT,
                                                         paddingLeftVec.data(),
                                                         sizeof(int) * nmodeC));

    HANDLE_ERROR(cutensorOperationDescriptorSetAttribute(handle,
                                                         desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT,
                                                         paddingRightVec.data(),
                                                         sizeof(int) * nmodeC));

    HANDLE_ERROR(cutensorOperationDescriptorSetAttribute(handle,
                                                         desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE,
                                                         &paddingValue,
                                                         sizeof(paddingValue)));

    /*****************************
     * Optional (but recommended): ensure that the scalar type is correct.
     *****************************/

    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle, desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                         (void*)&scalarType,
                                                         sizeof(scalarType)));

    assert(scalarType == CUTENSOR_R_32F);


    /**************************
    * Set the algorithm to use
    ***************************/

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t  planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(handle,
                                              &planPref,
                                              algo,
                                              CUTENSOR_JIT_MODE_NONE));

    /**************************
     * Create Plan
     **************************/

    cutensorPlan_t  plan;
    HANDLE_ERROR(cutensorCreatePlan(handle,
                                    &plan,
                                    desc,
                                    planPref,
                                    0 /* workspaceSizeLimit */));

    /**********************
     * Run
     **********************/

    double minTimeCUTENSOR = 1e100;
    for (int i = 0; i < 3; i++)
    {
        GPUTimer timer;
        timer.start();

        HANDLE_ERROR(cutensorPermute(handle,
                        plan,
                        &alpha, A_d, C_d, nullptr /* stream */));

        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
    }

    /*************************/

    double transferedBytes = 2.0 * sizeC;
    transferedBytes /= 1e9;
    printf("cuTensor: %.2f GB/s\n", transferedBytes / minTimeCUTENSOR);

    HANDLE_ERROR(cutensorDestroy(handle));
    HANDLE_ERROR(cutensorDestroyPlan(plan));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_ERROR(cutensorDestroyPlanPreference(planPref));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));

    if (A) cudaFreeHost(A);
    if (C) cudaFreeHost(C);
    if (A_d) cudaFree(A_d);
    if (C_d) cudaFree(C_d);

    return 0;
}
