/*  
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cutensorDataType_t          const typeA       = CUTENSOR_R_32F;
    cutensorDataType_t          const typeB       = CUTENSOR_R_32F;
    cutensorDataType_t          const typeC       = CUTENSOR_R_32F;
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

    void *A_d, *B_d, *C_d, *D_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &A_d, sizeA));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &B_d, sizeB));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &C_d, sizeC));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &D_d, sizeC));

    const uint32_t kAlignment = 256; // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(B_d) % kAlignment == 0);
    assert(uintptr_t(C_d) % kAlignment == 0);
    assert(uintptr_t(D_d) % kAlignment == 0);

    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

    if (A == nullptr || B == nullptr || C == nullptr)
    {
        printf("Error: Host allocation of A, B, or C.\n");
        return -1;
    }

    /*******************
     * Initialize data
     *******************/

    for (size_t i = 0; i < elementsA; i++)
        A[i] = (((float) rand())/RAND_MAX)*100;
    for (size_t i = 0; i < elementsB; i++)
        B[i] = (((float) rand())/RAND_MAX)*100;
    for (size_t i = 0; i < elementsC; i++)
        C[i] = (((float) rand())/RAND_MAX)*100;

    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(C_d, sizeC, C, sizeC, sizeC, 1, cudaMemcpyDefault, nullptr));
    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(D_d, sizeC, C, sizeC, sizeC, 1, cudaMemcpyDefault, nullptr));
    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(A_d, sizeA, A, sizeA, sizeA, 1, cudaMemcpyDefault, nullptr));
    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(B_d, sizeB, B, sizeB, sizeB, 1, cudaMemcpyDefault, nullptr));

    /*************************
     * Memcpy perf 
     *************************/

    double minTimeMEMCPY = 1e100;
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    GPUTimer timer;
    timer.start();
    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(D_d, sizeC, C_d, sizeC, sizeC, 1, cudaMemcpyDefault, nullptr));
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    minTimeMEMCPY = timer.seconds();

    /*************************
     * cuTENSOR
     *************************/

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t  descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descA, nmodeA, extentA.data(),
                                                nullptr /* stride */,
                                                typeA,
                                                kAlignment));

    cutensorTensorDescriptor_t  descB;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descB, nmodeB, extentB.data(),
                                                nullptr /* stride */,
                                                typeB,
                                                kAlignment));

    cutensorTensorDescriptor_t  descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descC, nmodeC, extentC.data(),
                                                nullptr /* stride */,
                                                typeC,
                                                kAlignment));

    /*******************************
     * Create Elementwise Trinary Descriptor
     *******************************/

    cutensorOperationDescriptor_t desc;
    HANDLE_ERROR(cutensorCreateElementwiseTrinary(handle, 
                                                  &desc,
                                                  descA, modeA.data(), /* unary operator A */ CUTENSOR_OP_IDENTITY,
                                                  descB, modeB.data(), /* unary operator B */ CUTENSOR_OP_IDENTITY,
                                                  descC, modeC.data(), /* unary operator C */ CUTENSOR_OP_IDENTITY,
                                                  descC, modeC.data(),
                                                  /* binary operator AC  */ CUTENSOR_OP_ADD,
                                                  /* binary operator ABC */ CUTENSOR_OP_ADD,
                                                  descCompute));

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
                                    0 /*workspaceSizeEstimate*/));

    /**********************
     * Run
     **********************/

    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

    double minTimeCUTENSOR = 1e100;
    for (int i = 0; i < 3; i++)
    {
        HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(D_d, sizeC, C, sizeC, sizeC, 1, cudaMemcpyDefault, nullptr));
        timer.start();
        HANDLE_ERROR(cutensorElementwiseTrinaryExecute(handle, plan,
                                                (void*)&alpha, A_d,
                                                (void*)&beta , B_d,
                                                (void*)&gamma, C_d,
                                                               D_d, stream));
        auto time = timer.seconds();
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

    HANDLE_ERROR(cutensorDestroy(handle));
    HANDLE_ERROR(cutensorDestroyPlan(plan));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);

    return 0;
}
