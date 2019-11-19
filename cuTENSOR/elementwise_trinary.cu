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

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>

#define HANDLE_ERROR(x)                                               \
{ const auto err = x;                                                 \
  if( err != CUTENSOR_STATUS_SUCCESS )                                \
  { printf("Error: %s\n", cutensorGetErrorString(err)); return err; } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("Error: %s\n", cudaGetErrorString(err)); return err; } \
};

struct GPUTimer
{
    GPUTimer() 
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    ~GPUTimer() 
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() 
    {
        cudaEventRecord(start_, 0);
    }

    float seconds() 
    {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return time * 1e-3;
    }
    private:
    cudaEvent_t start_, stop_;
};


int main(int argc, char** argv)
{
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cudaDataType_t typeCompute = CUDA_R_32F;

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

    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

    if (A == NULL || B == NULL || C == NULL)
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

    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(C_d, sizeC, C, sizeC, sizeC, 1, cudaMemcpyDefault, 0));
    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(D_d, sizeC, C, sizeC, sizeC, 1, cudaMemcpyDefault, 0));
    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(A_d, sizeA, A, sizeA, sizeA, 1, cudaMemcpyDefault, 0));
    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(B_d, sizeB, B, sizeB, sizeB, 1, cudaMemcpyDefault, 0));

    /*************************
     * Memcpy perf 
     *************************/

    double minTimeMEMCPY = 1e100;
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    GPUTimer timer;
    timer.start();
    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(D_d, sizeC, C_d, sizeC, sizeC, 1, cudaMemcpyDefault, 0));
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    minTimeMEMCPY = timer.seconds();

    /*************************
     * cuTENSOR
     *************************/

    cutensorStatus_t err;
    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorInit(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                 &descA,
                 nmodeA,
                 extentA.data(),
                 NULL /* stride */,
                 typeA, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descB;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                 &descB,
                 nmodeB,
                 extentB.data(),
                 NULL /* stride */,
                 typeB, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                 &descC,
                 nmodeC,
                 extentC.data(),
                 NULL /* stride */,
                 typeC, CUTENSOR_OP_IDENTITY));

    /*************************/

    double minTimeCUTENSOR = 1e100;
    for (int i = 0; i < 3; i++)
    {
        HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(D_d, sizeC, C, sizeC, sizeC, 1, cudaMemcpyDefault, 0));
        timer.start();
        err = cutensorElementwiseTrinary(&handle,
                (void*)&alpha, A_d, &descA, modeA.data(),
                (void*)&beta , B_d, &descB, modeB.data(),
                (void*)&gamma, C_d, &descC, modeC.data(),
                               D_d, &descC, modeC.data(),
                CUTENSOR_OP_ADD, CUTENSOR_OP_ADD, typeCompute, 0 /* stream */);
        auto time = timer.seconds();
        if(err != CUTENSOR_STATUS_SUCCESS)
            printf("ERROR: %s\n", cutensorGetErrorString(err) );
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

    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);

    return 0;
}
