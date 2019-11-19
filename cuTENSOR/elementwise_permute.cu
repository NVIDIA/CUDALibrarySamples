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
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cudaDataType_t typeCompute = CUDA_R_32F;

    /**********************
     * This example illustrates the use case where an input tensor A (in host memory) is
     * permuted from an NCHW data layout to NHWC while moving the data from host to device
     * memory C:
     *
     * C_{c,w,h,n} = A_{w,h,c,n}
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

    std::vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);

    /**********************
     * Allocating data
     **********************/

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    void *A_d, *C_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &A_d, sizeA));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &C_d, sizeC));

    floatTypeA *A, *C;
    HANDLE_CUDA_ERROR(cudaMallocHost((void**) &A, sizeof(floatTypeA) * elementsA));
    HANDLE_CUDA_ERROR(cudaMallocHost((void**) &C, sizeof(floatTypeC) * elementsC));

    /*******************
     * Initialize data
     *******************/

    for (size_t i = 0; i < elementsA; i++)
    {
        A[i] = (((float) rand())/RAND_MAX)*100;
    }
    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(A_d, sizeA, A, sizeA, sizeA, 1, cudaMemcpyDefault, 0));

    /*************************
     * CUTENSOR
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

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                 &descC,
                 nmodeC,
                 extentC.data(),
                 NULL /* stride */,
                 typeC, CUTENSOR_OP_IDENTITY));

    double minTimeCUTENSOR = 1e100;
    for (int i = 0; i < 3; i++)
    {
        GPUTimer timer;
        timer.start();
        const floatTypeCompute one = 1.0f;

        err = cutensorPermutation(&handle,
                 &one, A_d, &descA, modeA.data(),
                       C_d, &descC, modeC.data(),
                 typeCompute, 0 /* stream */);

        auto time = timer.seconds();
        if (err != CUTENSOR_STATUS_SUCCESS)
            printf("ERROR: %s\n", cutensorGetErrorString(err));
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
    }

    /*************************/

    double transferedBytes = 2.0 * sizeC;
    transferedBytes /= 1e9;
    printf("cuTensor: %.2f GB/s\n", transferedBytes / minTimeCUTENSOR);

    if (A) cudaFreeHost(A);
    if (C) cudaFreeHost(C);
    if (A_d) cudaFree(A_d);
    if (C_d) cudaFree(C_d);

    return 0;
}
