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
    cudaDataType_t typeC = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_R_MIN_32F;

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

    void *A_d, *C_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&A_d, sizeA));
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&C_d, sizeC));

    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

    if (A == NULL || C == NULL)
    {
        printf("Error: Host allocation of A, B, or C.\n");
        return -1;
    }

    /*******************
     * Initialize data
     *******************/

    for (int64_t i = 0; i < elementsA; i++)
        A[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for (int64_t i = 0; i < elementsC; i++)
        C[i] = (((float) rand())/RAND_MAX - 0.5)*100;

    HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));

    /*************************
     * cuTENSOR
     *************************/ 

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

    const cutensorOperator_t opReduce = CUTENSOR_OP_ADD;

    /**********************
     * Querry workspace
     **********************/

    uint64_t worksize = 0;
    HANDLE_ERROR(cutensorReductionGetWorkspace(&handle, 
                 A_d, &descA, modeA.data(),
                 C_d, &descC, modeC.data(),
                 C_d, &descC, modeC.data(),
                 opReduce, typeCompute, &worksize));
    void *work = nullptr;
    if (worksize > 0)
    {
        if (cudaSuccess != cudaMalloc(&work, worksize))
        {
            work = nullptr;
            worksize = 0;
        }
    } 

    /**********************
     * Run
     **********************/

    double minTimeCUTENSOR = 1e100;
    cutensorStatus_t err;
    for(int i=0; i < 3; ++i)
    {
        HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

        // Set up timing
        GPUTimer timer;
        timer.start();

        err = cutensorReduction(&handle, 
                (const void*)&alpha, A_d, &descA, modeA.data(),
                (const void*)&beta,  C_d, &descC, modeC.data(), 
                                     C_d, &descC, modeC.data(), 
                opReduce, typeCompute, work, worksize, 0 /* stream */);

        // Synchronize and measure timing
        auto time = timer.seconds();

        if (err != CUTENSOR_STATUS_SUCCESS)
        {
            printf("ERROR: %s in line %d\n", cutensorGetErrorString(err), __LINE__);
        }
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
    }

    /*************************/

    double transferedBytes = sizeC + sizeA;
    transferedBytes += ((float) beta != 0.f) ? sizeC : 0;
    transferedBytes /= 1e9;
    printf("cuTensor: %.2f GB/s\n", transferedBytes / minTimeCUTENSOR);

    if (A) free(A);
    if (C) free(C);
    if (A_d) cudaFree(A_d);
    if (C_d) cudaFree(C_d);
    if (work) cudaFree(work);

    return 0;
}
