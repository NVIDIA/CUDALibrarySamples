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

#include <assert.h>
#include <chrono>
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>

// Handle cuTENSOR errors
#define HANDLE_ERROR(x) {                                                                \
  const auto err = x;                                                                    \
  if( err != CUTENSOR_STATUS_SUCCESS )                                                   \
  { printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); exit(-1); } \
};

// Handle CUDA errors
#define HANDLE_CUDA_ERROR(x) {                                                       \
  const auto err = x;                                                                \
  if( err != cudaSuccess )                                                           \
  { printf("Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); exit(-1); } \
};

class CPUTimer
{
public:
    void start()
    {
        start_ = std::chrono::steady_clock::now();
    }

    double seconds()
    {
        end_ = std::chrono::steady_clock::now();
        elapsed_ = end_ - start_;
        //return in ms
        return elapsed_.count() * 1000;
    }

private:
    typedef std::chrono::steady_clock::time_point tp;
    tp start_;
    tp end_;
    std::chrono::duration<double> elapsed_;
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

int main()
{
    typedef std::complex<float> TypeA;
    typedef std::complex<float> TypeB;
    typedef std::complex<float> TypeC;
    typedef std::complex<float> TypeScalar;

    auto alpha = TypeScalar(1.1, 0.0);
    auto beta  = TypeScalar(0.0, 0.0);

    cutensorDataType_t typeA = CUTENSOR_C_32F;
    cutensorDataType_t typeB = CUTENSOR_C_32F;
    cutensorDataType_t typeC = CUTENSOR_C_32F;
    cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_3XTF32;

    /**********************
     * Computing: C_{0,1,2,3,4,6,8,9,25,26,10,12,14,27,15,28,17,19,29,20,21,30,23,24} =
     *                   \alpha A_{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}
     *                   B_{25,26,27,28,29,30,5,7,11,13,16,18,22}
     *                   + \beta C_{0,1,2,3,4,6,8,9,25,26,10,12,14,27,15,28,17,19,29,20,21,30,23,24}
     **********************/

    /* ***************************** */

    // Create vector of modes
    std::vector<int> modeC{0,1,2,3,4,6,8,9,25,26,10,12,14,27,15,28,17,19,29,20,21,30,23,24};
    std::vector<int> modeA{0,2,1,4,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,19,21,22,23,24};
    std::vector<int> modeB{25,26,27,28,29,30,5,7,11,13,16,18,22};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    // Extents
    std::unordered_map<int, int64_t> extent;
    for (auto i = 0; i <= 30; i++)
        extent[i] = 2;

    // Create a vector of extents for each tensor
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

    // Number of elements of each tensor
    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsB = 1;
    for (auto mode : modeB)
        elementsB *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode];

    // Size in bytes
    size_t sizeA = sizeof(TypeA) * elementsA;
    size_t sizeB = sizeof(TypeB) * elementsB;
    size_t sizeC = sizeof(TypeC) * elementsC;
    printf("Total memory: %.2f GiB\n", (sizeA + sizeB + sizeC)/1024./1024./1024);

    // Allocate on device
    void *A_d, *B_d, *C_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &A_d, sizeA));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &B_d, sizeB));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &C_d, sizeC));

    TypeA *A = (TypeA*) malloc(sizeof(TypeA) * elementsA);
    TypeB *B = (TypeB*) malloc(sizeof(TypeB) * elementsB);
    TypeC *C = (TypeC*) malloc(sizeof(TypeC) * elementsC);

    if (A == nullptr || B == nullptr || C == nullptr)
    {
        printf("Error: Host allocation of A, B, or C.\n");
        exit(-1);
    }

    /*******************
     * Initialize data
     *******************/

    for (int64_t i = 0; i < elementsA; i++)
        A[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for (int64_t i = 0; i < elementsB; i++)
        B[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for (int64_t i = 0; i < elementsC; i++)
        C[i] = (((float) rand())/RAND_MAX - 0.5)*100;

    // Copy to device
    HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));

    const uint32_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(B_d) % kAlignment == 0);
    assert(uintptr_t(C_d) % kAlignment == 0);

    /*************************
     * cuTENSOR
     *************************/ 

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    // Read kernel cache from file (if the file was generated by a prior execution)
    auto readKernelCacheStatus = cutensorReadKernelCacheFromFile(handle, "kernelCache.bin");

    if (readKernelCacheStatus == CUTENSOR_STATUS_IO_ERROR)
        printf("No kernel cache found. It will be generated before the end of this execution.\n");
    else if (readKernelCacheStatus == CUTENSOR_STATUS_SUCCESS)
        printf("Kernel cache found and read successfully.\n");
    else
        HANDLE_ERROR(readKernelCacheStatus);

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descA,
                                                nmodeA,
                                                extentA.data(),
                                                NULL,/*stride*/
                                                typeA, kAlignment));

    cutensorTensorDescriptor_t descB;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descB,
                                                nmodeB,
                                                extentB.data(),
                                                NULL,/*stride*/
                                                typeB, kAlignment));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descC,
                                                nmodeC,
                                                extentC.data(),
                                                NULL,/*stride*/
                                                typeC, kAlignment));

    /*******************************
     * Create Contraction Descriptor
     *******************************/

    cutensorOperationDescriptor_t desc;
    HANDLE_ERROR(cutensorCreateContraction(handle, 
                                           &desc,
                                           descA, modeA.data(), /* unary operator A*/CUTENSOR_OP_IDENTITY,
                                           descB, modeB.data(), /* unary operator B*/CUTENSOR_OP_IDENTITY,
                                           descC, modeC.data(), /* unary operator C*/CUTENSOR_OP_IDENTITY,
                                           descC, modeC.data(),
                                           descCompute));

    /*****************************
     * Optional (but recommended): ensure that the scalar type is correct.
     *****************************/

    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle,
                                                         desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                         (void*)&scalarType,
                                                         sizeof(scalarType)));

    assert(scalarType == CUTENSOR_C_32F);

    /**************************
    * Set the algorithm to use -- without just-in-time compilation
    ***************************/

    const cutensorAlgo_t algo = CUTENSOR_ALGO_GETT;

    cutensorPlanPreference_t planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(handle,
                                              &planPref,
                                              algo,
                                              CUTENSOR_JIT_MODE_NONE));

    /**********************
     * Query workspace estimate
     **********************/

    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,
                                               desc,
                                               planPref,
                                               workspacePref,
                                               &workspaceSizeEstimate));

    /**************************
     * Create Contraction Plan -- without just-in-time compilation
     **************************/

    cutensorPlan_t plan;
    HANDLE_ERROR(cutensorCreatePlan(handle,
                                    &plan,
                                    desc,
                                    planPref,
                                    workspaceSizeEstimate));

    /**************************
     * Optional: Query information about the created plan
     **************************/

    // query actually used workspace
    uint64_t actualWorkspaceSize = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(handle,
                                              plan,
                                              CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                                              &actualWorkspaceSize,
                                              sizeof(actualWorkspaceSize)));

    // At this point the user knows exactly how much memory is need by the operation and
    // only the smaller actual workspace needs to be allocated
    assert(actualWorkspaceSize <= workspaceSizeEstimate);

    void *work = nullptr;
    if (actualWorkspaceSize > 0)
    {
        HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
        assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
    }

    /**********************
     * Execute the tensor contraction
     **********************/

    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

    double minTimeCUTENSOR = 1e100;
    for (int i=0; i < 3; ++i)
    {
        cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice);

        // Set up timing
        GPUTimer timer;
        timer.start();

        HANDLE_ERROR(cutensorContract(handle,
                                      plan,
                                      (void*) &alpha, A_d, B_d,
                                      (void*) &beta,  C_d, C_d,
                                      work, actualWorkspaceSize, stream))

        // Synchronize and measure timing
        auto time = timer.seconds();

        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
    }

    /*************************/

    /**************************
    * Set the algorithm to use -- with just-in-time compilation
    ***************************/

    cutensorPlanPreference_t planPrefJit;
    HANDLE_ERROR(cutensorCreatePlanPreference(handle,
                                              &planPrefJit,
                                              algo,
                                              CUTENSOR_JIT_MODE_DEFAULT));

    /**********************
     * Query workspace estimate
     **********************/

    uint64_t workspaceSizeEstimateJit = 0;
    const cutensorWorksizePreference_t workspacePrefJit = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,
                                               desc,
                                               planPrefJit,
                                               workspacePrefJit,
                                               &workspaceSizeEstimateJit));

    /**************************
     * Create Contraction Plan -- with just-in-time compilation
     **************************/

    cutensorPlan_t planJit;
    CPUTimer jitPlanTimer;
    jitPlanTimer.start();
    // This is where the kernel is actually compiled
    HANDLE_ERROR(cutensorCreatePlan(handle,
                                    &planJit,
                                    desc,
                                    planPrefJit,
                                    workspaceSizeEstimateJit));
    auto jitPlanTime = jitPlanTimer.seconds();

    /**************************
     * Optional: Query information about the created plan
     **************************/

    // query actually used workspace
    uint64_t actualWorkspaceSizeJit = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(handle,
                                              planJit,
                                              CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                                              &actualWorkspaceSizeJit,
                                              sizeof(actualWorkspaceSizeJit)));

    // At this point the user knows exactly how much memory is need by the operation and
    // only the smaller actual workspace needs to be allocated
    assert(actualWorkspaceSizeJit <= workspaceSizeEstimateJit);

    void *workJit = nullptr;
    if (actualWorkspaceSizeJit > 0)
    {
        HANDLE_CUDA_ERROR(cudaMalloc(&workJit, actualWorkspaceSizeJit));
        assert(uintptr_t(workJit) % 128 == 0); // workspace must be aligned to 128 byte-boundary
    }

    /**********************
     * Execute the tensor contraction using the JIT compiled kernel
     **********************/

    double minTimeCUTENSORJit = 1e100;
    for (int i=0; i < 3; ++i)
    {
        cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice);

        // Set up timing
        GPUTimer timer;
        timer.start();

        HANDLE_ERROR(cutensorContract(handle,
                                      planJit,
                                      (void*) &alpha, A_d, B_d,
                                      (void*) &beta,  C_d, C_d,
                                      workJit, actualWorkspaceSizeJit, stream))

        // Synchronize and measure timing
        auto time = timer.seconds();

        minTimeCUTENSORJit = (minTimeCUTENSORJit < time) ? minTimeCUTENSORJit : time;
    }

    /*************************/

    float flops;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle,
                                                         desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_FLOPS,
                                                         (void*)&flops,
                                                         sizeof(flops)));
    auto gflops = flops / 1e9;
    auto gflopsPerSec = gflops / minTimeCUTENSOR;
    auto gflopsPerSecJit = gflops / minTimeCUTENSORJit;

    printf("cuTENSOR    : %6.0f GFLOPs/s\n", gflopsPerSec);
    printf("cuTENSOR JIT: %6.0f GFLOPs/s\n", gflopsPerSecJit);
    printf("Speedup: %.1fx\n", gflopsPerSecJit / gflopsPerSec);
    printf("JIT Compilation time: %.1f seconds ", jitPlanTime / 1e3);
    if (readKernelCacheStatus == CUTENSOR_STATUS_SUCCESS)
        printf("(Kernel cache file was read successfully; Compilation was not required)\n");
    else
        printf("\n");

    // Write kernel cache to file
    HANDLE_ERROR(cutensorWriteKernelCacheToFile(handle, "kernelCache.bin"))
    printf("Kernel cache written to file. Will be read in next execution.\n");

    HANDLE_ERROR(cutensorDestroy(handle));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
    HANDLE_ERROR(cutensorDestroyPlanPreference(planPref));
    HANDLE_ERROR(cutensorDestroyPlan(plan));
    HANDLE_ERROR(cutensorDestroyPlanPreference(planPrefJit));
    HANDLE_ERROR(cutensorDestroyPlan(planJit));

    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (work) cudaFree(work);
    if (workJit) cudaFree(workJit);

    printf("Successful completion\n");
    return 0;
}
