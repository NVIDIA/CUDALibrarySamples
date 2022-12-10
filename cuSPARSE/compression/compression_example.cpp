/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <algorithm>          // std::fill
#include <cstdio>             // printf
#include <cstdlib>            // EXIT_FAILURE
#include <cuda.h>             // cuMemCreate
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseAxpby
#include <numeric>            // std::iota

#define CHECK_DRV(func)                                                        \
{                                                                              \
    CUresult status = (func);                                                  \
    if (status != CUDA_SUCCESS) {                                              \
        const char* error_str;                                                 \
        cuGetErrorString(status, &error_str);                                  \
        std::printf("cuMem API failed at line %d with error: %s (%d)\n",       \
                    __LINE__, error_str, status);                              \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::printf("CUDA API failed at line %d with error: %s (%d)\n",        \
                    __LINE__, cudaGetErrorString(status), status);             \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n",    \
                    __LINE__, cusparseGetErrorString(status), status);         \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
}

size_t round_up(size_t value, size_t div) {
    return ((value + div - 1) / div) * div;
}

//------------------------------------------------------------------------------

template<typename T>
class drvMemory {
public:
    explicit drvMemory(int64_t num_items, const T* h_values) {
        // DRIVER CONTEXT
        auto     size_bytes = num_items * sizeof(T);
        CUdevice dev        = 0;
        // (1) Set allocation properties, enable compression
        CUmemAllocationProp prop = {};
        prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
        prop.type                       = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type              = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id                = static_cast<int>(dev);
        prop.win32HandleMetaData        = 0;

        // (2) Retrieve allocation granularity
        size_t granularity = 0;
        CHECK_DRV( cuMemGetAllocationGranularity(&granularity, &prop,
                                            CU_MEM_ALLOC_GRANULARITY_MINIMUM) )
        _padded_size = round_up(size_bytes, granularity);

        // (3) Reverse Address range
        CUdeviceptr d_ptr_raw = 0ULL;
        CHECK_DRV( cuMemAddressReserve(&d_ptr_raw, _padded_size, 0, 0, 0) )

        // (4) Create the Allocation Handle
        CHECK_DRV( cuMemCreate(&_allocation_handle, _padded_size, &prop, 0) )

        // (5) Memory mappping
        CHECK_DRV( cuMemMap(d_ptr_raw, _padded_size, 0, _allocation_handle, 0) )

        // (6) Verify that the allocation properties are supported
        CHECK_DRV( cuMemGetAllocationPropertiesFromHandle(&prop,
                                                          _allocation_handle) )

        // (7) Set Access Permissions
        CUmemAccessDesc accessDesc = {};
        accessDesc.location        = prop.location;
        accessDesc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CHECK_DRV( cuMemSetAccess(d_ptr_raw, _padded_size, &accessDesc, 1) )

        // (7) Convert the raw pointer
        _d_ptr = reinterpret_cast<T*>(d_ptr_raw);
        CHECK_CUDA( cudaMemcpy(_d_ptr, h_values, num_items * sizeof(T),
                               cudaMemcpyHostToDevice) )
    }

    T* ptr() const { return static_cast<T*>(_d_ptr); }

    ~drvMemory() {
        auto ptr = reinterpret_cast<CUdeviceptr>(_d_ptr);
        CHECK_DRV( cuMemUnmap(ptr, _padded_size)       )
        CHECK_DRV( cuMemAddressFree(ptr, _padded_size) )
        CHECK_DRV( cuMemRelease(_allocation_handle)    )
    }

private:
    CUmemGenericAllocationHandle _allocation_handle {};
    size_t                       _padded_size       { 0 };
    void*                        _d_ptr             { nullptr };
};

//------------------------------------------------------------------------------

template<typename T>
class cudaMemory {
public:
    explicit cudaMemory(int64_t num_items, const T* h_values) {
        CHECK_CUDA( cudaMalloc(reinterpret_cast<void**>(&_d_ptr),
                               num_items * sizeof(T)) )
        CHECK_CUDA( cudaMemcpy(_d_ptr, h_values, num_items * sizeof(T),
                               cudaMemcpyHostToDevice) )
    }

    T* ptr() const { return _d_ptr; }

    ~cudaMemory() { CHECK_CUDA( cudaFree(_d_ptr) ) }

private:
    T* _d_ptr { nullptr };
};

//------------------------------------------------------------------------------

float benchmark(int64_t nnz,
                int64_t size,
                void*   dX_indices,
                void*   dX_values,
                void*   dY) {
    float alpha = 1.0f;
    float beta  = 1.0f;
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse vector X
    CHECK_CUSPARSE( cusparseCreateSpVec(&vecX, size, nnz, dX_indices, dX_values,
                                        CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, size, dY, CUDA_R_32F) )

    // Warmup run
    CHECK_CUSPARSE( cusparseAxpby(handle, &alpha, vecX, &beta, vecY) )
    CHECK_CUDA( cudaDeviceSynchronize() )
    // Timer setup
    float       elapsed_ms = 0;
    cudaEvent_t start_event{}, stop_event{};
    CHECK_CUDA( cudaEventCreate(&start_event) )
    CHECK_CUDA( cudaEventCreate(&stop_event) )
    CHECK_CUDA( cudaEventRecord(start_event, nullptr) )

    // Computation
    for (int i = 0; i < 10; i++)
        cusparseAxpby(handle, &alpha, vecX, &beta, vecY);

    CHECK_CUDA( cudaEventRecord(stop_event, nullptr) )
    CHECK_CUDA( cudaEventSynchronize(stop_event) )
    CHECK_CUDA( cudaEventElapsedTime(&elapsed_ms, start_event, stop_event) )
    CHECK_CUDA( cudaEventDestroy(start_event) )
    CHECK_CUDA( cudaEventDestroy(stop_event) )

    // Destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    return elapsed_ms;
}

//------------------------------------------------------------------------------
 constexpr int EXIT_UNSUPPORTED = 2;

int main() {
    cudaFree(nullptr);
    // Without a previous CUDA runtime call (e.g. cudaMalloc) we need to
    // explicitly initialize the context:
    //
    // CUcontext ctx;
    // CHECK_DRV( cuInit(0) )
    // CHECK_DRV( cuDevicePrimaryCtxRetain(&ctx, 0) )
    // CHECK_DRV( cuCtxSetCurrent(ctx) )
    // CHECK_DRV( cuCtxGetDevice(&dev) )

    // Check if Memory Compression and Virtual Address Management is supported
    CUdevice dev                 = 0;
    int      supportsCompression = 0;
    int      supportsVMM         = 0;
    CHECK_DRV( cuDeviceGetAttribute(
                    &supportsCompression,
                    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, dev) )
    if (supportsCompression == 0) {
        std::printf("\nL2 compression is not supported on the "
                    "current device\n\n");
        return EXIT_UNSUPPORTED;
    }
    CHECK_DRV( cuDeviceGetAttribute(
                    &supportsVMM,
                    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                    dev) )
    if (supportsVMM == 0) {
        std::printf("\nVirtual Memory Management is not supported on the "
                    "current device\n\n");
        return EXIT_UNSUPPORTED;
    }
    //--------------------------------------------------------------------------
    // Host problem definition
    int64_t size       = 134217728; // 2^27
    int64_t nnz        = 134217728; // 2^27
    auto    hX_indices = new int[nnz];
    auto    hX_values  = new float[nnz];
    auto    hY         = new float[size];

    std::iota(hX_indices, hX_indices + nnz, 0);
    std::fill(hX_values,  hX_values + nnz,  1.0f);
    std::fill(hY,         hY + size,        1.0f);
    //--------------------------------------------------------------------------
    // Device memory management
    float cuda_elapsed_ms = 0.0f;
    float drv_elapsed_ms  = 0.0f;
    {
        cudaMemory<int>   dX_indices_cuda{nnz, hX_indices};
        cudaMemory<float> dX_values_cuda{nnz,  hX_values};
        cudaMemory<float> dY_cuda{size, hY};
        cuda_elapsed_ms = benchmark(nnz, size, dX_indices_cuda.ptr(),
                                    dX_values_cuda.ptr(), dY_cuda.ptr());
    }
    {
        drvMemory<int>   dX_indices_drv{nnz, hX_indices};
        drvMemory<float> dX_values_drv{nnz,  hX_values};
        drvMemory<float> dY_drv{size, hY};
        drv_elapsed_ms = benchmark(nnz, size, dX_indices_drv.ptr(),
                                   dX_values_drv.ptr(), dY_drv.ptr());
    }
    delete[] hX_indices;
    delete[] hX_values;
    delete[] hY;
    auto speedup = ((cuda_elapsed_ms - drv_elapsed_ms) / cuda_elapsed_ms)
                    * 100.0f;
    std::printf("\nStandard call:    %.1f ms"
                "\nL2 Compression:   %.1f ms"
                "\nPerf improvement: %.1f%%\n\n",
                cuda_elapsed_ms, drv_elapsed_ms, speedup);
    return EXIT_SUCCESS;
}
