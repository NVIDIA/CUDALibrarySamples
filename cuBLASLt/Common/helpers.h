/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cstdio>
#include <stdexcept>
#include <vector>
#include <functional>

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime_api.h>


static size_t roundoff(size_t  x, size_t granul) {
  return granul * ((x + (granul - 1)) / granul);
}

template <typename T>
static T ceildiv(T x, size_t divisor) {
  return (x + (divisor - 1)) / divisor;
}

template <typename T>
struct StorageType {
    static constexpr size_t packing = 1;
    using type = T;
};

template <>
struct StorageType<__nv_fp4_e2m1> {
    static constexpr size_t packing = 2;
    using type = __nv_fp4x2_e2m1;
};

template <typename T>
constexpr size_t sizeofElements(size_t N) {
  return ceildiv(N, StorageType<T>::packing);
}

template <typename T>
constexpr size_t sizeofBytes(size_t N) {
  return sizeofElements<T>(N) * sizeof(StorageType<T>::type);
}

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

// Block scales used for mxfp8 and nvfp8 require a special layout: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout for more details.
inline size_t getScaleTensorSize(int rows, int cols, cublasLtMatmulMatrixScale_t ScaleMode) {
    if (ScaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F) return 1;

    if (ScaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 || ScaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3) {
        static const size_t S_VSCALE = ScaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 ? 32 : 16;
        static const size_t S_BLOCK_COLS = 32;
        static const size_t S_BLOCK_ROWS = 4;
        static const size_t S_BLOCK_INNER = 4;

        static const size_t BLOCK_ROWS = S_BLOCK_INNER * S_VSCALE;
        static const size_t BLOCK_COLS = S_BLOCK_COLS * S_BLOCK_ROWS;

        size_t s_rows = roundoff(size_t(rows), BLOCK_ROWS) / S_VSCALE;
        size_t s_cols = roundoff(size_t(cols), BLOCK_COLS);

        return s_rows * s_cols;
    }

    return 0;
}

template <typename InTypeAB, typename OutType = InTypeAB, typename ComputeType = OutType, typename ScaleType = ComputeType, typename DScaleType = ScaleType, typename InTypeC = OutType>
struct TestBench {
    using SampleRunner = std::function<void()>;

    TestBench(int m, int n, int k,
            ComputeType alpha = ComputeType{0.0f}, ComputeType beta = ComputeType{0.0f},
            size_t workspaceSize = 1024 * 1024 * 4, int N = 1,
            ScaleType Ascale = ScaleType{2.0f}, ScaleType Bscale = ScaleType{0.5f},
            ScaleType Cscale = ScaleType{1.0f}, DScaleType Dscale = DScaleType{1.0f},
            cublasLtMatmulMatrixScale_t AScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
            cublasLtMatmulMatrixScale_t BScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
            cublasLtMatmulMatrixScale_t CScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
            cublasLtMatmulMatrixScale_t DScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
            cublasLtMatmulMatrixScale_t DOutScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
            bool forceOutOfPlace = false) :
        outOfPlace(forceOutOfPlace || !std::is_same<InTypeC, OutType>::value),
        m(m), n(n), k(k), N(N), alpha(alpha), beta(beta), workspaceSize(workspaceSize), Ahost(sizeofElements<InTypeAB>(m * k * N)), Bhost(sizeofElements<InTypeAB>(n * k * N)),
        Chost(sizeofElements<InTypeC>(m * n * N)), Dhost(outOfPlace ? sizeofElements<OutType>(m * n * N) : 0), biasHost(sizeofElements<OutType>(m * N)),
        AScaleMode(AScaleMode), BScaleMode(BScaleMode), CScaleMode(CScaleMode), DScaleMode(DScaleMode), DOutScaleMode(DOutScaleMode),
        AscaleHost(getScaleTensorSize(m, k, AScaleMode)), BscaleHost(getScaleTensorSize(k, n, BScaleMode)), CscaleHost(getScaleTensorSize(m, n, CScaleMode)), DOutscaleHost(getScaleTensorSize(m, n, DOutScaleMode)),
        DscaleHost(getScaleTensorSize(m, n, DScaleMode)) {

        checkCublasStatus(cublasLtCreate(&ltHandle));

        checkCudaStatus(cudaStreamCreate(&stream));

        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Adev), Ahost.size() * sizeof(Ahost[0])));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Bdev), Bhost.size() * sizeof(Bhost[0])));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Cdev), Chost.size() * sizeof(Chost[0])));
        if (outOfPlace)
            checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Ddev), Dhost.size() * sizeof(Dhost[0])));
        else
            Ddev = reinterpret_cast<decltype(Ddev)>(Cdev);
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&biasDev), biasHost.size() * sizeof(biasHost[0])));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&workspace), workspaceSize));

        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&AscaleDev), AscaleHost.size() * sizeof(AscaleHost[0])));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&BscaleDev), BscaleHost.size() * sizeof(BscaleHost[0])));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&CscaleDev), CscaleHost.size() * sizeof(CscaleHost[0])));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&DOutscaleDev), DOutscaleHost.size() * sizeof(DOutscaleHost[0])));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&DscaleDev), DscaleHost.size() * sizeof(DscaleHost[0])));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&DamaxDev), sizeof(DamaxHost)));

        fillData();
        fillScales(Ascale, Bscale, Cscale, Dscale);
    }

    ~TestBench() {
        checkCublasStatus(cublasLtDestroy(ltHandle));
        checkCudaStatus(cudaFree(Adev));
        checkCudaStatus(cudaFree(Bdev));
        checkCudaStatus(cudaFree(Cdev));
        if (outOfPlace) checkCudaStatus(cudaFree(Ddev));
        checkCudaStatus(cudaFree(biasDev));
        checkCudaStatus(cudaFree(workspace));
        checkCudaStatus(cudaFree(AscaleDev));
        checkCudaStatus(cudaFree(BscaleDev));
        checkCudaStatus(cudaFree(CscaleDev));
        checkCudaStatus(cudaFree(DscaleDev));
        checkCudaStatus(cudaFree(DOutscaleDev));
        checkCudaStatus(cudaFree(DamaxDev));
        checkCudaStatus(cudaStreamDestroy(stream));
    }

    void fillData() {
        for (size_t i = 0; i < Ahost.size(); i++) Ahost[i] = InTypeAB(i);
        for (size_t i = 0; i < Bhost.size(); i++) Bhost[i] = InTypeAB(i);
        for (size_t i = 0; i < Chost.size(); i++) Chost[i] = InTypeC(i);
        for (size_t i = 0; i < biasHost.size(); i++) biasHost[i] = OutType(i + 1);
    }

    void fillScales(ScaleType Ascale, ScaleType Bscale, ScaleType Cscale, DScaleType Dscale) {
        for (size_t i = 0; i < AscaleHost.size(); i++) AscaleHost[i] = Ascale;
        for (size_t i = 0; i < BscaleHost.size(); i++) BscaleHost[i] = Bscale;
        for (size_t i = 0; i < CscaleHost.size(); i++) CscaleHost[i] = Cscale;
        for (size_t i = 0; i < DscaleHost.size(); i++) DscaleHost[i] = Dscale;
    }

    void copyDataToDevice() {
        checkCudaStatus(cudaMemcpyAsync(Adev, Ahost.data(), Ahost.size() * sizeof(Ahost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(Bdev, Bhost.data(), Bhost.size() * sizeof(Bhost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(biasDev, biasHost.data(), biasHost.size() * sizeof(biasHost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(AscaleDev, AscaleHost.data(), AscaleHost.size() * sizeof(AscaleHost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(BscaleDev, BscaleHost.data(), BscaleHost.size() * sizeof(BscaleHost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(CscaleDev, CscaleHost.data(), CscaleHost.size() * sizeof(CscaleHost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(DscaleDev, DscaleHost.data(), DscaleHost.size() * sizeof(DscaleHost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(DamaxDev, &DamaxHost, sizeof(DamaxHost), cudaMemcpyHostToDevice, stream));
    }

    void copyDataFromDevice() {
        if (outOfPlace)
            checkCudaStatus(cudaMemcpyAsync(Dhost.data(), Ddev, Dhost.size() * sizeof(Dhost[0]), cudaMemcpyDeviceToHost, stream));
        else
            checkCudaStatus(cudaMemcpyAsync(Chost.data(), Cdev, Chost.size() * sizeof(Chost[0]), cudaMemcpyDeviceToHost, stream));
        checkCudaStatus(cudaMemcpyAsync(DOutscaleHost.data(), DOutscaleDev, DOutscaleHost.size() * sizeof(DOutscaleHost[0]), cudaMemcpyDeviceToHost, stream));
    }

    void streamSynchronize() {
        checkCudaStatus(cudaStreamSynchronize(stream));
    }

    void run(const SampleRunner& runSample) {
        copyDataToDevice();

        runSample();

        copyDataFromDevice();
        streamSynchronize();
    }

    bool outOfPlace;
    int m, n, k, N;
    ComputeType alpha, beta;

    cudaStream_t stream;
    cublasLtHandle_t ltHandle;

    void *workspace;
    size_t workspaceSize;

    std::vector<typename StorageType<InTypeAB>::type> Ahost, Bhost;
    std::vector<typename StorageType<InTypeC>::type> Chost;
    std::vector<typename StorageType<OutType>::type> Dhost, biasHost;

    typename StorageType<InTypeAB>::type *Adev, *Bdev;
    typename StorageType<InTypeC>::type *Cdev;
    typename StorageType<OutType>::type *Ddev, *biasDev;

    cublasLtMatmulMatrixScale_t AScaleMode, BScaleMode, CScaleMode, DScaleMode, DOutScaleMode;

    std::vector<ScaleType> AscaleHost, BscaleHost, CscaleHost, DOutscaleHost;
    std::vector<DScaleType> DscaleHost;

    ScaleType *AscaleDev, *BscaleDev, *CscaleDev, *DOutscaleDev;
    DScaleType *DscaleDev;

    ComputeType DamaxHost;
    ComputeType *DamaxDev;
};

template <>
inline void TestBench<__half, __half, float>::fillData() {
    for (size_t i = 0; i < Ahost.size(); i++) Ahost[i] = __float2half_rn(float(i));
    for (size_t i = 0; i < Bhost.size(); i++) Bhost[i] = __float2half_rn(float(i));
    for (size_t i = 0; i < Chost.size(); i++) Chost[i] = __float2half_rn(float(i));
    for (size_t i = 0; i < biasHost.size(); i++) biasHost[i] = __float2half_rn(float(i + 1));
}

template <>
inline void TestBench<__half, __half, cuComplex>::fillData() {
    for (size_t i = 0; i < Ahost.size(); i++) Ahost[i] = __float2half_rn(i/100.f);
    for (size_t i = 0; i < Bhost.size(); i++) Bhost[i] = __float2half_rn(i/100.f);
    for (size_t i = 0; i < Chost.size(); i++) Chost[i] = __float2half_rn(i/100.f);
    for (size_t i = 0; i < biasHost.size(); i++) biasHost[i] = __float2half_rn(float(i + 1));
}

template <>
inline void TestBench<__nv_fp4_e2m1, __nv_fp4_e2m1, float, __nv_fp8_e4m3, float, __nv_bfloat16>::fillData() {
    for (size_t i = 0; i < Ahost.size(); i++) Ahost[i] = __nv_fp4x2_e2m1{float2{float(i % 5), float(i % 5) + 1}};
    for (size_t i = 0; i < Bhost.size(); i++) Bhost[i] = __nv_fp4x2_e2m1{float2{float(i % 5), float(i % 5) + 1}};
    for (size_t i = 0; i < Chost.size(); i++) Chost[i] = __nv_bfloat16(i % 5);
    for (size_t i = 0; i < biasHost.size(); i++) biasHost[i] = __nv_fp4x2_e2m1{float2{float(i % 5), float(i % 5) + 1}};
}

const char *tileToString(cublasLtMatmulTile_t tile);
