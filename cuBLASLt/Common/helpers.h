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


static unsigned int roundoff(unsigned int  x, unsigned int granul) {
  return granul * ((x + (granul - 1)) / granul);
}

template <typename T>
static T ceildiv(T x, unsigned int divisor) {
  return (x + (divisor - 1)) / divisor;
}

template <typename T>
constexpr size_t sizeofBits() {
  return sizeof(T) * 8;
}

template <>
constexpr size_t sizeofBits<__nv_fp4_e2m1>() {
  return 4;
}

template <typename T>
constexpr size_t sizeofElements(size_t N) {
  return ceildiv(N * sizeofBits<T>(), 8 * sizeof(T));
}

template <typename T>
constexpr size_t sizeofBytes(size_t N) {
  return sizeofElements<T>(N) * sizeof(T);
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
inline int getScaleTensorSize(int rows, int cols, cublasLtMatmulMatrixScale_t ScaleMode) {
    if (ScaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F) return 1;

    if (ScaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 || ScaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3) {
        static const size_t S_VSCALE = ScaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 ? 32 : 16;
        static const size_t S_BLOCK_COLS = 32;
        static const size_t S_BLOCK_ROWS = 4;
        static const size_t S_BLOCK_INNER = 4;

        static const size_t BLOCK_ROWS = S_BLOCK_INNER * S_VSCALE;
        static const size_t BLOCK_COLS = S_BLOCK_COLS * S_BLOCK_ROWS;

        int s_rows = roundoff(rows, BLOCK_ROWS) / S_VSCALE;
        int s_cols = roundoff(cols, BLOCK_COLS);

        return s_rows * s_cols;
    }

    return 0;
}

template <typename InType, typename OutType = InType, typename ComputeType = OutType, typename ScaleType = ComputeType>
struct TestBench {
    using SampleRunner = std::function<void()>;

    TestBench(int m, int n, int k,
            ComputeType alpha = ComputeType{0.0f}, ComputeType beta = ComputeType{0.0f},
            size_t workspaceSize = 1024 * 1024 * 4, int N = 1,
            ScaleType Ascale = ScaleType{2.0f}, ScaleType Bscale = ScaleType{0.5f},
            ScaleType Cscale = ScaleType{1.0f}, ScaleType Dscale = ScaleType{1.0f},
            cublasLtMatmulMatrixScale_t AScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
            cublasLtMatmulMatrixScale_t BScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
            cublasLtMatmulMatrixScale_t CScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
            cublasLtMatmulMatrixScale_t DScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
            cublasLtMatmulMatrixScale_t DOutScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F) :
        m(m), n(n), k(k), N(N), alpha(alpha), beta(beta), workspaceSize(workspaceSize), Ahost(sizeofElements<InType>(m * k * N)), Bhost(sizeofElements<InType>(n * k * N)),
        Chost(sizeofElements<OutType>(m * n * N)), biasHost(sizeofElements<OutType>(m * N)), AscaleHost(Ascale), BscaleHost(Bscale), CscaleHost(Cscale), DscaleHost(Dscale),
        AScaleMode(AScaleMode), BScaleMode(BScaleMode), CScaleMode(CScaleMode), DScaleMode(DScaleMode), DOutScaleMode(DOutScaleMode) {

        checkCublasStatus(cublasLtCreate(&ltHandle));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Adev), sizeofBytes<InType>(m * k * N)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Bdev), sizeofBytes<InType>(n * k * N)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Cdev), sizeofBytes<OutType>(m * n * N)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&biasDev), sizeofBytes<OutType>(m * N)));
        checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
        checkCudaStatus(cudaStreamCreate(&stream));

        // Currently only fp4 and fp8 support per-tensor scaling
        perTensorScalingEnabled = std::is_same<InType, __nv_fp8_e4m3>::value ||
                                  std::is_same<InType, __nv_fp8_e5m2>::value || std::is_same<InType, __nv_fp4_e2m1>::value;

        AScaleNum = getScaleTensorSize(m, k, AScaleMode);
        BScaleNum = getScaleTensorSize(k, n, BScaleMode);
        CScaleNum = getScaleTensorSize(m, n, CScaleMode);
        DScaleNum = getScaleTensorSize(m, n, DScaleMode);
        DOutScaleNum = getScaleTensorSize(m, n, DOutScaleMode);

        if (perTensorScalingEnabled) {
            if (AScaleNum > 1) AscaleHostBuffer = static_cast<ScaleType *>(malloc(AScaleNum * sizeof(ScaleType)));
            if (BScaleNum > 1) BscaleHostBuffer = static_cast<ScaleType *>(malloc(BScaleNum * sizeof(ScaleType)));
            if (CScaleNum > 1) CscaleHostBuffer = static_cast<ScaleType *>(malloc(CScaleNum * sizeof(ScaleType)));
            if (DScaleNum > 1) DscaleHostBuffer = static_cast<ScaleType *>(malloc(DScaleNum * sizeof(ScaleType)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&AscaleDev), AScaleNum * sizeof(*AscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&BscaleDev), BScaleNum * sizeof(*BscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&CscaleDev), CScaleNum * sizeof(*CscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&DscaleDev), DScaleNum * sizeof(*DscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&DOutscaleDev), DOutScaleNum * sizeof(*DOutscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&DamaxDev), sizeof(*DamaxDev)));
        }

        fillData();
        fillScales();
    }

    ~TestBench() {
        checkCublasStatus(cublasLtDestroy(ltHandle));
        checkCudaStatus(cudaFree(Adev));
        checkCudaStatus(cudaFree(Bdev));
        checkCudaStatus(cudaFree(Cdev));
        checkCudaStatus(cudaFree(biasDev));
        checkCudaStatus(cudaFree(workspace));
        if (perTensorScalingEnabled) {
            checkCudaStatus(cudaFree(AscaleDev));
            checkCudaStatus(cudaFree(BscaleDev));
            checkCudaStatus(cudaFree(CscaleDev));
            checkCudaStatus(cudaFree(DscaleDev));
            checkCudaStatus(cudaFree(DOutscaleDev));
            checkCudaStatus(cudaFree(DamaxDev));

            if (AScaleNum > 1) free(AscaleHostBuffer);
            if (BScaleNum > 1) free(BscaleHostBuffer);
            if (CScaleNum > 1) free(CscaleHostBuffer);
            if (DScaleNum > 1) free(DscaleHostBuffer);
        }
        checkCudaStatus(cudaStreamDestroy(stream));
    }

    void fillData() {
        for (int i = 0; i < m * k * N; i++) Ahost[i] = InType(i);
        for (int i = 0; i < n * k * N; i++) Bhost[i] = InType(i);
        for (int i = 0; i < m * N; i++) biasHost[i] = OutType(i + 1);
    }

    void fillScales() {
        if (AScaleNum > 1) for (int i = 0; i < AScaleNum; i++) AscaleHostBuffer[i] = AscaleHost;
        if (BScaleNum > 1) for (int i = 0; i < BScaleNum; i++) BscaleHostBuffer[i] = BscaleHost;
        if (CScaleNum > 1) for (int i = 0; i < CScaleNum; i++) CscaleHostBuffer[i] = CscaleHost;
        if (DScaleNum > 1) for (int i = 0; i < DScaleNum; i++) DscaleHostBuffer[i] = DscaleHost;
    }

    void copyDataToDevice() {
        checkCudaStatus(cudaMemcpyAsync(Adev, Ahost.data(), Ahost.size() * sizeof(Ahost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(Bdev, Bhost.data(), Bhost.size() * sizeof(Bhost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(biasDev, biasHost.data(), biasHost.size() * sizeof(biasHost[0]), cudaMemcpyHostToDevice));
        if (perTensorScalingEnabled) {
            checkCudaStatus(cudaMemcpyAsync(AscaleDev, AScaleNum > 1 ? AscaleHostBuffer : &AscaleHost, AScaleNum * sizeof(AscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(BscaleDev, BScaleNum > 1 ? BscaleHostBuffer : &BscaleHost, BScaleNum * sizeof(BscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(CscaleDev, CScaleNum > 1 ? CscaleHostBuffer : &CscaleHost, CScaleNum * sizeof(CscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(DscaleDev, DScaleNum > 1 ? DscaleHostBuffer : &DscaleHost, DScaleNum * sizeof(DscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(DamaxDev, &DamaxHost, sizeof(DamaxHost), cudaMemcpyHostToDevice));
        }
    }

    void copyDataFromDevice() {
        checkCudaStatus(cudaMemcpyAsync(Chost.data(), Cdev, Chost.size() * sizeof(Chost[0]), cudaMemcpyDeviceToHost, stream));
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

    bool perTensorScalingEnabled;
    int m, n, k, N;
    ComputeType alpha, beta;
    size_t workspaceSize;
    std::vector<InType> Ahost, Bhost;
    std::vector<OutType> Chost, biasHost;
    void *workspace;
    InType *Adev, *Bdev;
    OutType *Cdev, *biasDev;
    cudaStream_t stream;
    cublasLtHandle_t ltHandle;
    ScaleType AscaleHost, BscaleHost, CscaleHost, DscaleHost;
    ScaleType *AscaleHostBuffer, *BscaleHostBuffer, *CscaleHostBuffer, *DscaleHostBuffer;
    int AScaleNum, BScaleNum, CScaleNum, DScaleNum, DOutScaleNum;
    ComputeType DamaxHost;
    ScaleType *AscaleDev, *BscaleDev, *CscaleDev, *DscaleDev, *DOutscaleDev;
    ComputeType *DamaxDev;
    cublasLtMatmulMatrixScale_t AScaleMode, BScaleMode, CScaleMode, DScaleMode, DOutScaleMode;
};

template <>
inline void TestBench<__half, __half, float>::fillData() {
    for (int i = 0; i < m * k * N; i++) Ahost[i] = __float2half_rn(i);
    for (int i = 0; i < n * k * N; i++) Bhost[i] = __float2half_rn(i);
    for (int i = 0; i < m * N; i++) biasHost[i] = __float2half_rn(i + 1);
}

template <>
inline void TestBench<__half, __half, cuComplex>::fillData() {
    for (int i = 0; i < m * k * N; i++) Ahost[i] = __float2half_rn(i/100.);
    for (int i = 0; i < n * k * N; i++) Bhost[i] = __float2half_rn(i/100.);
    for (int i = 0; i < m * N; i++) biasHost[i] = __float2half_rn(i + 1);
}

template <>
inline void TestBench<__nv_fp4_e2m1, __nv_fp4_e2m1, float, __nv_fp8_e4m3>::fillData() {
    for (int i = 0; i < sizeofElements<__nv_fp4_e2m1>(m * k * N); i++) Ahost[i].__x = __nv_cvt_float2_to_fp4x2(float2{float(i % 5), float(i % 5) + 1}, __NV_E2M1, cudaRoundNearest);
    for (int i = 0; i < sizeofElements<__nv_fp4_e2m1>(n * k * N); i++) Bhost[i].__x = __nv_cvt_float2_to_fp4x2(float2{float(i % 5), float(i % 5) + 1}, __NV_E2M1, cudaRoundNearest);
    for (int i = 0; i < sizeofElements<__nv_fp4_e2m1>(m * N); i++) biasHost[i].__x =__nv_cvt_float2_to_fp4x2(float2{float(i % 5), float(i % 5) + 1}, __NV_E2M1, cudaRoundNearest);
}