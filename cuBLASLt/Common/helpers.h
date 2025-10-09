/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
inline size_t getScaleTensorSize(int inner, int outer, cublasLtMatmulMatrixScale_t scaleMode) {
    if (scaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F) return 1;

    if (scaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 || scaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3) {
        static const size_t S_VSCALE = scaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 ? 32 : 16;
        static const size_t S_BLOCK_COLS = 32;
        static const size_t S_BLOCK_ROWS = 4;
        static const size_t S_BLOCK_INNER = 4;

        static const size_t BLOCK_ROWS = S_BLOCK_INNER * S_VSCALE;
        static const size_t BLOCK_COLS = S_BLOCK_COLS * S_BLOCK_ROWS;

        size_t s_rows = roundoff(size_t(inner), BLOCK_ROWS) / S_VSCALE;
        size_t s_cols = roundoff(size_t(outer), BLOCK_COLS);

        return s_rows * s_cols;
    }

    if (scaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F) {
        return outer;
    }

    if (scaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F) {
        return ceildiv(inner, 128) * outer;
    }

    if (scaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F) {
        return roundoff(ceildiv(inner, 128), 4) * ceildiv(outer, 128);
    }

    return 0;
}

template <typename T>
static bool isNarrowPrecision() {
    return std::is_same<T, __nv_fp8_e4m3>::value || std::is_same<T, __nv_fp8_e5m2>::value || std::is_same<T, __nv_fp4_e2m1>::value;
}

template <typename InTypeAB, typename OutType = InTypeAB, typename ComputeType = OutType, typename ScaleType = ComputeType, typename DScaleType = ScaleType, typename InTypeC = OutType>
struct TestBench {
    using SampleRunner = std::function<void()>;

    TestBench(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
            ComputeType alpha = ComputeType{1.0f}, ComputeType beta = ComputeType{0.0f},
            size_t workspaceSize = 1024 * 1024 * 4, int N = 1,
              bool ptrArrayBatch = false, bool forceOutOfPlace = false) :
        TestBench(transa, transb, m, n, k, alpha, beta, workspaceSize, N, ScaleType{2.0f}, ScaleType{0.5f}, ScaleType{1.0f}, DScaleType{1.0f},
                  CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
                  CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
                  ptrArrayBatch, forceOutOfPlace) {}

    TestBench(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
            ComputeType alpha, ComputeType beta,
            size_t workspaceSize, int N,
            cublasLtMatmulMatrixScale_t AScaleMode,
            cublasLtMatmulMatrixScale_t BScaleMode,
            cublasLtMatmulMatrixScale_t CScaleMode,
            cublasLtMatmulMatrixScale_t DScaleMode,
            cublasLtMatmulMatrixScale_t DOutScaleMode):
        TestBench(transa, transb, m, n, k, alpha, beta, workspaceSize, N, ScaleType{2.0f}, ScaleType{0.5f}, ScaleType{1.0f}, DScaleType{1.0f},
                  AScaleMode, BScaleMode, CScaleMode, DScaleMode, DOutScaleMode, false, false) {}

    TestBench(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
            ComputeType alpha, ComputeType beta, size_t workspaceSize, int N,
            ScaleType Ascale, ScaleType Bscale,
            ScaleType Cscale, DScaleType Dscale,
            bool ptrArrayBatch = false, bool forceOutOfPlace = false) :
        TestBench(transa, transb, m, n, k, alpha, beta, workspaceSize, N, Ascale, Bscale, Cscale, Dscale,
                  CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
                  CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
                  ptrArrayBatch, forceOutOfPlace) {}

    TestBench(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
            ComputeType alpha, ComputeType beta, size_t workspaceSize, int N,
            ScaleType Ascale, ScaleType Bscale,
            ScaleType Cscale, DScaleType Dscale,
            cublasLtMatmulMatrixScale_t AScaleMode,
            cublasLtMatmulMatrixScale_t BScaleMode,
            cublasLtMatmulMatrixScale_t CScaleMode,
            cublasLtMatmulMatrixScale_t DScaleMode,
            cublasLtMatmulMatrixScale_t DOutScaleMode,
            bool ptrArrayBatch = false, bool forceOutOfPlace = false) :
        outOfPlace(forceOutOfPlace || !std::is_same<InTypeC, OutType>::value),
        transa(transa), transb(transb), m(m), n(n), k(k), N(N), lda(transa != CUBLAS_OP_N ? k : m), ldb(transb != CUBLAS_OP_N ? n : k), ldc(m), ldd(m),
        alpha(alpha), beta(beta), workspaceSize(workspaceSize), Ahost(sizeofElements<InTypeAB>(m * k * N)), Bhost(sizeofElements<InTypeAB>(n * k * N)),
        Chost(sizeofElements<InTypeC>(m * n * N)), Dhost(outOfPlace ? sizeofElements<OutType>(m * n * N) : 0), biasHost(sizeofElements<OutType>(m * N)),
        APtrArrayHost(ptrArrayBatch ? N : 0), BPtrArrayHost(ptrArrayBatch ? N : 0), CPtrArrayHost(ptrArrayBatch ? N : 0), DPtrArrayHost(ptrArrayBatch && outOfPlace ? N : 0),
        AScaleMode(AScaleMode), BScaleMode(BScaleMode), CScaleMode(CScaleMode), DScaleMode(DScaleMode), DOutScaleMode(DOutScaleMode), ptrArrayBatch(ptrArrayBatch) {

        // Currently only fp8 supports per-tensor scaling (from second file)
        perTensorScalingEnabled = std::is_same<InTypeAB, __nv_fp8_e4m3>::value || std::is_same<InTypeAB, __nv_fp8_e5m2>::value;

        if (isNarrowPrecision<InTypeAB>()) {
            AscaleHost.resize(getScaleTensorSize(transa != CUBLAS_OP_N ? k : m, transa != CUBLAS_OP_N ? m : k, AScaleMode));
            BscaleHost.resize(getScaleTensorSize(transb != CUBLAS_OP_N ? n : k, transb != CUBLAS_OP_N ? k : n, BScaleMode));
        }

        if (isNarrowPrecision<InTypeC>()) {
            CscaleHost.resize(getScaleTensorSize(m, n, CScaleMode));
        }

        if (isNarrowPrecision<OutType>()) {
            DOutscaleHost.resize(getScaleTensorSize(m, n, DOutScaleMode));
            DscaleHost.resize(getScaleTensorSize(m, n, DScaleMode));
        }

        checkCublasStatus(cublasLtCreate(&ltHandle));

        checkCudaStatus(cudaStreamCreate(&stream));

        if (ptrArrayBatch) {
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&APtrArrayDev), N * sizeof(void*)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&BPtrArrayDev), N * sizeof(void*)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&CPtrArrayDev), N * sizeof(void*)));
            if (outOfPlace)
                checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&DPtrArrayDev), N * sizeof(void*)));
            else
                DPtrArrayDev = reinterpret_cast<decltype(DPtrArrayDev)>(CPtrArrayDev);
            for (int i = 0; i < N; ++i) {
                checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&APtrArrayHost[i]), Ahost.size() * sizeof(Ahost[0]) / N));
                checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&BPtrArrayHost[i]), Bhost.size() * sizeof(Bhost[0]) / N));
                checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&CPtrArrayHost[i]), Chost.size() * sizeof(Chost[0]) / N));
                if (outOfPlace)
                    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&DPtrArrayHost[i]), Dhost.size() * sizeof(Dhost[0]) / N));
            }
        } else {
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Adev), Ahost.size() * sizeof(Ahost[0])));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Bdev), Bhost.size() * sizeof(Bhost[0])));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Cdev), Chost.size() * sizeof(Chost[0])));
            if (outOfPlace)
                checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Ddev), Dhost.size() * sizeof(Dhost[0])));
            else
                Ddev = reinterpret_cast<decltype(Ddev)>(Cdev);
        }
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&biasDev), biasHost.size() * sizeof(biasHost[0])));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&workspace), workspaceSize));

        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&AscaleDev), AscaleHost.size() * sizeof(AscaleHost[0])));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&BscaleDev), BscaleHost.size() * sizeof(BscaleHost[0])));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&CscaleDev), CscaleHost.size() * sizeof(CscaleHost[0])));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&DOutscaleDev), DOutscaleHost.size() * sizeof(DOutscaleHost[0])));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&DscaleDev), DscaleHost.size() * sizeof(DscaleHost[0])));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&DamaxDev), sizeof(DamaxHost)));

        // Additional per-tensor scaling allocation (from second file)
        if (perTensorScalingEnabled && AscaleHost.empty()) {
            // Fallback allocation for simple per-tensor scaling when not using advanced scaling modes
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&AscaleDev), sizeof(ScaleType)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&BscaleDev), sizeof(ScaleType)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&CscaleDev), sizeof(ScaleType)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&DscaleDev), sizeof(DScaleType)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&DamaxDev), sizeof(ComputeType)));
        }

        fillData();
        fillScales(Ascale, Bscale, Cscale, Dscale);
    }

    ~TestBench() {
        checkCublasStatus(cublasLtDestroy(ltHandle));

        if (ptrArrayBatch) {
            for (int i = 0; i < N; ++i) {
                checkCudaStatus(cudaFree(APtrArrayHost[i]));
                checkCudaStatus(cudaFree(BPtrArrayHost[i]));
                checkCudaStatus(cudaFree(CPtrArrayHost[i]));
                if (outOfPlace) checkCudaStatus(cudaFree(DPtrArrayHost[i]));
            }
            checkCudaStatus(cudaFree(APtrArrayDev));
            checkCudaStatus(cudaFree(BPtrArrayDev));
            checkCudaStatus(cudaFree(CPtrArrayDev));
            if (outOfPlace) checkCudaStatus(cudaFree(DPtrArrayDev));
        } else {
            checkCudaStatus(cudaFree(Bdev));
            checkCudaStatus(cudaFree(Adev));
            checkCudaStatus(cudaFree(Cdev));
            if (outOfPlace) checkCudaStatus(cudaFree(Ddev));
            checkCudaStatus(cudaFree(biasDev));
        }
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
        if (ptrArrayBatch) {
            checkCudaStatus(cudaMemcpyAsync(APtrArrayDev, APtrArrayHost.data(), N * sizeof(APtrArrayHost[0]), cudaMemcpyHostToDevice, stream));
            checkCudaStatus(cudaMemcpyAsync(BPtrArrayDev, BPtrArrayHost.data(), N * sizeof(BPtrArrayHost[0]), cudaMemcpyHostToDevice, stream));
            checkCudaStatus(cudaMemcpyAsync(CPtrArrayDev, CPtrArrayHost.data(), N * sizeof(CPtrArrayHost[0]), cudaMemcpyHostToDevice, stream));
            if (outOfPlace) checkCudaStatus(cudaMemcpyAsync(DPtrArrayDev, DPtrArrayHost.data(), N * sizeof(DPtrArrayHost[0]), cudaMemcpyHostToDevice, stream));

            for (int i = 0; i < N; ++i) {
                checkCudaStatus(cudaMemcpyAsync(APtrArrayHost[i], &Ahost[i * m * n], Ahost.size() / N * sizeof(Ahost[0]), cudaMemcpyHostToDevice, stream));
                checkCudaStatus(cudaMemcpyAsync(BPtrArrayHost[i], &Bhost[i * m * n], Bhost.size() / N * sizeof(Bhost[0]), cudaMemcpyHostToDevice, stream));
            }
        } else {
            checkCudaStatus(cudaMemcpyAsync(Adev, Ahost.data(), Ahost.size() * sizeof(Ahost[0]), cudaMemcpyHostToDevice, stream));
            checkCudaStatus(cudaMemcpyAsync(Bdev, Bhost.data(), Bhost.size() * sizeof(Bhost[0]), cudaMemcpyHostToDevice, stream));
        }

        checkCudaStatus(cudaMemcpyAsync(biasDev, biasHost.data(), biasHost.size() * sizeof(biasHost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(AscaleDev, AscaleHost.data(), AscaleHost.size() * sizeof(AscaleHost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(BscaleDev, BscaleHost.data(), BscaleHost.size() * sizeof(BscaleHost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(CscaleDev, CscaleHost.data(), CscaleHost.size() * sizeof(CscaleHost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(DscaleDev, DscaleHost.data(), DscaleHost.size() * sizeof(DscaleHost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(DamaxDev, &DamaxHost, sizeof(DamaxHost), cudaMemcpyHostToDevice, stream));
        
        // Additional per-tensor scaling copy logic (from second file)
        if (perTensorScalingEnabled && AscaleHost.empty()) {
            // For simple per-tensor scaling, copy individual scale values
            ScaleType tempAscale = ScaleType{2.0f}, tempBscale = ScaleType{0.5f}, tempCscale = ScaleType{1.0f};
            DScaleType tempDscale = DScaleType{1.0f};
            ComputeType tempDamax = ComputeType{0.0f};
            checkCudaStatus(cudaMemcpyAsync(AscaleDev, &tempAscale, sizeof(tempAscale), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(BscaleDev, &tempBscale, sizeof(tempBscale), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(CscaleDev, &tempCscale, sizeof(tempCscale), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(DscaleDev, &tempDscale, sizeof(tempDscale), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(DamaxDev, &tempDamax, sizeof(tempDamax), cudaMemcpyHostToDevice));
        }
    }

    void copyDataFromDevice() {
        if (ptrArrayBatch) {
            for (int i = 0; i < N; ++i) {
                if (outOfPlace)
                    checkCudaStatus(cudaMemcpyAsync(&Dhost[i * m * n], DPtrArrayHost[i], Dhost.size() / N * sizeof(Dhost[0]), cudaMemcpyDeviceToHost, stream));
                else
                    checkCudaStatus(cudaMemcpyAsync(&Chost[i * m * n], CPtrArrayHost[i], Chost.size() / N * sizeof(Chost[0]), cudaMemcpyDeviceToHost, stream));
            }
        } else {
            if (outOfPlace)
                checkCudaStatus(cudaMemcpyAsync(Dhost.data(), Ddev, Dhost.size() * sizeof(Dhost[0]), cudaMemcpyDeviceToHost, stream));
            else
                checkCudaStatus(cudaMemcpyAsync(Chost.data(), Cdev, Chost.size() * sizeof(Chost[0]), cudaMemcpyDeviceToHost, stream));
        }

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
    bool ptrArrayBatch;
    bool perTensorScalingEnabled;
    cublasOperation_t transa, transb;
    int m, n, k, N;
    int lda, ldb, ldc, ldd;
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

    std::vector<typename StorageType<InTypeAB>::type *> APtrArrayHost, BPtrArrayHost;
    std::vector<typename StorageType<InTypeC>::type *> CPtrArrayHost;
    std::vector<typename StorageType<OutType>::type *> DPtrArrayHost;

    typename StorageType<InTypeAB>::type **APtrArrayDev, **BPtrArrayDev;
    typename StorageType<InTypeC>::type **CPtrArrayDev;
    typename StorageType<OutType>::type **DPtrArrayDev;

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
