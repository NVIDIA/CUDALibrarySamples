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

#include <complex>
#include <iostream>
#include <iomanip>

#include "gemms.hxx"
#include "cublas_gemm.hxx"

// Debug print macro.
#ifdef DEBUG_PRINTS
    #define DEBUG_PRINT(x) std::cout << x
#else
    #define DEBUG_PRINT(x)
#endif


// Standard BLAS GEMM.
extern "C" {
    void dgemm_(const char *transa, const char *transb,
                const int *m, const int *n, const int *k,
                const double *alpha, const double *a, const int *lda,
                const double *b, const int *ldb,
                const double *beta, double *c, const int *ldc);

    void zgemm_(const char *transa, const char *transb,
                const int *m, const int *n, const int *k,
                const void *alpha, const void *a, const int *lda,
                const void *b, const int *ldb,
                const void *beta, void *c, const int *ldc);
}


// Compute C := A * B.
template<typename T>
void gemm(int64_t n, const T *A, int64_t lda, const T *B, int64_t ldb,
          T *C, int64_t ldc, GemmAlgorithm algo) {
    const char transa = 'N';
    const char transb = 'N';
    const T alpha = T(1.0);
    const T beta = T(0.0);
    int n32 = (int)n;
    int lda32 = (int)lda;
    int ldb32 = (int)ldb;
    int ldc32 = (int)ldc;

    switch (algo) {
        case GemmAlgorithm::NATIVE_STRASSEN:
            gemm_strassen(C, A, B, n32, n32, n32, lda32, ldb32, ldc32);
            break;
        case GemmAlgorithm::REF_GEMM:
            if constexpr (std::is_same_v<T, double>) {
                dgemm_(&transa, &transb, &n32, &n32, &n32,
                       &alpha, A, &lda32,
                       B, &ldb32,
                       &beta, C, &ldc32);
            } else {
                zgemm_(&transa, &transb, &n32, &n32, &n32,
                       static_cast<const void*>(&alpha), static_cast<const void*>(A), &lda32,
                       static_cast<const void*>(B), &ldb32,
                       static_cast<const void*>(&beta), static_cast<void*>(C), &ldc32);
            }
            break;
        case GemmAlgorithm::FIXED_STRASSEN:
            gemm_strassen_fixed(C, A, B, n32, n32, n32, lda32, ldb32, ldc32);
            break;
        case GemmAlgorithm::CUDA_NATIVE_GEMM:
            cublas_native_gemm<T>(n, A, lda, B, ldb, C, ldc);
            break;
        case GemmAlgorithm::CUDA_EMULATED_GEMM:
            cublas_emulated_gemm<T>(n, A, lda, B, ldb, C, ldc);
            break;
    }
}

// Explicit template instantiations.
template void gemm<double>(int64_t, const double*, int64_t, const double*, int64_t, double*, int64_t, GemmAlgorithm);
template void gemm<std::complex<double>>(int64_t, const std::complex<double>*, int64_t, const std::complex<double>*, int64_t, std::complex<double>*, int64_t, GemmAlgorithm);
