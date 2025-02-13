#ifndef CUBLASDX_EXAMPLE_CUBLAS_REFERENCE_HPP
#define CUBLASDX_EXAMPLE_CUBLAS_REFERENCE_HPP

#include <type_traits>
#include <functional>
#include <cuda/std/complex>

#include <cublas_v2.h>
#include <cublasdx.hpp>

#include "../common.hpp"

#ifndef CUBLAS_CHECK_AND_EXIT
#    define CUBLAS_CHECK_AND_EXIT(error)                                                                            \
        {                                                                                                           \
            auto status = static_cast<cublasStatus_t>(error);                                                       \
            if (status != CUBLAS_STATUS_SUCCESS) {                                                                  \
                std::cout << cublasGetStatusString(status) << " (" << status << ") " << __FILE__ << ":" << __LINE__ \
                          << std::endl;                                                                             \
                std::exit(status);                                                                                  \
            }                                                                                                       \
        }
#endif // CUBLAS_CHECK_AND_EXIT

namespace example {
    constexpr cublasOperation_t get_cublas_transpose_mode(cublasdx::transpose_mode tmode) {
        if (tmode == cublasdx::transpose_mode::non_transposed) {
            return CUBLAS_OP_N;
        } else if (tmode == cublasdx::transpose_mode::transposed) {
            return CUBLAS_OP_T;
        }
        return CUBLAS_OP_C;
    }

    constexpr cublasOperation_t get_cublas_transpose_mode(cublasdx::arrangement arr) {
        if (arr == cublasdx::col_major) {
            return CUBLAS_OP_N;
        }
        return CUBLAS_OP_T;
    }

    template<class ReferenceValueType>
    void reference_gemm_cublas(unsigned int               m,
                                unsigned int              n,
                                unsigned int              k,
                                ReferenceValueType        alpha,
                                const ReferenceValueType* a,
                                unsigned int              lda,
                                cublasdx::transpose_mode  a_trans,
                                const ReferenceValueType* b,
                                unsigned int              ldb,
                                cublasdx::transpose_mode  b_trans,
                                ReferenceValueType        beta,
                                ReferenceValueType*       c,
                                unsigned int              ldc,
                                cublasdx::arrangement     c_arrangement,
                                cudaStream_t              stream = 0) {
        //
        // cuBLAS
        //

        // For C being row-major, computing C^T = B^T x A^T + C^T instead:
        //     C_{m,ldc} (row-major) =     A_{lda,k} (col-major) x     B_{k,ldb} (row-major) +     C_{m,ldc} (row-major)
        // [C^T]_{ldc,m} (col-major) = [B^T]_{ldb,k} (col-major) x [A^T]_{k,lda} (row-major) + [C^T]_{ldc,m} (col-major)

        cublasHandle_t handle;
        CUBLAS_CHECK_AND_EXIT(cublasCreate(&handle));
        CUBLAS_CHECK_AND_EXIT(cublasSetStream(handle, stream));
        const auto                  a_transpose  = get_cublas_transpose_mode(a_trans);
        const auto                  b_transpose  = get_cublas_transpose_mode(b_trans);
        [[maybe_unused]] const auto at_transpose = get_cublas_transpose_mode(
            (a_trans == cublasdx::transpose_mode::non_transposed) ? cublasdx::transpose_mode::transposed
                                                                      : cublasdx::transpose_mode::non_transposed);
        [[maybe_unused]] const auto bt_transpose = get_cublas_transpose_mode(
            (b_trans == cublasdx::transpose_mode::non_transposed) ? cublasdx::transpose_mode::transposed
                                                                      : cublasdx::transpose_mode::non_transposed);

        // reference always runs on double precision so complex implies complex<double>
        if constexpr (is_complex<ReferenceValueType>()) {
            if (c_arrangement == cublasdx::arrangement::col_major) {
                CUBLAS_CHECK_AND_EXIT(cublasZgemm(handle,
                                                  a_transpose,
                                                  b_transpose,
                                                  m,
                                                  n,
                                                  k,
                                                  reinterpret_cast<const cuDoubleComplex*>(&alpha),
                                                  reinterpret_cast<const cuDoubleComplex*>(a),
                                                  lda,
                                                  reinterpret_cast<const cuDoubleComplex*>(b),
                                                  ldb,
                                                  reinterpret_cast<const cuDoubleComplex*>(&beta),
                                                  reinterpret_cast<cuDoubleComplex*>(c),
                                                  ldc));
            } else {
                CUBLAS_CHECK_AND_EXIT(cublasZgemm(handle,
                                                  bt_transpose,
                                                  at_transpose,
                                                  n,
                                                  m,
                                                  k,
                                                  reinterpret_cast<const cuDoubleComplex*>(&alpha),
                                                  reinterpret_cast<const cuDoubleComplex*>(b),
                                                  ldb,
                                                  reinterpret_cast<const cuDoubleComplex*>(a),
                                                  lda,
                                                  reinterpret_cast<const cuDoubleComplex*>(&beta),
                                                  reinterpret_cast<cuDoubleComplex*>(c),
                                                  ldc));
            }
        } else {
            if (c_arrangement == cublasdx::arrangement::col_major) {
                CUBLAS_CHECK_AND_EXIT(
                    cublasDgemm(handle, a_transpose, b_transpose, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
            } else {
                CUBLAS_CHECK_AND_EXIT(cublasDgemm(
                    handle, bt_transpose, at_transpose, n, m, k, &alpha, b, ldb, a, lda, &beta, c, ldc));
            }
        }

        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
        CUBLAS_CHECK_AND_EXIT(cublasDestroy(handle));
    }
} // namespace example

#endif // CUBLASDX_EXAMPLE_CUBLAS_REFERENCE_HPP
