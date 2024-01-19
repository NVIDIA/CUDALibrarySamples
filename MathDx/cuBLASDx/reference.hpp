#include <type_traits>
#include <cuda/std/complex>

#include <cublas_v2.h>
#include <cublasdx.hpp>

#include "common.hpp"

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
    namespace detail {
        constexpr cublasOperation_t get_cublas_transpose_mode(cublasdx::transpose_mode tmode) {
            if (tmode == cublasdx::transpose_mode::non_transposed) {
                return CUBLAS_OP_N;
            } else if (tmode == cublasdx::transpose_mode::transposed) {
                return CUBLAS_OP_T;
            }
            return CUBLAS_OP_C;
        }

        template <class T, class = void>
        struct promote;

        template <class T>
        struct promote<T, std::enable_if_t<std::is_floating_point_v<T> || std::is_same_v<T, __half>>> {
            using value_type = double;
        };

        template <class T, template<class> class Complex>
        struct promote<Complex<T>, std::enable_if_t<is_complex<Complex<T>>()>> {
            using value_type = Complex<double>;
        };

        template<class ValueType>
        using get_reference_value_type_t = typename promote<ValueType>::value_type;

        template<class BLAS, class ValueType, class ReferenceValueType = get_reference_value_type_t<ValueType>>
        void reference_gemm_cublas(ReferenceValueType        alpha,
                                   const ReferenceValueType* a,
                                   unsigned int              lda,
                                   const ReferenceValueType* b,
                                   unsigned int              ldb,
                                   ReferenceValueType        beta,
                                   ReferenceValueType*       c,
                                   unsigned int              ldc) {
            static constexpr unsigned int m = cublasdx::size_of<BLAS>::m;
            static constexpr unsigned int n = cublasdx::size_of<BLAS>::n;
            static constexpr unsigned int k = cublasdx::size_of<BLAS>::k;

            //
            // cuBLAS
            //
            cublasHandle_t handle;
            CUBLAS_CHECK_AND_EXIT(cublasCreate(&handle));
            const auto a_transpose = get_cublas_transpose_mode(cublasdx::transpose_mode_of<BLAS>::a_transpose_mode);
            const auto b_transpose = get_cublas_transpose_mode(cublasdx::transpose_mode_of<BLAS>::b_transpose_mode);

            // reference always runs on double precision so complex implies complex<double>
            if constexpr (is_complex<typename BLAS::value_type>()) {
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
                CUBLAS_CHECK_AND_EXIT(cublasDgemm(handle, a_transpose, b_transpose, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
            }

            CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
            CUBLAS_CHECK_AND_EXIT(cublasDestroy(handle));
        }

    } // namespace detail

    template<class BLAS,
             class ValueType,
             class ReferenceValueType = detail::get_reference_value_type_t<ValueType>>
    std::vector<ReferenceValueType> reference_gemm(ValueType                     alpha,
                                                   const std::vector<ValueType>& host_a,
                                                   unsigned int                  lda,
                                                   const std::vector<ValueType>& host_b,
                                                   unsigned int                  ldb,
                                                   ValueType                     beta,
                                                   const std::vector<ValueType>& host_c,
                                                   unsigned int                  ldc) {
        using reference_value_type = ReferenceValueType;

        std::vector<reference_value_type> ref_host_a(host_a.begin(), host_a.end());
        std::vector<reference_value_type> ref_host_b(host_b.begin(), host_b.end());
        std::vector<reference_value_type> ref_host_c(host_c.begin(), host_c.end());

        reference_value_type* inputs;
        auto inputs_size       = BLAS::a_size + BLAS::b_size + BLAS::c_size;
        auto inputs_size_bytes = inputs_size * sizeof(reference_value_type);
        CUDA_CHECK_AND_EXIT(cudaMalloc(&inputs, inputs_size_bytes));

        reference_value_type* ref_a = inputs;
        reference_value_type* ref_b = ref_a + (BLAS::a_size);
        reference_value_type* ref_c = ref_b + (BLAS::b_size);

        CUDA_CHECK_AND_EXIT(
            cudaMemcpy(ref_a, ref_host_a.data(), BLAS::a_size * sizeof(reference_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(
            cudaMemcpy(ref_b, ref_host_b.data(), BLAS::b_size * sizeof(reference_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(
            cudaMemcpy(ref_c, ref_host_c.data(), BLAS::c_size * sizeof(reference_value_type), cudaMemcpyHostToDevice))
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        // Execute cuBLAS
        detail::reference_gemm_cublas<BLAS, ValueType>(convert<reference_value_type>(alpha),
                                    ref_a,
                                    lda,
                                    ref_b,
                                    ldb,
                                    convert<reference_value_type>(beta),
                                    ref_c,
                                    ldc);

        // Copy results to host
        CUDA_CHECK_AND_EXIT(
            cudaMemcpy(ref_host_c.data(), ref_c, BLAS::c_size * sizeof(reference_value_type), cudaMemcpyDeviceToHost))
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        CUDA_CHECK_AND_EXIT(cudaFree(inputs));

        return ref_host_c;
    }

    template<class BLAS,
             class ValueType          = typename BLAS::value_type,
             class ReferenceValueType = detail::get_reference_value_type_t<ValueType>>
    std::vector<ReferenceValueType> reference_gemm(ValueType                     alpha,
                                                   const std::vector<ValueType>& host_a,
                                                   const std::vector<ValueType>& host_b,
                                                   ValueType                     beta,
                                                   const std::vector<ValueType>& host_c) {

        const auto [lda, ldb, ldc] = cublasdx::leading_dimension_of_v<BLAS>;
        return reference_gemm<BLAS, ValueType>(alpha, host_a, lda, host_b, ldb, beta, host_c, ldc);
    }

    template<typename T, typename K>
    double relative_l2_norm(const std::vector<T>& results, const std::vector<K>& reference) {
        if (results.size() != reference.size()) {
            throw std::invalid_argument("Vectors must have the same length.");
        }

        double l2_norm_diff      = 0.0;
        double l2_norm_reference = 0.0;
        for (size_t i = 0; i < results.size(); ++i) {
            double diff = detail::cbabs(static_cast<K>(results[i]) - reference[i]);
            l2_norm_diff += diff * diff;
            l2_norm_reference += detail::cbabs(reference[i]) * detail::cbabs(reference[i]);
        }

        double relative_l2_norm = std::sqrt(l2_norm_diff / l2_norm_reference);
        return relative_l2_norm;
    }

    template<typename T, typename K>
    bool check(const std::vector<T>& results, const std::vector<K>& reference) {
        return (relative_l2_norm(results, reference) < 0.01);
    }
} // namespace example
