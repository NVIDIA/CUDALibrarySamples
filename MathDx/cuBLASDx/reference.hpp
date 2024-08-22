#ifndef CUBLASDX_EXAMPLE_REFERENCE_HPP
#define CUBLASDX_EXAMPLE_REFERENCE_HPP

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
        template<class T, class K>
        struct type_cast_op {
            __host__ __device__ constexpr T operator()(const K arg) const { return static_cast<T>(arg); }
        };

        template<cublasdx::arrangement Arr, class T, class K, class TransformOp>
        void transform_matrix(T* data_out, K* data_in, unsigned nrows, unsigned ncols, unsigned ld, TransformOp op) {
            for (unsigned j = 0; j < ncols; j++) {
                for (unsigned i = 0; i < nrows; i++) {
                    unsigned idx  = (Arr == cublasdx::arrangement::col_major) ? j * ld + i : i * ld + j;
                    data_out[idx] = op(data_in[idx]);
                }
            }
        }

        template<cublasdx::arrangement Arr, class T, class TransformOp>
        void transform_matrix(T* data, unsigned nrows, unsigned ncols, unsigned ld, TransformOp op) {
            transform_matrix<Arr, T, T, TransformOp>(data, data, nrows, ncols, ld, op);
        }

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

        template<class T, class = void>
        struct promote;

        template<class T>
        struct promote<T, std::enable_if_t<std::is_floating_point_v<T>      ||
                                           std::is_same_v<T, __half>        ||
                                           std::is_same_v<T, __nv_bfloat16>>> {
            using value_type = double;
        };

        template<class T, template<class> class Complex>
        struct promote<Complex<T>, std::enable_if_t<is_complex<Complex<T>>()>> {
            using value_type = Complex<double>;
        };

        template<class ValueType>
        using get_reference_value_type_t = typename promote<ValueType>::value_type;

        template<class BLAS, class ReferenceValueType = get_reference_value_type_t<typename BLAS::c_value_type>>
        void reference_gemm_cublas(ReferenceValueType        alpha,
                                   const ReferenceValueType* a,
                                   unsigned int              lda,
                                   cublasdx::arrangement     a_arrangement,
                                   const ReferenceValueType* b,
                                   unsigned int              ldb,
                                   cublasdx::arrangement     b_arrangement,
                                   ReferenceValueType        beta,
                                   ReferenceValueType*       c,
                                   unsigned int              ldc,
                                   cublasdx::arrangement     c_arrangement) {

            static constexpr unsigned int m = cublasdx::size_of<BLAS>::m;
            static constexpr unsigned int n = cublasdx::size_of<BLAS>::n;
            static constexpr unsigned int k = cublasdx::size_of<BLAS>::k;

            //
            // cuBLAS
            //

            // For C being row-major, computing C^T = B^T x A^T + C^T instead:
            //     C_{m,ldc} (row-major) =     A_{lda,k} (col-major) x     B_{k,ldb} (row-major) +     C_{m,ldc} (row-major)
            // [C^T]_{ldc,m} (col-major) = [B^T]_{ldb,k} (col-major) x [A^T]_{k,lda} (row-major) + [C^T]_{ldc,m} (col-major)

            cublasHandle_t handle;
            CUBLAS_CHECK_AND_EXIT(cublasCreate(&handle));
            const auto                  a_transpose  = get_cublas_transpose_mode(a_arrangement);
            const auto                  b_transpose  = get_cublas_transpose_mode(b_arrangement);
            [[maybe_unused]] const auto at_transpose = get_cublas_transpose_mode(
                (a_arrangement == cublasdx::arrangement::col_major) ? cublasdx::arrangement::row_major
                                                                    : cublasdx::arrangement::col_major);
            [[maybe_unused]] const auto bt_transpose = get_cublas_transpose_mode(
                (b_arrangement == cublasdx::arrangement::col_major) ? cublasdx::arrangement::row_major
                                                                    : cublasdx::arrangement::col_major);

            // reference always runs on double precision so complex implies complex<double>
            if constexpr (is_complex<typename BLAS::c_value_type>()) {
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

            CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
            CUBLAS_CHECK_AND_EXIT(cublasDestroy(handle));
        }

    } // namespace detail

    template<class BLAS,
             class AValueType         = typename BLAS::a_value_type,
             class BValueType         = typename BLAS::b_value_type,
             class CValueType         = typename BLAS::c_value_type,
             class ReferenceValueType = detail::get_reference_value_type_t<typename BLAS::c_value_type>,
             class ALoadOp            = cublasdx::identity,
             class BLoadOp            = cublasdx::identity,
             class CLoadOp            = cublasdx::identity,
             class CStoreOp           = cublasdx::identity>
    std::vector<ReferenceValueType> reference_gemm(CValueType                     alpha,
                                                   const std::vector<AValueType>& host_a,
                                                   unsigned int                   lda,
                                                   const std::vector<BValueType>& host_b,
                                                   unsigned int                   ldb,
                                                   CValueType                     beta,
                                                   const std::vector<CValueType>& host_c,
                                                   unsigned int                   ldc,
                                                   const ALoadOp&                 a_load_op  = {},
                                                   const BLoadOp&                 b_load_op  = {},
                                                   const CLoadOp&                 c_load_op  = {},
                                                   const CStoreOp&                c_store_op = {}) {

        static_assert(cublasdx::transpose_mode_of_a<BLAS> != cublasdx::transpose_mode::conj_transposed,
                      "Conjugated transposed is not supported in reference calculations");
        static_assert(cublasdx::transpose_mode_of_b<BLAS> != cublasdx::transpose_mode::conj_transposed,
                      "Conjugated transposed is not supported in reference calculations");

        static constexpr unsigned int                  m = cublasdx::size_of<BLAS>::m;
        static constexpr unsigned int                  n = cublasdx::size_of<BLAS>::n;
        [[maybe_unused]] static constexpr unsigned int k = cublasdx::size_of<BLAS>::k;

        using reference_value_type = ReferenceValueType;

        // use copy_hth() as a workaround to avoid FP8 type to reference type conversion error
        std::vector<reference_value_type> ref_host_a(host_a.size());
        std::vector<reference_value_type> ref_host_b(host_b.size());
        copy_hth(host_a, ref_host_a, host_a.size());
        copy_hth(host_b, ref_host_b, host_b.size());

        // Only copy elements in matrix C (examples have different global matrix for input and output C)
        std::vector<reference_value_type> ref_host_c(host_c.size());
        detail::transform_matrix<cublasdx::arrangement_of<BLAS>::c>(
            ref_host_c.data(), host_c.data(), m, n, ldc, detail::type_cast_op<ReferenceValueType, CValueType> {});

        if constexpr (!std::is_same_v<ALoadOp, cublasdx::identity>) {
            detail::transform_matrix<cublasdx::arrangement_of<BLAS>::a>(ref_host_a.data(), m, k, lda, a_load_op);
        }

        if constexpr (!std::is_same_v<BLoadOp, cublasdx::identity>) {
            detail::transform_matrix<cublasdx::arrangement_of<BLAS>::b>(ref_host_b.data(), k, n, ldb, b_load_op);
        }

        if constexpr (!std::is_same_v<CLoadOp, cublasdx::identity>) {
            detail::transform_matrix<cublasdx::arrangement_of<BLAS>::c>(ref_host_c.data(), m, n, ldc, c_load_op);
        }

        constexpr auto global_a_size = example::global_memory_size_of<BLAS>::a_size;
        constexpr auto global_b_size = example::global_memory_size_of<BLAS>::b_size;
        constexpr auto global_c_size = example::global_memory_size_of<BLAS>::c_size;

        reference_value_type* inputs;
        auto                  inputs_size       = global_a_size + global_b_size + global_c_size;
        auto                  inputs_size_bytes = inputs_size * sizeof(reference_value_type);
        CUDA_CHECK_AND_EXIT(cudaMalloc(&inputs, inputs_size_bytes));

        reference_value_type* ref_a = inputs;
        reference_value_type* ref_b = ref_a + global_a_size;
        reference_value_type* ref_c = ref_b + global_b_size;

        CUDA_CHECK_AND_EXIT(
            cudaMemcpy(ref_a, ref_host_a.data(), global_a_size * sizeof(reference_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(
            cudaMemcpy(ref_b, ref_host_b.data(), global_b_size * sizeof(reference_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(
            cudaMemcpy(ref_c, ref_host_c.data(), global_c_size * sizeof(reference_value_type), cudaMemcpyHostToDevice))
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        // Execute cuBLAS
        detail::reference_gemm_cublas<BLAS>(convert<reference_value_type>(alpha),
                                            ref_a,
                                            lda,
                                            cublasdx::arrangement_of<BLAS>::a,
                                            ref_b,
                                            ldb,
                                            cublasdx::arrangement_of<BLAS>::b,
                                            convert<reference_value_type>(beta),
                                            ref_c,
                                            ldc,
                                            cublasdx::arrangement_of<BLAS>::c);

        // Copy results to host
        CUDA_CHECK_AND_EXIT(
            cudaMemcpy(ref_host_c.data(), ref_c, global_c_size * sizeof(reference_value_type), cudaMemcpyDeviceToHost))
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
        if constexpr (!std::is_same_v<CStoreOp, cublasdx::identity>) {
            detail::transform_matrix<cublasdx::arrangement_of<BLAS>::c>(ref_host_c.data(), m, n, ldc, c_store_op);
        }
        CUDA_CHECK_AND_EXIT(cudaFree(inputs));

        return ref_host_c;
    }

    template<class BLAS,
             class AValueType         = typename BLAS::a_value_type,
             class BValueType         = typename BLAS::b_value_type,
             class CValueType         = typename BLAS::c_value_type,
             class ReferenceValueType = detail::get_reference_value_type_t<CValueType>,
             class ALoadOp            = cublasdx::identity,
             class BLoadOp            = cublasdx::identity,
             class CLoadOp            = cublasdx::identity,
             class CStoreOp           = cublasdx::identity>
    std::vector<ReferenceValueType> reference_gemm(CValueType                     alpha,
                                                   const std::vector<AValueType>& host_a,
                                                   const std::vector<BValueType>& host_b,
                                                   CValueType                     beta,
                                                   const std::vector<CValueType>& host_c,
                                                   const ALoadOp&                 a_load_op  = {},
                                                   const BLoadOp&                 b_load_op  = {},
                                                   const CLoadOp&                 c_load_op  = {},
                                                   const CStoreOp&                c_store_op = {}) {
        const auto [m, n, k] = cublasdx::size_of<BLAS>::value;
        const auto lda = cublasdx::arrangement_of<BLAS>::a == cublasdx::arrangement::col_major ? m : k;
        const auto ldb = cublasdx::arrangement_of<BLAS>::b == cublasdx::arrangement::col_major ? k : n;
        const auto ldc = cublasdx::arrangement_of<BLAS>::c == cublasdx::arrangement::col_major ? m : n;

        return reference_gemm<BLAS,
                              AValueType,
                              BValueType,
                              CValueType,
                              ReferenceValueType,
                              ALoadOp,
                              BLoadOp,
                              CLoadOp,
                              CStoreOp>(
            alpha, host_a, lda, host_b, ldb, beta, host_c, ldc, a_load_op, b_load_op, c_load_op, c_store_op);
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

#endif // CUBLASDX_EXAMPLE_REFERENCE_HPP
