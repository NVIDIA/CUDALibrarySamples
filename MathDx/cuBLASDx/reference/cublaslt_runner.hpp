#ifndef CUBLASDX_EXAMPLE_CUBLASLT_RUNNER_HPP
#define CUBLASDX_EXAMPLE_CUBLASLT_RUNNER_HPP

#include "../common/common.hpp"

#include <cublasLt.h>

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

constexpr cublasLtOrder_t get_cublas_layout_order(cublasdx::arrangement arr) {
    return (arr == cublasdx::col_major) ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_ROW;
}

template<class T>
constexpr cudaDataType_t get_cublas_data_type() {
    if constexpr(cute::is_same_v<T, cublasdx::complex<double>>) {
        return CUDA_C_64F;
    } else if constexpr(cute::is_same_v<T, cublasdx::complex<float>>) {
        return CUDA_C_32F;
    } else if constexpr(cute::is_same_v<T, cublasdx::complex<__half>>) {
        return CUDA_C_16F;
    } else if constexpr(cute::is_same_v<T, cublasdx::complex<__nv_bfloat16>>) {
        return CUDA_C_16BF;
    } else if constexpr(cute::is_same_v<T, cublasdx::complex<int8_t>>) {
        return CUDA_C_8I;
    } else if constexpr(cute::is_same_v<T, cublasdx::complex<int32_t>>) {
        return CUDA_C_32I;
    } else if constexpr(cute::is_same_v<T, double>) {
        return CUDA_R_64F;
    } else if constexpr(cute::is_same_v<T, float>) {
        return CUDA_R_32F;
    } else if constexpr(cute::is_same_v<T, cublasdx::tfloat32_t>) {
        return CUDA_R_32F;
    } else if constexpr(cute::is_same_v<T, __half>) {
        return CUDA_R_16F;
    } else if constexpr(cute::is_same_v<T, __nv_bfloat16>) {
        return CUDA_R_16BF;
    #if CUBLASDX_EXAMPLE_SUPPORTS_FP8
    } else if constexpr(cute::is_same_v<T, __nv_fp8_e5m2>) {
        return CUDA_R_8F_E5M2;
    } else if constexpr(cute::is_same_v<T, __nv_fp8_e4m3>) {
        return CUDA_R_8F_E4M3;
    #endif
    } else if constexpr(cute::is_same_v<T, int8_t>) {
        return CUDA_R_8I;
    } else if constexpr(cute::is_same_v<T, uint8_t>) {
        return CUDA_R_8U;
    } else if constexpr(cute::is_same_v<T, int32_t>) {
        return CUDA_R_32I;
    } else {
        static_assert(sizeof(T) == 0, "Unsupported data type");
    }

    CUTE_GCC_UNREACHABLE;
}

template<class T, class Precision>
constexpr bool is_precision() {
    return cute::is_same_v<T, Precision> or cute::is_same_v<T, cublasdx::complex<Precision>>;
}

template<class I, class O>
constexpr cublasComputeType_t get_cublas_compute_type() {
    if (is_precision<O, double>()) {
        return CUBLAS_COMPUTE_64F;
    } else if (is_precision<I, cublasdx::tfloat32_t>()) {
        return CUBLAS_COMPUTE_32F_FAST_TF32;
    } else if (is_precision<O, __half>()) {
        return CUBLAS_COMPUTE_16F;
    } else if(is_precision<O, int32_t>()) {
        return CUBLAS_COMPUTE_32I;
    }

    return CUBLAS_COMPUTE_32F;
}

enum cublaslt_heuristic {
    run_default_best,
    search_for_best
};

template<class AComputeType, class BComputeType = AComputeType, class CComputeType = BComputeType>
struct cublaslt_runner {
    unsigned int result_size {};

    cublasLtOrder_t cublas_order_a {};
    cublasLtOrder_t cublas_order_b {};
    cublasLtOrder_t cublas_order_c {};

    cudaDataType_t cublas_data_type_a {};
    cudaDataType_t cublas_data_type_b {};
    cudaDataType_t cublas_data_type_c {};

    cudaDataType_t cublas_scale_type {};
    cublasComputeType_t cublas_compute_type {};

    cublasLtMatmulDesc_t operation_desc {};
    cublasLtMatmulPreference_t preference {};
    cublasLtMatrixLayout_t a_desc {};
    cublasLtMatrixLayout_t b_desc {};
    cublasLtMatrixLayout_t c_desc {};

    cublasLtHandle_t lt_handle;
    cublasLtMatmulHeuristicResult_t default_algorithm = {};

    // Note: 32MB is the suggested size of workspace for cublasLt starting from Hopper arch
    size_t workspace_size_in_bytes = 32 * 1024 * 1024;
    device_vector<char> workspace_vector = device_vector<char>(workspace_size_in_bytes);

    template<class GEMMShape, class GEMMArr, class GEMMLD>
    cublaslt_runner(GEMMShape           gemm_shape,
                    GEMMArr             gemm_arr,
                    GEMMLD              gemm_ld) {
        const auto [m, n, k] = gemm_shape;
        const auto [lda, ldb, ldc] = gemm_ld;
        const auto [arr_a, arr_b, arr_c] = gemm_arr;

        result_size = m * n;

        cublas_order_a = get_cublas_layout_order(arr_a);
        cublas_order_b = get_cublas_layout_order(arr_b);
        cublas_order_c = get_cublas_layout_order(arr_c);

        cublas_data_type_a = get_cublas_data_type<AComputeType>();
        cublas_data_type_b = get_cublas_data_type<BComputeType>();
        cublas_data_type_c = get_cublas_data_type<CComputeType>();

        cublas_scale_type = cublas_data_type_c;
        cublas_compute_type = get_cublas_compute_type<AComputeType, CComputeType>();


        CUBLAS_CHECK_AND_EXIT(cublasLtCreate(&lt_handle));

        CUBLAS_CHECK_AND_EXIT(cublasLtMatmulDescCreate(&operation_desc, cublas_compute_type, cublas_scale_type));

        CUBLAS_CHECK_AND_EXIT(cublasLtMatrixLayoutCreate(&a_desc, cublas_data_type_a, m, k, lda));
        CUBLAS_CHECK_AND_EXIT(cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &cublas_order_a, sizeof(cublas_order_a)));

        CUBLAS_CHECK_AND_EXIT(cublasLtMatrixLayoutCreate(&b_desc, cublas_data_type_b, k, n, ldb));
        CUBLAS_CHECK_AND_EXIT(cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &cublas_order_b, sizeof(cublas_order_b)));

        CUBLAS_CHECK_AND_EXIT(cublasLtMatrixLayoutCreate(&c_desc, cublas_data_type_c, m, n, ldc));
        CUBLAS_CHECK_AND_EXIT(cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &cublas_order_c, sizeof(cublas_order_c)));

        // Heuristic utils
        int returned_results = 0;

        CUBLAS_CHECK_AND_EXIT(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_CHECK_AND_EXIT(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size_in_bytes, sizeof(workspace_size_in_bytes)));

        CUBLAS_CHECK_AND_EXIT(cublasLtMatmulAlgoGetHeuristic(lt_handle, operation_desc, a_desc, b_desc, c_desc, c_desc, preference,
            1, &default_algorithm, &returned_results));

        if (returned_results == 0) {
            CUBLAS_CHECK_AND_EXIT(CUBLAS_STATUS_NOT_SUPPORTED);
        }

    }

    void execute(CComputeType const& alpha,
                 AComputeType  const* a,
                 BComputeType  const* b,
                 CComputeType const& beta,
                 CComputeType      * c,
                 cudaStream_t stream = 0) const {
        auto runner = [&](cudaStream_t stream) {
            CUBLAS_CHECK_AND_EXIT(cublasLtMatmul(lt_handle,
            operation_desc,
            reinterpret_cast<void const*>(&alpha),
            reinterpret_cast<void const*>(a),
            a_desc,
            reinterpret_cast<void const*>(b),
            b_desc,
            reinterpret_cast<void const*>(&beta),
            reinterpret_cast<void*>(c),
            c_desc,
            reinterpret_cast<void*>(c),
            c_desc,
            &default_algorithm.algo,
            workspace_vector.data(),
            workspace_size_in_bytes,
            stream));
        };

        runner(stream);
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    }

    [[nodiscard]] std::vector<CComputeType>
    execute_with_results(CComputeType    const& alpha,
                         AComputeType     const* a,
                         BComputeType     const* b,
                         CComputeType    const& beta,
                         CComputeType         * c,
                         cudaStream_t        stream = 0) const {
        std::vector<CComputeType> results(result_size);
        this->execute(alpha, a, b, beta, c, stream);
        CUDA_CHECK_AND_EXIT(cudaMemcpy(results.data(), c, results.size() * sizeof(CComputeType), cudaMemcpyDeviceToHost));

        return results;
    }


    [[nodiscard]] float
    execute_with_time(CComputeType  const& alpha,
                      AComputeType  const* a,
                      BComputeType  const* b,
                      CComputeType  const& beta,
                      CComputeType       * c,
                      unsigned             kernel_warm_up_repeats,
                      unsigned             kernel_repeats,
                      cudaStream_t         stream = 0) const {
        // Find best algorithm
        auto heuristic_runner = [&](auto algo, cudaStream_t stream) {
            CUBLAS_CHECK_AND_EXIT(cublasLtMatmul(lt_handle,
            operation_desc,
            reinterpret_cast<void const*>(&alpha),
            reinterpret_cast<void const*>(a),
            a_desc,
            reinterpret_cast<void const*>(b),
            b_desc,
            reinterpret_cast<void const*>(&beta),
            reinterpret_cast<void*>(c),
            c_desc,
            reinterpret_cast<void*>(c),
            c_desc,
            &algo,
            workspace_vector.data(),
            workspace_size_in_bytes,
            stream));
        };

        constexpr int repeat_algo_check = 5;
        const int requested_algo_count = 8;
        cublasLtMatmulHeuristicResult_t heuristic_results[requested_algo_count] = {};
        int returned_results = 0;
        int best_algo_index = 0;
        float best_algo_time = 0;

        CUBLAS_CHECK_AND_EXIT(cublasLtMatmulAlgoGetHeuristic(
            lt_handle,
            operation_desc,
            a_desc,
            b_desc,
            c_desc,
            c_desc,
            preference,
            requested_algo_count,
            heuristic_results,
            &returned_results));

        if (returned_results == 0) {
            CUBLAS_CHECK_AND_EXIT(CUBLAS_STATUS_NOT_SUPPORTED);
        }

        for (int algo_idx = 0; algo_idx < returned_results; ++algo_idx) {
            auto time = example::measure::execution([&](auto stream) { heuristic_runner(heuristic_results[algo_idx].algo, stream);},
                                                    1 /* warm up runs*/, repeat_algo_check /* kernel runs */, stream);

            if (algo_idx == 0 || time < best_algo_time) {
              best_algo_time = time;
              best_algo_index = algo_idx;
            }
        }

        auto time_cublas = example::measure::execution([&](auto stream) { heuristic_runner(heuristic_results[best_algo_index].algo, stream); },
                                                       kernel_warm_up_repeats, kernel_repeats, stream);
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        return time_cublas;
    }

    [[nodiscard]] std::tuple<float, std::vector<CComputeType>>
    execute_with_time_and_results(CComputeType     const& alpha,
                                  AComputeType     const* a,
                                  BComputeType     const* b,
                                  CComputeType     const& beta,
                                  CComputeType          * c,
                                  unsigned                kernel_warm_up_repeats,
                                  unsigned                kernel_repeats,
                                  cudaStream_t            stream = 0) const {
        auto results = this->execute_with_results(alpha, a, b, beta, c, stream);
        auto time = this->execute_with_time(alpha, a, b, beta, c, kernel_warm_up_repeats, kernel_repeats, stream);

        return std::make_tuple(time, results);
    }

    ~cublaslt_runner() {
        if (preference) CUBLAS_CHECK_AND_EXIT(cublasLtMatmulPreferenceDestroy(preference));
        if (c_desc) CUBLAS_CHECK_AND_EXIT(cublasLtMatrixLayoutDestroy(c_desc));
        if (b_desc) CUBLAS_CHECK_AND_EXIT(cublasLtMatrixLayoutDestroy(b_desc));
        if (a_desc) CUBLAS_CHECK_AND_EXIT(cublasLtMatrixLayoutDestroy(a_desc));
        if (operation_desc) CUBLAS_CHECK_AND_EXIT(cublasLtMatmulDescDestroy(operation_desc));
    }
};
}

#endif // CUBLASDX_EXAMPLE_CUBLASLT_RUNNER_HPP
