#pragma once

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>

#include "../common/block_io.hpp"
#include "../common/common.hpp"
#include "../common/random.hpp"

namespace example {

    namespace detail {
        auto get_cufft_convolution_plans(int x, int in_x, int out_x, int y, int in_y, int out_y, int z, int in_z, int out_z, int batches, bool is_real_convolution, bool is_double_precision, bool is_forward_convolution, cudaStream_t stream = 0) {
            cufftHandle plan_front, plan_back;

            const auto cufft_forward_double_enum = is_real_convolution ? CUFFT_D2Z : CUFFT_Z2Z;
            const auto cufft_inverse_double_enum = is_real_convolution ? CUFFT_Z2D : CUFFT_Z2Z;
            const auto cufft_forward_float_enum  = is_real_convolution ? CUFFT_R2C : CUFFT_C2C;
            const auto cufft_inverse_float_enum  = is_real_convolution ? CUFFT_C2R : CUFFT_C2C;

            const auto cufft_forward_enum = is_double_precision ? cufft_forward_double_enum : cufft_forward_float_enum;
            const auto cufft_inverse_enum = is_double_precision ? cufft_inverse_double_enum : cufft_inverse_float_enum;

            const auto cufft_front_enum = is_forward_convolution ? cufft_forward_enum : cufft_inverse_enum;
            const auto cufft_back_enum  = is_forward_convolution ? cufft_inverse_enum : cufft_forward_enum;

            // This will execute a 2D FFT
            int n = 3;

            // The dimensions will be outermost dimensions,
            // so the ones with biggest strides
            int dims[] = {x, y, z};

            // element index is computed as:
            // input[batch][x][y] = input[batch * idist + ((x * inembed[1]) + y) * istride]
            int istride = 1;

            // Stride between input batches
            int idist = in_x * in_y * in_z;

            // element index is computed as:
            // output[batch][x][y] = output[batch * odist + ((x * onembed[1]) + y) * ostride]
            int ostride = 1;

            // Stride between output batches
            int odist = out_x * out_y * out_z;

            int n_batches = batches;

            CUFFT_CHECK_AND_EXIT(cufftPlanMany(&plan_front, n, dims, NULL, istride, idist, NULL, ostride, odist, cufft_front_enum, n_batches));
            CUFFT_CHECK_AND_EXIT(cufftPlanMany(&plan_back, n, dims, NULL, ostride, odist, NULL, istride, idist, cufft_back_enum, n_batches));

            // Set plan streams
            CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_front, stream));
            CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_back, stream));

            return std::make_tuple(plan_front, plan_back);
        }

        template<bool IsReal, bool IsForward, bool IsDouble, typename... Args>
        cufftResult execute_cufft(Args&&... args) {
            cufftResult status;
            if constexpr (IsReal and IsForward and IsDouble) {
                status = cufftExecD2Z(std::forward<Args>(args)...);
            } else if constexpr (IsReal and IsForward and not IsDouble) {
                status = cufftExecR2C(std::forward<Args>(args)...);
            } else if constexpr (IsReal and not IsForward and IsDouble) {
                status = cufftExecZ2D(std::forward<Args>(args)...);
            } else if constexpr (IsReal and not IsForward and not IsDouble) {
                status = cufftExecC2R(std::forward<Args>(args)...);
            } else if constexpr (not IsReal and IsForward and IsDouble) {
                status = cufftExecZ2Z(std::forward<Args>(args)..., CUFFT_FORWARD);
            } else if constexpr (not IsReal and IsForward and not IsDouble) {
                status = cufftExecC2C(std::forward<Args>(args)..., CUFFT_FORWARD);
            } else if constexpr (not IsReal and not IsForward and IsDouble) {
                status = cufftExecZ2Z(std::forward<Args>(args)..., CUFFT_INVERSE);
            } else if constexpr (not IsReal and not IsForward and not IsDouble) {
                status = cufftExecC2C(std::forward<Args>(args)..., CUFFT_INVERSE);
            }
            return status;
        }
    } // namespace detail

    template<bool IsReal, bool IsForward, class LoadFunctor, class FilterFunctor, class StoreFunctor, typename IOType>
    auto cufft_3d_convolution(int          fft_size_x,
                              int          fft_size_y,
                              int          fft_size_z,
                              int          batches,
                              IOType*      input_dx,
                              IOType*      output_dx,
                              cudaStream_t stream,
                              unsigned int warm_up_runs,
                              unsigned int perf_runs) {
        constexpr bool is_real_conv    = IsReal;
        constexpr bool is_forward_conv = IsForward;
        using cufft_type               = cufft_value_type_t<IOType>;
        using precision                = get_value_type_t<IOType>;
        using inter_type               = std::conditional_t<IsReal and not IsForward, precision, cufftdx::complex<precision>>;
        using cufft_inter_type         = cufft_value_type_t<inter_type>;

        constexpr bool is_r2c_conv = is_real_conv and is_forward_conv;
        constexpr bool is_c2r_conv = is_real_conv and not is_forward_conv;

        int x_input_length = fft_size_x;
        int y_input_length = fft_size_y;
        int z_input_length = is_c2r_conv ? (fft_size_z / 2 + 1) : fft_size_z;

        int x_output_length = fft_size_x;
        int y_output_length = fft_size_y;
        int z_output_length = is_r2c_conv ? (fft_size_z / 2 + 1) : fft_size_z;

        auto input  = reinterpret_cast<cufft_type*>(input_dx);
        auto output = reinterpret_cast<cufft_type*>(output_dx);
        // This buffer needs to be created for intermediate FFT results, since it cannot
        // trivially happen in-place without being padded in global
        cufft_inter_type* inter;
        const auto        conv_length           = x_output_length * y_output_length * z_output_length;
        const auto        flat_inter_size_bytes = conv_length * sizeof(cufft_inter_type);
        CUDA_CHECK_AND_EXIT(cudaMalloc(&inter, batches * flat_inter_size_bytes));

        constexpr bool is_double = std::is_same_v<precision, double>;
        using host_ref_type      = std::conditional_t<is_r2c_conv, precision, typename vector_type<precision>::type>;

        // Create cuFFT plan
        int flat_input_size  = x_input_length * y_input_length * z_input_length;
        int flat_output_size = x_output_length * y_output_length * z_output_length;

        std::tuple cufft_plans = detail::get_cufft_convolution_plans(
            fft_size_x, x_input_length, x_output_length, fft_size_y, y_input_length, y_output_length, fft_size_z, z_input_length, z_output_length, batches, is_real_conv, is_double, is_forward_conv, stream);

        // Cannot use structured binding because capturing bindings inside of an lambda
        // is a C++20 feature.
        auto cufft_plan_front = std::get<0>(cufft_plans);
        auto cufft_plan_back  = std::get<1>(cufft_plans);

        // Thrust preparation
#if (THRUST_VERSION >= 101600)
        auto execution_policy = thrust::cuda::par_nosync.on(stream);
#else
        auto execution_policy = thrust::cuda::par.on(stream);
#endif

        // Execute cuFFT
        auto cufft_execution = [&](cudaStream_t /* stream */) {
            constexpr bool use_preprocess = not std::is_same_v<LoadFunctor, example::identity>;
            if constexpr (use_preprocess) {
                thrust::transform(execution_policy, input, input + batches * flat_input_size, output, LoadFunctor {});
            }
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
            auto cufft_res = detail::execute_cufft<is_real_conv, is_forward_conv, is_double>(cufft_plan_front, use_preprocess ? output : input, inter);
            CUFFT_CHECK_AND_EXIT(cufft_res);
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            if constexpr (not std::is_same_v<FilterFunctor, example::identity>) {
                // This needs to be performed on always the max number of elements present, because cuFFT
                // performs a padded FFT when R2C / C2R has to be done inplace.
                thrust::transform(execution_policy, inter, inter + batches * flat_output_size, inter, FilterFunctor {});
            }
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            cufft_res = detail::execute_cufft<is_real_conv, not is_forward_conv, is_double>(cufft_plan_back, inter, output);
            CUFFT_CHECK_AND_EXIT(cufft_res);
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            if constexpr (not std::is_same_v<StoreFunctor, example::identity>) {
                thrust::transform(execution_policy, output, output + batches * flat_input_size, output, StoreFunctor {});
            }
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
        };

        // Correctness run
        cufft_execution(stream);
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
        // Copy results to host
        const size_t               flat_input_size_bytes = flat_input_size * sizeof(host_ref_type);
        std::vector<host_ref_type> output_host(batches * flat_input_size, example::get_nan<host_ref_type>());
        CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), output, batches * flat_input_size_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        // Performance measurements
        auto time = example::measure_execution_ms(
            cufft_execution,
            warm_up_runs,
            perf_runs,
            stream);

        // Clean-up
        CUFFT_CHECK_AND_EXIT(cufftDestroy(cufft_plan_front));
        CUFFT_CHECK_AND_EXIT(cufftDestroy(cufft_plan_back));
        CUDA_CHECK_AND_EXIT(cudaFree(inter));

        // Return results
        return example::fft_results<host_ref_type> {output_host, (time / perf_runs)};
    }

} // namespace example
