#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "../common.hpp"
#include "../random.hpp"

#include "io_strided_conv_smem.hpp"
#include "kernels.hpp"
#include "reference.hpp"

// Used for getting stable performance results
inline constexpr unsigned int cufftdx_example_warm_up_runs     = 5;
inline constexpr unsigned int cufftdx_example_performance_runs = 15;

using namespace example;

// This function performs fused 3D FFT convolution with pre- and post-processing, ie. it executes:
// pre-processing, front 3D FFT, filter element-wise function (a.k.a. kernel), back 3D FFT.
// Front and back FFTs can either be inverse and forward FFTs, or more commonly forward and inverse.
template<int Batches, bool IsForwardConv, class FFTXPartial, class FFTYPartial, class FFTZPartial, class LoadFunctor, class FilterFunctor, class StoreFunctor, typename ValueType>
auto cufftdx_3d_convolution(ValueType* input, ValueType* output, cudaStream_t stream) {
    using namespace cufftdx;

    using id_op = example::identity;

    // Retrieve precision information from description type
    using precision          = cufftdx::precision_of_t<FFTXPartial>;
    constexpr bool is_double = std::is_same_v<precision, double>;
    using vector_type        = std::conditional_t<is_double, double2, float2>;
    using value_type         = ValueType;

    // Retrieve size information from description types
    static constexpr unsigned int fft_size_x      = cufftdx::size_of<FFTXPartial>::value;
    static constexpr unsigned int fft_size_y      = cufftdx::size_of<FFTYPartial>::value;
    static constexpr unsigned int fft_size_z      = cufftdx::size_of<FFTZPartial>::value;
    static constexpr unsigned int flat_batch_size = fft_size_x * fft_size_y * fft_size_z;

    // Create and configure kernel for the X dimension (strided)
    using FFTX  = decltype(FFTXPartial() + Direction < IsForwardConv ? fft_direction::forward : fft_direction::inverse > ());
    using IFFTX = decltype(FFTXPartial() + Direction < IsForwardConv ? fft_direction::inverse : fft_direction::forward > ());

    using FFTY  = decltype(FFTYPartial() + Direction < IsForwardConv ? fft_direction::forward : fft_direction::inverse > ());
    using IFFTY = decltype(FFTYPartial() + Direction < IsForwardConv ? fft_direction::inverse : fft_direction::forward > ());

    using FFTZ  = decltype(FFTZPartial() + Direction < IsForwardConv ? fft_direction::forward : fft_direction::inverse > ());
    using IFFTZ = decltype(FFTZPartial() + Direction < IsForwardConv ? fft_direction::inverse : fft_direction::forward > ());

    static constexpr auto x_fpb = FFTX::ffts_per_block;
    static constexpr auto y_fpb = FFTY::ffts_per_block;
    static constexpr auto z_fpb = FFTZ::ffts_per_block;

    static constexpr unsigned int x_batches = flat_batch_size / fft_size_x;
    static constexpr unsigned int y_batches = flat_batch_size / fft_size_y;
    static constexpr unsigned int z_batches = flat_batch_size / fft_size_z;

    // This is for C2C only, so front/back FFT executions require same memory accesses
    // hence "Front" parameter is true in all cases
    using io_x = io_strided_conv_smem<dimension::x, true, Batches, FFTX, IFFTX, FFTY, IFFTY, FFTZ, IFFTZ>;
    using io_y = io_strided_conv_smem<dimension::y, true, Batches, FFTX, IFFTX, FFTY, IFFTY, FFTZ, IFFTZ>;
    using io_z = io_strided_conv_smem<dimension::z, true, Batches, FFTX, IFFTX, FFTY, IFFTY, FFTZ, IFFTZ>;

    cudaError_t err;

    auto workspace_x = cufftdx::make_workspace<FFTX>(err, stream);
    auto workspace_y = cufftdx::make_workspace<FFTY>(err, stream);
    auto workspace_z = cufftdx::make_workspace<FFTZ>(err, stream);

    // Increase max shared memory if needed (includes extra padding)
    constexpr int x_max_bytes = io_x::get_shared_bytes();
    constexpr int y_max_bytes = io_y::get_shared_bytes();
    constexpr int z_max_bytes = io_z::get_shared_bytes();

    // Define kernels to set maximal shared memory in CUDA runtime
    // Refer to kernels.hpp for a detailed kernel explanation.
    auto set_kernel_shared_size = [](auto kernel, auto size) {
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            size));
    };

    auto kernel_z_front = fft_kernel<FFTZ, io_z, LoadFunctor, id_op, value_type>;
    set_kernel_shared_size(kernel_z_front, z_max_bytes);

    // No transform ops for middle dimension
    auto kernel_y_front = fft_kernel<FFTY, io_y, id_op, id_op, value_type>;
    set_kernel_shared_size(kernel_y_front, y_max_bytes);

    auto kernel_x = convolution_kernel<FFTX, IFFTX, FilterFunctor, io_x>;
    set_kernel_shared_size(kernel_x, x_max_bytes);

    // No transform ops for middle dimension
    auto kernel_y_back = fft_kernel<IFFTY, io_y, id_op, id_op, value_type>;
    set_kernel_shared_size(kernel_y_back, y_max_bytes);

    auto kernel_z_back = fft_kernel<IFFTZ, io_z, id_op, StoreFunctor, value_type>;
    set_kernel_shared_size(kernel_z_back, z_max_bytes);

    // Execute cuFFTDx in Z-Y-X order
    auto cufftdx_execution = [&](cudaStream_t stream) {
        // CUDA Grid configuration is as follows:
        // Grid --> (Total Subbatches / FPB, Batches, 1)
        // Block --> (Size / EPT, FPB)
        kernel_z_front<<<dim3 {example::div_up(z_batches, z_fpb), Batches, 1}, FFTZ::block_dim, z_max_bytes, stream>>>(
            z_batches,
            input,
            output,
            workspace_z);
        kernel_y_front<<<dim3 {example::div_up(y_batches, y_fpb), Batches, 1}, FFTY::block_dim, y_max_bytes, stream>>>(
            y_batches,
            output,
            output,
            workspace_y);
        // Convolution is performed in the strided dimension to save on strided global memory transfers
        // which occur otherwise.
        kernel_x<<<dim3 {example::div_up(x_batches, x_fpb), Batches, 1}, FFTX::block_dim, x_max_bytes, stream>>>(
            x_batches,
            output,
            output,
            workspace_x);
        kernel_y_back<<<dim3 {example::div_up(y_batches, y_fpb), Batches, 1}, FFTY::block_dim, y_max_bytes, stream>>>(
            y_batches,
            output,
            output,
            workspace_y);
        kernel_z_back<<<dim3 {example::div_up(z_batches, z_fpb), Batches, 1}, FFTZ::block_dim, z_max_bytes, stream>>>(
            z_batches,
            output,
            output,
            workspace_z);
    };

    // Correctness run
    cufftdx_execution(stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results to host
    const size_t             flat_fft_size       = fft_size_x * fft_size_y * fft_size_z;
    const size_t             flat_fft_size_bytes = flat_fft_size * sizeof(vector_type);
    std::vector<vector_type> output_host(Batches * flat_fft_size, {std::numeric_limits<precision>::quiet_NaN(), std::numeric_limits<precision>::quiet_NaN()});
    CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), output, Batches * flat_fft_size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Performance measurements
    auto time = example::measure_execution_ms(
        cufftdx_execution,
        cufftdx_example_warm_up_runs,
        cufftdx_example_performance_runs,
        stream);

    // Return results
    return example::fft_results<vector_type> {output_host, (time / cufftdx_example_performance_runs)};
}

template<int Arch>
int conv_3d() {
    using namespace cufftdx;

    // 3D Convolution configuration
    static constexpr unsigned int batches = 100;

    // X - the outermost (most strided) dimension
    static constexpr unsigned int fft_size_x = 81;
    static constexpr unsigned int x_ept      = 9;
    static constexpr unsigned int x_fpb      = 16;

    // Y - the middle dimension, strided
    static constexpr unsigned int fft_size_y = 81;
    static constexpr unsigned int y_ept      = 9;
    static constexpr unsigned int y_fpb      = 16;

    // Z - the contiguous dimension
    static constexpr unsigned int fft_size_z = 128;
    static constexpr unsigned int z_ept      = 16;
    static constexpr unsigned int z_fpb      = 16;

    // Definition of functors for preprocessing, filtering
    // and postprocessing. These will be fused with FFT kernels
    using load_functor   = example::rational_scaler<1, 1>;
    using filter_functor = example::rational_scaler<3, 5>;
    using store_functor  = example::rational_scaler<3, 4>;

    // Is this a forward or inverse convolution
    static constexpr bool is_forward = false;

    // Only FP32 and FP64 supported for cuFFT reference
    constexpr bool is_double_precision = true;
    using precision                    = std::conditional_t<is_double_precision, double, float>;

    // Create cuFFTDx description type summarizing all information
    // passed above regarding the outermost dimension.
    using fftx_partial = decltype(Block() + Size<fft_size_x>() + Type<fft_type::c2c>() +
                                  ElementsPerThread<x_ept>() + FFTsPerBlock<x_fpb>() +
                                  Precision<precision>() + SM<Arch>());

    // Create cuFFTDx description type summarizing all information
    // passed above regarding the middle dimension.
    using ffty_partial = decltype(Block() + Size<fft_size_y>() + Type<fft_type::c2c>() +
                                  ElementsPerThread<y_ept>() + FFTsPerBlock<y_fpb>() +
                                  Precision<precision>() + SM<Arch>());

    // Create cuFFTDx description type summarizing all information
    // passed above regarding the contiguous dimension.
    using fftz_partial = decltype(Block() + Size<fft_size_z>() + Type<fft_type::c2c>() +
                                  ElementsPerThread<z_ept>() + FFTsPerBlock<z_fpb>() +
                                  Precision<precision>() + SM<Arch>());

    // Helper types
    using value_type = cufftdx::complex<precision>;

    // Generate random input data on host
    const unsigned int flat_fft_size = fft_size_x * fft_size_y * fft_size_z;

    auto host_input = example::get_random_complex_data<precision>(batches * flat_fft_size, -1, 1);

    // Allocate managed memory for device input/output
    value_type* input;
    value_type* output;
    const auto  flat_fft_size_bytes = flat_fft_size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&input, batches * flat_fft_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output, batches * flat_fft_size_bytes));

    // Copy input to the device
    CUDA_CHECK_AND_EXIT(cudaMemcpy(input, host_input.data(), batches * flat_fft_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    // Run cuFFTDx
    auto cufftdx_results = cufftdx_3d_convolution<batches, is_forward, fftx_partial, ffty_partial, fftz_partial, load_functor, filter_functor, store_functor>(input, output, stream);

    // Run cuFFT
    auto cufft_results = cufft_3d_convolution<false, is_forward, load_functor, filter_functor, store_functor>(fft_size_x, fft_size_y, fft_size_z, batches, input, output, stream, cufftdx_example_warm_up_runs, cufftdx_example_performance_runs);

    // Clean-up
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(input));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    // Check if cuFFTDx results are correct
    auto fft_error =
        example::fft_signal_error::calculate_for_complex_values(cufftdx_results.output, cufft_results.output);

    std::cout << "FFT: (" << fft_size_x << ", " << fft_size_y << ", " << fft_size_z << ")\n";

    bool success = fft_error.l2_relative_error < 0.001;
    std::cout << "Correctness results:\n";
    std::cout << "L2 error: " << fft_error.l2_relative_error << "\n";
    std::cout << "Peak error (index: " << fft_error.peak_error_index << "): " << fft_error.peak_error << "\n";

    // Print performance results
    if (success) {
        std::cout << "\nPerformance results:\n";
        std::cout << std::setw(10) << "cuFFTDx: " << cufftdx_results.avg_time_in_ms << " [ms]\n";
        std::cout << std::setw(10) << "cuFFT + thrust: " << cufft_results.avg_time_in_ms << " [ms]\n";
    }

    if (success) {
        std::cout << "Success\n";
        return 0;
    } else {
        std::cout << "Failure\n";
        return 1;
    }
}

template<unsigned int Arch>
struct conv_3d_functor {
    int operator()() {
        return conv_3d<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner<conv_3d_functor>();
}
