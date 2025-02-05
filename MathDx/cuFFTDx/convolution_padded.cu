#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>
#ifdef CUFFTDX_EXAMPLES_CUFFT_CALLBACK
#include <cufftXt.h>
#endif

#include "padded_io.hpp"
#include "common.hpp"
#include "random.hpp"

constexpr float scaling_factor = 123.f;

template<int SignalLength, class FFT, class IFFT,
         typename ComplexType = typename FFT::value_type, typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__ void convolution_kernel(ScalarType*                   in,
                                                                                 ScalarType*                   out,
                                                                                 typename FFT::workspace_type  workspace,
                                                                                 typename IFFT::workspace_type workspace_inverse) {
    using complex_type = typename FFT::value_type;
    using scalar_type  = typename complex_type::value_type;

    constexpr bool needs_padding = (SignalLength != cufftdx::size_of<FFT>::value);

    using input_utils = std::conditional_t<needs_padding, example::io_padded<FFT, SignalLength>, example::io<FFT>>;
    using output_utils = std::conditional_t<needs_padding, example::io_padded<IFFT, SignalLength>, example::io<IFFT>>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;

    // Load data from global memory to registers
    input_utils::load(in, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Scale values
    scalar_type scale = (1.0 / cufftdx::size_of<FFT>::value) * scaling_factor;
    for (unsigned int i = 0; i < FFT::storage_size; i++) {
        thread_data[i].x *= scale;
        thread_data[i].y *= scale;
    }

    // Execute inverse FFT
    IFFT().execute(thread_data, shared_mem, workspace_inverse);

    // Save results
    output_utils::store(thread_data, out, local_fft_id);
}

// Scaling kernel; transforms data between cuFFTs.
template<unsigned int fft_size>
__global__ void scaling_kernel(cufftComplex*      data,
                               const unsigned int input_size,
                               const unsigned int ept) {

    static constexpr float scale = (1.f / fft_size) * scaling_factor;

    cufftComplex temp;
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < ept; i++) {
        if (index < input_size) {
            temp = data[index];
            temp.x *= scale;
            temp.y *= scale;
            data[index] = temp;
            index += blockDim.x * gridDim.x;
        }
    }
}

template<int SignalLength, class FFT, class IFFT, typename ComplexType = typename FFT::value_type,
         typename ScalarType = typename ComplexType::value_type>
example::fft_results<float>
measure_cufftdx(const unsigned int       kernel_runs,
                const unsigned int       warm_up_runs,
                const unsigned int       correctness_batches,
                const unsigned int       cuda_blocks,
                ScalarType*              input_data,
                ScalarType*              output_data,
                cudaStream_t             stream) {

    using namespace cufftdx;
    using complex_type = typename FFT::value_type;

    const auto size = SignalLength * correctness_batches;

    // create workspaces for FFT and IFFT
    cudaError_t error_code = cudaSuccess;
    auto        workspace  = make_workspace<FFT>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_inverse = make_workspace<IFFT>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);

    // Initial execution for correctness
    convolution_kernel<SignalLength, FFT, IFFT><<<cuda_blocks, FFT::block_dim, FFT::shared_memory_size, stream>>>(
                input_data, output_data, workspace, workspace_inverse);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Create host result vector to be returned later
    CUFFTDX_STD::vector<float> host_output(size);
    auto size_bytes = size * sizeof(float);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(reinterpret_cast<void*>(host_output.data()),
                                   reinterpret_cast<void*>(output_data),
                                   size_bytes, cudaMemcpyDeviceToHost));

    // run cuFFTDx
    float time = example::measure_execution_ms(
        [&](cudaStream_t stream) {
            // There are (ffts_per_block * fft_size * cuda_blocks) elements
            convolution_kernel<SignalLength, FFT, IFFT><<<cuda_blocks, FFT::block_dim, FFT::shared_memory_size, stream>>>(
                input_data, output_data, workspace, workspace_inverse);
        },
        warm_up_runs, kernel_runs, stream);

    return {host_output, time};
}

template<unsigned int SignalLength, typename PaddedFFT>
example::fft_results<float>
measure_cufft(const unsigned int kernel_runs,
              const unsigned int warm_up_runs,
              const unsigned int correctness_batches,
              const unsigned int batch_size,
              cufftReal*          input_data,
              cufftReal*          output_data,
              cufftComplex*       work_buffer,
              cudaStream_t        stream) {

    static constexpr unsigned int block_dim_scaling_kernel = 1024;
    const auto size = SignalLength * correctness_batches;

    // Calculating parameters for scaling_kernel execution.
    // Get maximum number of running CUDA blocks per multiprocessor.
    int blocks_per_multiprocessor = 0;

    // cuFFT always uses layout equivalent to cuFFTDx complex_layout::natural
    // hence the hard coded indexing.
    constexpr auto convolved_data_length = SignalLength / 2 + 1;
    CUDA_CHECK_AND_EXIT(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor,
                                                      scaling_kernel<SignalLength>,
                                                      block_dim_scaling_kernel,
                                                      0));

    // Get maximum number of CUDA blocks running on all multiprocessors.
    // This many CUDA blocks will be run for simple_kernel.
    const unsigned int cuda_blocks = blocks_per_multiprocessor * example::get_multiprocessor_count();

    const unsigned int input_length        = convolved_data_length * batch_size;
    const unsigned int elements_per_block  = (input_length + cuda_blocks - 1) / cuda_blocks;
    const unsigned int elements_per_thread = (elements_per_block + block_dim_scaling_kernel - 1) / block_dim_scaling_kernel;

    // prepare cuFFT runs
    cufftHandle plan_forward, plan_inverse;
    CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan_forward, SignalLength, CUFFT_R2C, batch_size));
    CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan_inverse, SignalLength, CUFFT_C2R, batch_size));

    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_forward, stream));
    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_inverse, stream));

    // Single execution for correctness check
    if (cufftExecR2C(plan_forward, input_data, work_buffer) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
        return {};
    }
    scaling_kernel<SignalLength>
        <<<cuda_blocks, block_dim_scaling_kernel, 0, stream>>>(work_buffer, input_length, elements_per_thread);
    if (cufftExecC2R(plan_inverse, work_buffer, output_data) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
        return {};
    }

    // Create host result vector to be returned later
    CUFFTDX_STD::vector<float> host_output(size);
    auto size_bytes = size * sizeof(float);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(reinterpret_cast<void*>(host_output.data()),
                                   reinterpret_cast<void*>(output_data),
                                   size_bytes, cudaMemcpyDeviceToHost));

    // run convolution
    float time_cufft = example::measure_execution_ms(
        [&](cudaStream_t stream) {
            if (cufftExecR2C(plan_forward, input_data, work_buffer) != CUFFT_SUCCESS) {
                fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
                return;
            }
            scaling_kernel<SignalLength>
                <<<cuda_blocks, block_dim_scaling_kernel, 0, stream>>>(work_buffer, input_length, elements_per_thread);
            if (cufftExecC2R(plan_inverse, work_buffer, output_data) != CUFFT_SUCCESS) {
                fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
                return;
            }
        },
        warm_up_runs, kernel_runs, stream);

    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan_forward));
    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan_inverse));
    return {host_output, time_cufft};
}

// This example showcases a method for using cuFFTDx features to get superior
// performance in performing a R2C / C2R convolution. To achieve this goal
// the real_mode::folded optimization is used and input data is padded with
// zeros to the closest power of 2. Final performance is compared with cuFFTDx
// executed without padding and optimization and cuFFT based convolution.
// --------------------------------------------------------------------------
// Data is generated on host, copied to device buffer and processed by FFTs.
// Each cuFFTDx execution runs one kernel, each cuFFT execution - three kernels.
// The experiment runs with the following principles:
// - at least 2GB of data is allocated in GPU and transformed by both convolutions,
// - for cuFFTDx kernel run, number of CUDA blocks is always divisible
//   by maximum number of CUDA blocks that can run simultaneously on the GPU.
template<unsigned int Arch, int SignalLength>
void convolution() {
    using namespace cufftdx;

    static constexpr unsigned int minimum_input_size_bytes   = (1 << 30); // At least one GB of data will be processed by FFTs.
    static constexpr unsigned int signal_length              = SignalLength;
    static constexpr unsigned int fft_size                   = example::closest_power_of_2(signal_length);
    static constexpr unsigned int kernel_runs                = 10;
    static constexpr unsigned int warm_up_runs               = 1;

    // To determine the total input length (number of fft batches to run), the maximum number of
    // simultanously running cuFFTDx CUDA blocks is calculated.

    // Declaration of cuFFTDx padded FFT
    using padded_fft_incomplete = decltype(Block() + Size<fft_size>() + Precision<float>() + SM<Arch>());
    using padded_real_options = RealFFTOptions<complex_layout::natural, real_mode::folded>;
    using padded_fft_base       = decltype(padded_fft_incomplete() + Type<fft_type::r2c>() + padded_real_options());
    using padded_ifft_base      = decltype(padded_fft_incomplete() + Type<fft_type::c2r>() + padded_real_options());

    static constexpr unsigned int padded_elements_per_thread = padded_fft_base::elements_per_thread;
    static constexpr unsigned int padded_ffts_per_block      = padded_fft_base::suggested_ffts_per_block;

    using padded_fft          = decltype(padded_fft_base() + ElementsPerThread<padded_elements_per_thread>() + FFTsPerBlock<padded_ffts_per_block>());
    using padded_ifft         = decltype(padded_ifft_base() + ElementsPerThread<padded_elements_per_thread>() + FFTsPerBlock<padded_ffts_per_block>());

    // Declaration of non-padded FFT
    using fft_incomplete = decltype(Block() + Size<signal_length>() + Precision<float>() + SM<Arch>());
    using real_options = RealFFTOptions<complex_layout::natural, real_mode::normal>;
    using fft_base       = decltype(fft_incomplete() + Type<fft_type::r2c>() + real_options());
    using ifft_base      = decltype(fft_incomplete() + Type<fft_type::c2r>() + real_options());

    static constexpr unsigned int elements_per_thread = fft_base::elements_per_thread;
    static constexpr unsigned int ffts_per_block      = fft_base::suggested_ffts_per_block;

    using fft          = decltype(fft_base() + ElementsPerThread<elements_per_thread>() + FFTsPerBlock<ffts_per_block>());
    using ifft         = decltype(ifft_base() + ElementsPerThread<elements_per_thread>() + FFTsPerBlock<ffts_per_block>());

    using complex_type = typename fft::value_type;
    using scalar_type  = typename complex_type::value_type;

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        convolution_kernel<signal_length, padded_fft, padded_ifft>, cudaFuncAttributeMaxDynamicSharedMemorySize, padded_fft::shared_memory_size));
    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        convolution_kernel<signal_length, fft, ifft>, cudaFuncAttributeMaxDynamicSharedMemorySize, fft::shared_memory_size));

    // Get maximum number of running CUDA blocks per multiprocessor
    int bpm_padded = 0, bpm = 0;
    CUDA_CHECK_AND_EXIT(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpm_padded,
                                                      convolution_kernel<signal_length, padded_fft, padded_ifft>,
                                                      padded_fft::block_dim.x * padded_fft::block_dim.y * padded_fft::block_dim.z,
                                                      padded_fft::shared_memory_size));
    CUDA_CHECK_AND_EXIT(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpm,
                                                      convolution_kernel<signal_length, fft, ifft>,
                                                      fft::block_dim.x * fft::block_dim.y * fft::block_dim.z,
                                                      fft::shared_memory_size));

    // Make sure that full wavefronts will be executed for both kernels
    int blocks_per_multiprocessor = bpm_padded * bpm;

    // Get maximum number of CUDA blocks running on all multiprocessors
    const unsigned int device_blocks = blocks_per_multiprocessor * example::get_multiprocessor_count();

    // Input size in bytes if device_blocks CUDA blocks were run.
    const unsigned int data_size_device_blocks_bytes = device_blocks * ffts_per_block * padded_ffts_per_block * signal_length * sizeof(complex_type);

    // cuda_blocks = minimal number of CUDA blocks to run, such that:
    //   - cuda_blocks is divisible by device_blocks,
    //   - total input size is not less than minimum_input_size_bytes.
    // executed_blocks_multiplyer = cuda_blocks / device_blocks
    const unsigned int executed_blocks_multiplyer =
        (minimum_input_size_bytes + data_size_device_blocks_bytes - 1) / data_size_device_blocks_bytes;
    const unsigned int cuda_blocks = device_blocks * executed_blocks_multiplyer;
    const unsigned int cuda_blocks_padded  = cuda_blocks * ffts_per_block;
    const unsigned int cuda_blocks_non_padded  = cuda_blocks * padded_ffts_per_block;
    const unsigned int input_length = ffts_per_block * padded_ffts_per_block * cuda_blocks * signal_length;

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    // Host data
    std::vector<scalar_type> input =
        example::get_random_real_data<scalar_type>(input_length, -1, 1);

    // Device data
    scalar_type* input_data, *output;
    auto          input_size_bytes = input.size() * sizeof(scalar_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&input_data, input_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output, input_size_bytes));

    complex_type* work_buffer;
    auto          work_size_bytes = (ffts_per_block * padded_ffts_per_block * cuda_blocks * (signal_length / 2 + 1)) * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&work_buffer, work_size_bytes));

    // Copy host to device
    CUDA_CHECK_AND_EXIT(cudaMemcpy(input_data, input.data(), input_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());



    // Number of batches to use for correctness verification
    // Entire input is:
    // ffts_per_block * padded_ffts_per_block * cuda blocks
    const int correctness_batches = ffts_per_block * padded_ffts_per_block * cuda_blocks;

    // Measure performance and obtain results for correctness verification
    auto res_cufft   = measure_cufft<signal_length, fft>(
        kernel_runs, warm_up_runs, correctness_batches, cuda_blocks * ffts_per_block * padded_ffts_per_block, (cufftReal*)input_data, output, (cufftComplex*)work_buffer, stream);
    auto res_cufftdx = measure_cufftdx<signal_length, fft, ifft>(kernel_runs, warm_up_runs, correctness_batches, cuda_blocks_non_padded, input_data, output, stream);
    auto res_cufftdx_padded = measure_cufftdx<signal_length, padded_fft, padded_ifft>(kernel_runs, warm_up_runs, correctness_batches, cuda_blocks_padded, input_data, output, stream);

    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(input_data));
    CUDA_CHECK_AND_EXIT(cudaFree(output));
    CUDA_CHECK_AND_EXIT(cudaFree(work_buffer));

    // Report results.
    auto report_time_and_performance = [&](std::string name, double time) -> void {
        double gflops = 1.0 * kernel_runs * ffts_per_block * padded_ffts_per_block * cuda_blocks * 5.0 * signal_length *
                        (std::log(signal_length) / std::log(2)) / time / 1000000.0;

        std::cout << std::endl;
        std::cout << name << std::endl;
        std::cout << "Avg Time [ms_n]: " << time / kernel_runs << std::endl;
        std::cout << "Time (all) [ms_n]: " << time << std::endl;
        std::cout << "Performance [GFLOPS]: " << gflops << std::endl;
    };
    bool success = true;

    auto verify_correctness = [&success](const auto& result_array, const auto& result_name, const auto& reference_array, const auto reference_name) {
        auto fft_error = example::fft_signal_error::calculate_for_real_values(
            result_array,
            reference_array);

        std::cout << "Correctness results for: " << result_name << " using " << reference_name << " as reference\n";
        std::cout << "L2 relative error: " << fft_error.l2_relative_error << "\n";
        std::cout << "Peak error (index: " << fft_error.peak_error_index << "): " << fft_error.peak_error << "\n";

        if (success) {
            success = (fft_error.l2_relative_error < 0.01);
        }
    };

    verify_correctness(res_cufftdx.output, "cuFFTDx Bluestein", res_cufft.output, "cuFFT");
    verify_correctness(res_cufftdx_padded.output, "cuFFTDx Padded", res_cufft.output, "cuFFT");
    verify_correctness(res_cufftdx_padded.output, "cuFFTDx Padded", res_cufftdx.output, "cuFFTDx Bluestein");

    std::cout << "FFT size: " << signal_length << std::endl;
    std::cout << "FFTs run: " << ffts_per_block * padded_ffts_per_block * cuda_blocks << std::endl;
    std::cout << "Correctness tested batches: " << correctness_batches << std::endl;

    report_time_and_performance("cuFFTDx", res_cufftdx.avg_time_in_ms);
    report_time_and_performance("cuFFTDx Padded", res_cufftdx_padded.avg_time_in_ms);
    report_time_and_performance("cuFFT", res_cufft.avg_time_in_ms);

    if (success) {
        std::cout << "Success!" << std::endl;
    }
}

template<unsigned int Arch>
struct convolution_functor {
    void operator()() {
        std::cout << "Power of 2:" << std::endl;
        convolution<Arch, 2048>();
        std::cout << "Power of 3:" << std::endl;
        convolution<Arch, 6561>();
        std::cout << "Power of 12:" << std::endl;
        convolution<Arch, 1728>();
        std::cout << "Odd length:" << std::endl;
        convolution<Arch, 2003>();
        std::cout << "Bluestein:" << std::endl;
        convolution<Arch, 3668>();
    }
};

int main(int, char**) {
    return example::sm_runner<convolution_functor>();
}
