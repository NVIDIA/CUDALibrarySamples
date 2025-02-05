#include <iostream>
#include <vector>
#include <algorithm>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>

#include "block_io.hpp"
#include "mixed_io.hpp"
#include "common.hpp"
#include "random.hpp"

template<typename FFT, typename Kernel>
auto get_max_blocks_per_multiprocessor(unsigned int shared_size, Kernel kernel) {
    // Get maximum number of running CUDA blocks per multiprocessor
    int bpm;
    CUDA_CHECK_AND_EXIT(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpm,
                                                      kernel,
                                                      FFT::block_dim.x * FFT::block_dim.y * FFT::block_dim.z,
                                                      shared_size));
    return bpm;
}

template<class FFT>
__launch_bounds__(FFT::max_threads_per_block)
    __global__ void simple_fft_1d_kernel(typename FFT::value_type* input,
                                         typename FFT::value_type* output) {
    using complex_type = typename FFT::value_type;

    // Local array for thread

    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    example::io<FFT>::load(input, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results
    example::io<FFT>::store(thread_data, output, local_fft_id);
}

template<class FFT>
__launch_bounds__(FFT::max_threads_per_block)
    __global__ void simple_fft_1d_shared_kernel(typename FFT::value_type* input,
                                                typename FFT::value_type* output) {
    using complex_type = typename FFT::value_type;

    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];

    // Since each block has a defined number of FFTs to compute, in here we offset global memory start address
    // as to get a pointer to batch group assigned to the current block
    //
    // sub_batch_address = global_start_address + size(single FFT) * (batches per block) * (block index)
    auto this_block_input  = input + cufftdx::size_of<FFT>::value * (FFT::ffts_per_block / FFT::implicit_type_batching) * blockIdx.x;
    auto this_block_output = output + cufftdx::size_of<FFT>::value * (FFT::ffts_per_block / FFT::implicit_type_batching) * blockIdx.x;

    example::io<FFT>::load_to_smem(this_block_input, reinterpret_cast<unsigned char*>(shared_mem));

    FFT().execute(reinterpret_cast<void*>(shared_mem));

    example::io<FFT>::store_from_smem(reinterpret_cast<unsigned char*>(shared_mem), this_block_output);
}

template<typename FFT, typename InputOutputType>
__launch_bounds__(FFT::max_threads_per_block)
    __global__ void mixed_fft_1d_shared_kernel(const InputOutputType* input,
                                               InputOutputType*       output) {
    extern __shared__ __align__(alignof(float4)) typename FFT::value_type shared_mem[];

    // Since each block has a defined number of FFTs to compute, in here we offset global memory start address
    // as to get a pointer to batch group assigned to the current block
    //
    // sub_batch_address = global_start_address + size(single FFT) * (batches per block) * (block index)
    auto this_block_input  = input + cufftdx::size_of<FFT>::value * (FFT::ffts_per_block / FFT::implicit_type_batching) * blockIdx.x;
    auto this_block_output = output + cufftdx::size_of<FFT>::value * (FFT::ffts_per_block / FFT::implicit_type_batching) * blockIdx.x;

    // Load data from global memory to shared memory
    // This load overload will perform a simple conversion
    // from InputOutputPrecision into ComputePrecision
    example::io_mixed<FFT>::load(this_block_input, shared_mem);

    FFT().execute(shared_mem);

    // Save results to global memory
    // This store overload will perform a simple conversion
    // from ComputePrecision into InputOutputPrecision
    example::io_mixed<FFT>::store(shared_mem, this_block_output);
}

template<typename FFT, typename InputOutputType>
__launch_bounds__(FFT::max_threads_per_block)
    __global__ void mixed_fft_1d_kernel(const InputOutputType* input,
                                        InputOutputType*       output) {
    using complex_type = typename FFT::value_type;
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;

    // Load data from global memory to registers
    // This load overload will perform a simple conversion
    // from InputOutputPrecision into ComputePrecision
    example::io_mixed<FFT>::load(input, thread_data, local_fft_id);

    // Execute FFT in full precision
    FFT().execute(thread_data, shared_mem);

    // Save results to global memory
    // This store overload will perform a simple conversion
    // from ComputePrecision into InputOutputPrecision
    example::io_mixed<FFT>::store(thread_data, output, local_fft_id);
}

template<typename FFT>
example::fft_results<typename FFT::value_type> cufft_fft_1d(unsigned int cuda_blocks, cudaStream_t stream, unsigned int warm_up_runs, unsigned int kernel_runs, typename FFT::value_type* device_input, typename FFT::value_type* device_output) {
    using cufft_complex_type = typename example::make_cufft_compatible<typename FFT::value_type>::type;

    auto  size         = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto* cufft_input  = reinterpret_cast<cufft_complex_type*>(device_input);
    auto* cufft_output = reinterpret_cast<cufft_complex_type*>(device_output);

    // Create cuFFT plan
    cufftHandle plan;
    CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan, cufftdx::size_of<FFT>::value, std::is_same_v<cufft_complex_type, cufftComplex> ? CUFFT_C2C : CUFFT_Z2Z, cuda_blocks * FFT::ffts_per_block));

    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan, stream));

    if constexpr (std::is_same_v<cufft_complex_type, cufftComplex>) {
        CUFFT_CHECK_AND_EXIT(cufftExecC2C(plan, cufft_input, cufft_output, CUFFT_FORWARD));
    } else if constexpr (std::is_same_v<cufft_complex_type, cufftDoubleComplex>) {
        CUFFT_CHECK_AND_EXIT(cufftExecZ2Z(plan, cufft_input, cufft_output, CUFFT_FORWARD));
    }

    // Get initial results for correctness check
    std::vector<typename FFT::value_type> host_output(size);
    cudaMemcpy(reinterpret_cast<void*>(host_output.data()), reinterpret_cast<void*>(cufft_output), size * sizeof(typename FFT::value_type), cudaMemcpyDeviceToHost);


    float time_cufft = example::measure_execution_ms(
        [&]([[maybe_unused]] cudaStream_t stream) {
            if constexpr (std::is_same_v<cufft_complex_type, cufftComplex>) {
                CUFFT_CHECK_AND_EXIT(cufftExecC2C(plan, cufft_input, cufft_output, CUFFT_FORWARD));
            } else if constexpr (std::is_same_v<cufft_complex_type, cufftDoubleComplex>) {
                CUFFT_CHECK_AND_EXIT(cufftExecZ2Z(plan, cufft_input, cufft_output, CUFFT_FORWARD));
            }
        },
        warm_up_runs,
        kernel_runs,
        stream);

    // Clean-up
    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan));

    return {host_output, time_cufft / kernel_runs};
}


template<typename FFT, typename InputOutputType, typename FFTKernel>
example::fft_results<typename FFT::value_type> cufftdx_result_launcher(FFTKernel fk, unsigned int shared_size, unsigned int cuda_blocks, cudaStream_t stream, unsigned int warm_up_runs, unsigned int kernel_runs, InputOutputType* device_input, InputOutputType* device_output) {
    using complex_compute_type = typename FFT::value_type;
    constexpr bool is_mixed    = !std::is_same_v<typename FFT::value_type, InputOutputType>;

    // Here size is number of elements per block need to be loaded
    auto size = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        fk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_size));

    // Execute kernel for correctness check
    // Invokes cuFFTDx kernel passed as an argument with FFT::block_dim threads in CUDA block
    fk<<<cuda_blocks, FFT::block_dim, shared_size, stream>>>(device_input, device_output);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Create host result vector to be returned later
    std::vector<complex_compute_type> host_output(size);

    if constexpr (is_mixed) {
        // If the data is of mixed precision first copy InputOutputPrecision data into host memory
        auto                         size_io_bytes = size * sizeof(InputOutputType);
        std::vector<InputOutputType> host_mixed_output(size);
        CUDA_CHECK_AND_EXIT(cudaMemcpy(reinterpret_cast<void*>(host_mixed_output.data()), reinterpret_cast<void*>(device_output), size_io_bytes, cudaMemcpyDeviceToHost));

        // And then transform it to ComputePrecision and copy to the return vector created above
        std::transform(begin(host_mixed_output), end(host_mixed_output), begin(host_output), [](const auto& v) {
            return example::convert<complex_compute_type>(v);
        });
    } else {
        // If the data is of constant precision perform a simple copy into host memory
        auto size_compute_bytes = size * sizeof(complex_compute_type);
        CUDA_CHECK_AND_EXIT(cudaMemcpy(reinterpret_cast<void*>(host_output.data()), reinterpret_cast<void*>(device_output), size_compute_bytes, cudaMemcpyDeviceToHost));
    }

    // Run performance tests
    float time = example::measure_execution_ms(
        [&](cudaStream_t stream) {
            // There are (ffts_per_block * fft_size * cuda_blocks) elements
            fk<<<cuda_blocks, FFT::block_dim, shared_size, stream>>>(device_input, device_output);
        },
        warm_up_runs,
        kernel_runs,
        stream);

    return {host_output, time / kernel_runs};
}


// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// Data is generated on host, copied to device buffer, and then results are copied back to host.
// The example calculates FFT using: cuFFT (for reference), cuFFTDx, and cuFFTDx with different
// precisions for data and computations (aka mixed precision). Additionally, both register and
// shared APIs of cuFFTDx are tested.
// At the end the performance and accuracy results of each solution are presented.
template<unsigned int Arch, unsigned int fft_size = 512>
void mixed_precision_fft_1d() {
    using namespace cufftdx;

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    // Choose mixed precision options
    // Input and Output type
    using io_precision = float;
    // Computation type
    using compute_precision = double;
    // Initialization happens to make sure this mixed configuration is supported
    using mixed = example::mixed_precision<compute_precision, io_precision>;

    // Performance testing arguments
    static constexpr unsigned int minimum_input_size_bytes = (1 << 30); // At least one GB of data will be processed by FFTs.
    static constexpr unsigned int warm_up_runs             = 1;
    static constexpr unsigned int kernel_runs              = 10;

    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    using fft_base = decltype(Block() + Size<fft_size>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                              Precision<compute_precision>() + SM<Arch>());
    // cuFFTDx performance suggestions are used, as this example also tests execution times
    using FFT = decltype(fft_base() + ElementsPerThread<fft_base::elements_per_thread>() + FFTsPerBlock<fft_base::ffts_per_block>());

    // Data types which will be used, based on previously choisen
    // compute and input/output values
    using complex_io_type      = typename mixed::complex_io_type;
    using complex_compute_type = typename FFT::value_type;

    // Shared memory byte count necessary per block for shared API of the above defined FFT computation
    constexpr auto max_smem = std::max<unsigned int>(FFT::shared_memory_size, FFT::ffts_per_block * fft_size * sizeof(complex_compute_type));
    // Choose the configuration such that the entire GPU will be saturated
    // necessary for reliable performance testing
    const auto blocks_per_multiprocessor = std::max<int>({get_max_blocks_per_multiprocessor<FFT>(FFT::shared_memory_size, &mixed_fft_1d_kernel<FFT, complex_io_type>),
                                                          get_max_blocks_per_multiprocessor<FFT>(max_smem, &mixed_fft_1d_shared_kernel<FFT, complex_io_type>),
                                                          get_max_blocks_per_multiprocessor<FFT>(FFT::shared_memory_size, &simple_fft_1d_kernel<FFT>),
                                                          get_max_blocks_per_multiprocessor<FFT>(max_smem, &simple_fft_1d_shared_kernel<FFT>)});
    if (blocks_per_multiprocessor < 1 ) {
        std::cout << "============\nNot enough resources for FFT of size: " << fft_size << "\n";
        return; 
    }
    // Get maximum number of CUDA blocks running on all multiprocessors
    const unsigned int device_blocks = blocks_per_multiprocessor * example::get_multiprocessor_count();
    // Input size in bytes if device_blocks CUDA blocks were run.
    const unsigned int data_size_device_blocks_bytes = device_blocks * FFT::ffts_per_block * fft_size * sizeof(complex_io_type);

    // cuda_blocks = minimal number of CUDA blocks to run, such that:
    //   - cuda_blocks is divisible by device_blocks,
    //   - total input size is not less than minimum_input_size_bytes.
    // executed_blocks_multiplyer = cuda_blocks / device_blocks
    const unsigned int executed_blocks_multiplyer =
        (minimum_input_size_bytes + data_size_device_blocks_bytes - 1) / data_size_device_blocks_bytes;
    const unsigned int cuda_blocks = device_blocks * executed_blocks_multiplyer;

    // Memory Allocation and copies

    // size is entire input length, not FFT size
    const auto single_fft_size               = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto       single_fft_size_io_bytes      = single_fft_size * sizeof(complex_io_type);
    auto       single_fft_size_compute_bytes = single_fft_size * sizeof(complex_compute_type);

    const auto size               = single_fft_size * cuda_blocks;
    auto       size_io_bytes      = size * sizeof(complex_io_type);
    auto       size_compute_bytes = size * sizeof(complex_compute_type);


    // Shared memory must fit input data and must be big enough to run FFT
    complex_io_type *     input_io, *output_io;
    complex_compute_type *input_compute, *output_compute;

    CUDA_CHECK_AND_EXIT(cudaMalloc(&input_io, size_io_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output_io, size_io_bytes));

    CUDA_CHECK_AND_EXIT(cudaMalloc(&input_compute, size_compute_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output_compute, size_compute_bytes));

    // Host data
    std::vector<complex_compute_type> single_fft_host_input =
        example::get_random_complex_data<compute_precision>(single_fft_size, -1, 1);
    std::vector<complex_io_type> single_fft_host_mixed_input(single_fft_size);

    std::transform(begin(single_fft_host_input), end(single_fft_host_input), begin(single_fft_host_mixed_input), [](const auto& v) {
        return example::convert<io_precision>(v);
    });

    // Copy to device
    for (unsigned int i = 0; i < cuda_blocks; ++i) {
        cudaMemcpy(reinterpret_cast<void*>(input_compute + i * single_fft_size), single_fft_host_input.data(), single_fft_size_compute_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(reinterpret_cast<void*>(input_io + i * single_fft_size), single_fft_host_mixed_input.data(), single_fft_size_io_bytes, cudaMemcpyHostToDevice);
    }

    // Perform reference cuFFT host API computation
    const auto cufft_results = cufft_fft_1d<FFT>(cuda_blocks, stream, warm_up_runs, kernel_runs, input_compute, output_compute);

    // Launch cuFFTDx kernels with both mixed and full precision
    const auto mixed_results = cufftdx_result_launcher<FFT>(
        mixed_fft_1d_kernel<FFT, complex_io_type>, FFT::shared_memory_size, cuda_blocks, stream, warm_up_runs, kernel_runs, input_io, output_io);
    const auto mixed_shared_results = cufftdx_result_launcher<FFT>(
        mixed_fft_1d_shared_kernel<FFT, complex_io_type>, max_smem, cuda_blocks, stream, warm_up_runs, kernel_runs, input_io, output_io);
    const auto simple_results = cufftdx_result_launcher<FFT>(
        simple_fft_1d_kernel<FFT>, FFT::shared_memory_size, cuda_blocks, stream, warm_up_runs, kernel_runs, input_compute, output_compute);
    const auto simple_shared_results = cufftdx_result_launcher<FFT>(
        simple_fft_1d_shared_kernel<FFT>, max_smem, cuda_blocks, stream, warm_up_runs, kernel_runs, input_compute, output_compute);

    // Verify Correctness
    bool success          = true;
    using result_vector_t = std::vector<complex_compute_type>;
    std::cout << "============\nCorrectness results:\n";

    auto verify_correctness = [&success](const auto& result_array, const auto& result_name, const auto& reference_array, const auto reference_name) {
        auto fft_error = example::fft_signal_error::calculate_for_complex_values(
            result_array,
            reference_array);

        std::cout << "Correctness results for: " << result_name << " using " << reference_name << " as reference\n";
        std::cout << "L2 relative error: " << fft_error.l2_relative_error << "\n";
        std::cout << "Peak error (index: " << fft_error.peak_error_index << "): " << fft_error.peak_error << "\n";
        if (success) {
            success = (fft_error.l2_relative_error < 0.001);
        }
    };

    verify_correctness(mixed_results.output, "cuFFTDx Mixed Register API", cufft_results.output, "cuFFT");
    verify_correctness(mixed_shared_results.output, "cuFFTDx Mixed Shared API", cufft_results.output, "cuFFT");
    verify_correctness(mixed_results.output, "cuFFTDx Mixed Precision", simple_results.output, "cuFFTDx Full Precision");
    verify_correctness(mixed_shared_results.output, "cuFFTDx Mixed Precision Shared", simple_shared_results.output, "cuFFTDx Full Precision Shared");

    // Report performance results
    std::cout << "============\nPerformance results:\n";

    auto report_time_and_performance = [&](std::string name, float avg_time) -> void {
        double gflops = 1.0 * kernel_runs * FFT::ffts_per_block * cuda_blocks * 5.0 * fft_size *
                        (std::log(fft_size) / std::log(2)) / (avg_time * kernel_runs) / 1000000.0;

        std::cout << std::endl;
        std::cout << name << std::endl;
        std::cout << "Avg Time [ms_n]: " << avg_time << std::endl;
        std::cout << "Time (all) [ms_n]: " << kernel_runs * avg_time << std::endl;
        std::cout << "Performance [GFLOPS]: " << gflops << std::endl;
    };


    std::cout << "FFT size: " << fft_size << std::endl;
    std::cout << "FFTs run: " << FFT::ffts_per_block * cuda_blocks << std::endl;
    std::cout << "FFTs elements per thread: " << FFT::elements_per_thread << std::endl;
    std::cout << "FFTs per block: " << FFT::ffts_per_block << std::endl;
    std::cout << "CUDA blocks: " << cuda_blocks << std::endl;
    std::cout << "Blocks per multiprocessor: " << blocks_per_multiprocessor << std::endl;

    report_time_and_performance("cuFFT", cufft_results.avg_time_in_ms);
    report_time_and_performance("cuFFTDx Mixed Precision", mixed_results.avg_time_in_ms);
    report_time_and_performance("cuFFTDx Mixed Precision Shared", mixed_shared_results.avg_time_in_ms);
    report_time_and_performance("cuFFTDx Full Precision", simple_results.avg_time_in_ms);
    report_time_and_performance("cuFFTDx Full Precision Shared", simple_shared_results.avg_time_in_ms);

    if (success) {
        std::cout << "Success" << std::endl;
    }

    CUDA_CHECK_AND_EXIT(cudaFree(input_io));
    CUDA_CHECK_AND_EXIT(cudaFree(input_compute));
    CUDA_CHECK_AND_EXIT(cudaFree(output_io));
    CUDA_CHECK_AND_EXIT(cudaFree(output_compute));
}

template<unsigned int Arch>
struct mixed_precision_fft_1d_functor {
    void operator()() { 
        
        mixed_precision_fft_1d<Arch,128>();
        mixed_precision_fft_1d<Arch,512>();
        mixed_precision_fft_1d<Arch,2048>();
        mixed_precision_fft_1d<Arch,4096>();
        mixed_precision_fft_1d<Arch,8192>();
    }
};

int main(int, char**) {
    return example::sm_runner<mixed_precision_fft_1d_functor>();
}
