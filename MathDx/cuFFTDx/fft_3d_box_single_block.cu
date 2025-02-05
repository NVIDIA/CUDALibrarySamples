#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>

#include "block_io.hpp"
#include "common.hpp"
#include "random.hpp"

// #define CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_3D
// #define CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_3D_SIMPLE_IO
inline constexpr unsigned int cufftdx_example_warm_up_runs = 5;
inline constexpr unsigned int cufftdx_example_performance_runs = 50;

template<unsigned int MaxThreadsPerBlock,
         class FFTX,
         class FFTY,
         class FFTZ,
         class ComplexType                = typename FFTX::value_type,
         unsigned int RequiredStorageSize = std::max({FFTX::storage_size, FFTY::storage_size, FFTZ::storage_size})>
__launch_bounds__(MaxThreadsPerBlock) __global__
    void cufftdx_3d_fft_single_block_kernel(const ComplexType* input, ComplexType* output) {
    using complex_type                       = ComplexType;
    static constexpr unsigned int fft_size_x = cufftdx::size_of<FFTX>::value;
    static constexpr unsigned int fft_size_y = cufftdx::size_of<FFTY>::value;
    static constexpr unsigned int fft_size_z = cufftdx::size_of<FFTZ>::value;

    // Local array for thread
    constexpr auto required_storage_size = RequiredStorageSize;
    complex_type   thread_data[required_storage_size];

    // Shared memory use for exchanging data between threads
    extern __shared__ __align__(alignof(float4)) complex_type shared_memory[];

    // Load data from global memory to registers.
    static constexpr unsigned int stride_x = fft_size_y * fft_size_z;
    unsigned int                  index    = (threadIdx.x + threadIdx.y * fft_size_y);
    if (threadIdx.x < fft_size_y && threadIdx.y < fft_size_z) {
        for (size_t i = 0; i < FFTX::elements_per_thread; i++) {
            thread_data[i] = input[index];
            index += stride_x;
        }

        // Execute FFT in X dimension
        FFTX().execute(thread_data);

        // Exchange/transpose via shared memory
        index = (threadIdx.x + threadIdx.y * fft_size_y);
        for (size_t i = 0; i < FFTX::elements_per_thread; i++) {
            shared_memory[index] = thread_data[i];
            index += stride_x;
        }
    }

    __syncthreads();
    static constexpr unsigned int stride_y = fft_size_z;
    index                                  = threadIdx.x + threadIdx.y * fft_size_y * fft_size_z;
    if (threadIdx.x < fft_size_z && threadIdx.y < fft_size_x) {
        for (size_t i = 0; i < FFTY::elements_per_thread; i++) {
            thread_data[i] = shared_memory[index];
            index += stride_y;
        }

        // Execute FFT in Y dimension
        FFTY().execute(thread_data);

        // Exchange/transpose via shared memory
        index = threadIdx.x + threadIdx.y * fft_size_y * fft_size_z;
        for (size_t i = 0; i < FFTY::elements_per_thread; i++) {
            shared_memory[index] = thread_data[i];
            index += stride_y;
        }
    }

    __syncthreads();
    if (threadIdx.x < fft_size_x && threadIdx.y < fft_size_y) {
        index = (threadIdx.x + threadIdx.y * fft_size_x) * fft_size_z;
        for (size_t i = 0; i < FFTZ::elements_per_thread; i++) {
            thread_data[i] = shared_memory[index];
            index += 1;
        }
        // Execute FFT in Z dimension
        FFTZ().execute(thread_data);
    }

    // Save results
#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_3D_SIMPLE_IO
    // Simple IO with poor global memory pattern:
    // Storing the data with stride=1 results in poor global memory
    // write pattern with little or none coalescing
    if (threadIdx.x < fft_size_x && threadIdx.y < fft_size_y) {
        index = (threadIdx.x + threadIdx.y * fft_size_x) * fft_size_z;
        for (size_t i = 0; i < FFTZ::elements_per_thread; i++) {
            output[index] = thread_data[i];
            index += 1;
        }
    }
#else
    // Shared memory IO:
    // Exchanging data via shared memory results in a much better global
    // memory patter with good coalescing
    if (threadIdx.x < fft_size_x && threadIdx.y < fft_size_y) {
        index = (threadIdx.x + threadIdx.y * fft_size_x) * fft_size_z;
        for (size_t i = 0; i < FFTZ::elements_per_thread; i++) {
            shared_memory[index] = thread_data[i];
            index += 1;
        }
    }
    __syncthreads();
    if (threadIdx.x < fft_size_y && threadIdx.y < fft_size_z) {
        index = (threadIdx.x + threadIdx.y * fft_size_y);
        for (size_t i = 0; i < FFTX::elements_per_thread; i++) {
            thread_data[i] = shared_memory[index];
            index += stride_x;
        }

        index = (threadIdx.x + threadIdx.y * fft_size_y);
        for (size_t i = 0; i < FFTX::elements_per_thread; i++) {
            output[index] = thread_data[i];
            index += stride_x;
        }
    }
#endif
}

example::fft_results<float2> cufft_3d_fft(unsigned int  fft_size_x,
                                          unsigned int  fft_size_y,
                                          unsigned int  fft_size_z,
                                          cufftComplex* input,
                                          cufftComplex* output,
                                          cudaStream_t  stream) {
    // Create cuFFT plan
    cufftHandle plan;
    CUFFT_CHECK_AND_EXIT(cufftPlan3d(&plan, fft_size_x, fft_size_y, fft_size_z, CUFFT_C2C));
    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan, stream));

    // Execute cuFFT
    auto cufft_execution = [&](cudaStream_t /* stream */) {
        CUFFT_CHECK_AND_EXIT(cufftExecC2C(plan, input, output, CUFFT_FORWARD))
    };

    // Correctness run
    cufft_execution(stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    // Copy results to host
    const size_t        flat_fft_size       = fft_size_x * fft_size_y * fft_size_z;
    const size_t        flat_fft_size_bytes = flat_fft_size * sizeof(float2);
    std::vector<float2> output_host(flat_fft_size, {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()});
    CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), output, flat_fft_size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Performance measurements
    auto time = example::measure_execution_ms(
        cufft_execution,
        cufftdx_example_warm_up_runs,
        cufftdx_example_performance_runs,
        stream);

    // Clean-up
    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan));

    // Return results
    return example::fft_results<float2> {output_host, (time / cufftdx_example_performance_runs)};
}

template<unsigned int FFTSizeX, unsigned int FFTSizeY, unsigned int FFTSizeZ>
example::fft_results<float2> cufftdx_3d_fft_single_block(float2* input, float2* output, cudaStream_t stream) {
    using namespace cufftdx;

    static constexpr unsigned int fft_size_x = FFTSizeX;
    static constexpr unsigned int fft_size_y = FFTSizeY;
    static constexpr unsigned int fft_size_z = FFTSizeZ;

    using fft_base =
        decltype(Thread() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() + Precision<float>());
    using fft_x        = decltype(fft_base() + Size<fft_size_x>());
    using fft_y        = decltype(fft_base() + Size<fft_size_y>());
    using fft_z        = decltype(fft_base() + Size<fft_size_z>());
    using complex_type = typename fft_x::value_type;

    constexpr unsigned int max_dim               = std::max({fft_size_x, fft_size_y, fft_size_z});
    constexpr dim3         block_dim             = {max_dim, max_dim, 1};
    constexpr unsigned int max_threads_per_block = block_dim.x * block_dim.y * block_dim.z;
    const size_t           shared_memory_size    = (fft_size_x * fft_size_y * fft_size_z) * sizeof(complex_type);

    const auto kernel = cufftdx_3d_fft_single_block_kernel<max_threads_per_block, fft_x, fft_y, fft_z, complex_type>;

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

    complex_type* cufftdx_input  = reinterpret_cast<complex_type*>(input);
    complex_type* cufftdx_output = reinterpret_cast<complex_type*>(output);
    auto fft_3d_execution = [&](cudaStream_t stream) {
        kernel<<<1, block_dim, shared_memory_size, stream>>>(cufftdx_input, cufftdx_output);
        CUDA_CHECK_AND_EXIT(cudaGetLastError());
    };

    // Correctness run
    fft_3d_execution(stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    // Copy results to host
    static constexpr size_t flat_fft_size       = fft_size_x * fft_size_y * fft_size_z;
    static constexpr size_t flat_fft_size_bytes = flat_fft_size * sizeof(float2);
    std::vector<float2> output_host(flat_fft_size, {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()});
    CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), cufftdx_output, flat_fft_size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Performance measurements
    auto time = example::measure_execution_ms(
        [&](cudaStream_t stream) {
            fft_3d_execution(stream);
        },
        cufftdx_example_warm_up_runs,
        cufftdx_example_performance_runs,
        stream);

    // Return results
    return example::fft_results<float2>{ output_host, (time / cufftdx_example_performance_runs) };
}

// Example showing how to perform a small 3D FFT in a single CUDA block using cuFFTDx thread execution.
//
// Notes:
// * This example shows how to use cuFFTDx to runa multi-dimensional FFT. Final performance will vary depending on the
// FFT definitions (precision, size, type, ept, fpb) and other user customizations.
// * Best possible performance requires adapting parameters in the sample to particular set of parameters and code customizations.
// * Only FP32 was tested for this example, other types might require adjustments.
int main(int, char**) {
    // 3D FFT
    static constexpr unsigned int fft_size_x = 16;
    static constexpr unsigned int fft_size_y = 15;
    static constexpr unsigned int fft_size_z = 14;

    // Generate random input data on host
    const unsigned int flat_fft_size = fft_size_x * fft_size_y * fft_size_z;
#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_3D
    std::vector<float2> host_input(flat_fft_size);
    for (size_t i = 0; i < flat_fft_size; i++) {
        float sign      = (i % 3 == 0) ? -1.0f : 1.0f;
        host_input[i].x = sign * static_cast<float>(i) / 100;
        host_input[i].y = sign * static_cast<float>(i) / 100;
    }
#else
    auto host_input = example::get_random_complex_data<float>(flat_fft_size, -1, 1);
#endif

    // Allocate managed memory for device input/output
    // float2 has the same size and alignment as cuFFTDx fp32 complex type cufftdx::complex<float> and cufftComplex
    float2*    input;
    float2*    output;
    const auto flat_fft_size_bytes = flat_fft_size * sizeof(float2);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&input, flat_fft_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output, flat_fft_size_bytes));

    // Copy input to the device
    CUDA_CHECK_AND_EXIT(cudaMemcpy(input, host_input.data(), flat_fft_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    // Run cuFFT
    auto cufft_results = cufft_3d_fft(fft_size_x, fft_size_y, fft_size_z, input, output, stream);

    // Run cuFFTDx
    auto cufftdx_results = cufftdx_3d_fft_single_block<fft_size_x, fft_size_y, fft_size_z>(input, output, stream);

    // Clean-up
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(input));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    // Check if cuFFTDx results are correct
    auto fft_error = example::fft_signal_error::calculate_for_complex_values(cufftdx_results.output, cufft_results.output);

    std::cout << "FFT: (" << fft_size_x << ", " << fft_size_y << ", " << fft_size_z <<")\n";

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_3D
    std::cout << "cuFFT, cuFFTDx\n";
    for (size_t i = 0; i < 8; i++) {
        std::cout << i << ": ";
        std::cout << "(" << cufft_results.output[i].x << ", " << cufft_results.output[i].y << ")";
        std::cout << ", ";
        std::cout << "(" << cufftdx_results.output[i].x << ", " << cufftdx_results.output[i].y << ")";
        std::cout << "\n";
    }
#endif

    bool success = fft_error.l2_relative_error < 0.001;
    std::cout << "Correctness results:\n";
    std::cout << "L2 error: " << fft_error.l2_relative_error << "\n";
    std::cout << "Peak error (index: " << fft_error.peak_error_index << "): " << fft_error.peak_error << "\n";

    // Print performance results
    if(success) {
        std::cout << "\nPerformance results:\n";
        std::cout << std::setw(10) << "cuFFTDx: " << cufftdx_results.avg_time_in_ms << " [ms]\n";
        std::cout << std::setw(10) << "cuFFT: " << cufft_results.avg_time_in_ms << " [ms]\n";
    }

    if (success) {
        std::cout << "Success\n";
        return 0;
    } else {
        std::cout << "Failure\n";
        return 1;
    }
}
