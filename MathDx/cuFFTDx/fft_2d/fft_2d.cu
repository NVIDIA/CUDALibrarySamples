#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>

#include "block_io.hpp"
#include "block_io_strided.hpp"
#include "common.hpp"
#include "random.hpp"

// #define CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_2D
inline constexpr unsigned int cufftdx_example_warm_up_runs = 5;
inline constexpr unsigned int cufftdx_example_performance_runs = 20;

template<class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void fft_2d_kernel_y(const ComplexType* input, ComplexType* output, typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    example::io<FFT>::load(input, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Save results
    example::io<FFT>::store(thread_data, output, local_fft_id);
}

template<class FFT, unsigned int Stride, bool UseSharedMemoryStridedIO, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void fft_2d_kernel_x(const ComplexType* input, ComplexType* output, typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;

    extern __shared__ complex_type shared_mem[];

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    if constexpr (UseSharedMemoryStridedIO) {
        example::io_strided<FFT>::load_strided<Stride>(input, thread_data, shared_mem, local_fft_id);
    } else {
        example::io_strided<FFT>::load_strided<Stride>(input, thread_data, local_fft_id);
    }

    // Execute FFT
    FFT().execute(thread_data, shared_mem, workspace);

    // Save results
    if constexpr (UseSharedMemoryStridedIO) {
        example::io_strided<FFT>::store_strided<Stride>(thread_data, shared_mem, output, local_fft_id);
    } else {
        example::io_strided<FFT>::store_strided<Stride>(thread_data, output, local_fft_id);
    }
}

template<class T>
example::fft_results<T> cufft_fft_2d(unsigned int fft_size_x, unsigned int fft_size_y, T* input, T* output, cudaStream_t stream) {
    using complex_type = cufftComplex;
    static_assert(sizeof(T) == sizeof(complex_type), "");
    static_assert(std::alignment_of_v<T> == std::alignment_of_v<complex_type>, "");

    complex_type* cufft_input  = reinterpret_cast<complex_type*>(input);
    complex_type* cufft_output = reinterpret_cast<complex_type*>(output);

    // Create cuFFT plan
    cufftHandle plan;
    CUFFT_CHECK_AND_EXIT(cufftPlan2d(&plan, fft_size_x, fft_size_y, CUFFT_C2C));
    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan, stream));

    // Execute cuFFT
    auto cufft_execution = [&](cudaStream_t /* stream */){
        CUFFT_CHECK_AND_EXIT(cufftExecC2C(plan, cufft_input, cufft_output, CUFFT_FORWARD));
    };

    // Correctness run
    cufft_execution(stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    // Copy results to host
    const size_t flat_fft_size       = fft_size_x * fft_size_y;
    const size_t flat_fft_size_bytes = flat_fft_size * sizeof(complex_type);
    std::vector<T> output_host(flat_fft_size, {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()});
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
    return example::fft_results<T>{ output_host, (time / cufftdx_example_performance_runs) };
}

template<class FFTX, class FFTY, bool UseSharedMemoryStridedIO, class T>
example::fft_results<T> cufftdx_fft_2d(T* input, T* output, cudaStream_t stream) {
    using complex_type                       = typename FFTX::value_type;
    static constexpr unsigned int fft_size_y = cufftdx::size_of<FFTY>::value;
    static constexpr unsigned int fft_size_x = cufftdx::size_of<FFTX>::value;

    // Checks that FFTX and FFTY are correctly defined
    static_assert(std::is_same_v<cufftdx::precision_of_t<FFTX>, cufftdx::precision_of_t<FFTY>>,
                  "FFTY and FFTX must have the same precision");
    static_assert(std::is_same_v<typename FFTX::value_type, typename FFTY::value_type>,
                  "FFTY and FFTX must operator on the same type");
    static_assert(sizeof(T) == sizeof(complex_type), "");
    static_assert(std::alignment_of_v<T> == std::alignment_of_v<complex_type>, "");
    // Checks below are not caused by any limitation in cuFFTDx, but rather in the example IO functions.
    static_assert((fft_size_x % FFTY::ffts_per_block == 0),
                  "FFTsPerBlock for FFTX must divide Y dimension as IO doesn't check if a batch is in range");

    complex_type* cufftdx_input  = reinterpret_cast<complex_type*>(input);
    complex_type* cufftdx_output = reinterpret_cast<complex_type*>(output);

    // Set shared memory requirements
    auto error_code = cudaFuncSetAttribute(
        fft_2d_kernel_y<FFTY, complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTY::shared_memory_size);
    CUDA_CHECK_AND_EXIT(error_code);

    // Shared memory IO for strided kernel may require more memory than FFTX::shared_memory_size.
    // Note: For some fft_size_x and depending on GPU architecture fft_x_shared_memory_smem_io may exceed max shared
    // memory and cudaFuncSetAttribute will fail.
    unsigned int fft_x_shared_memory_smem_io =
        std::max<unsigned>({FFTX::shared_memory_size, FFTX::ffts_per_block * fft_size_x * sizeof(complex_type)});
    unsigned int fft_x_shared_memory =
        UseSharedMemoryStridedIO ? fft_x_shared_memory_smem_io : FFTX::shared_memory_size;
    error_code = cudaFuncSetAttribute(fft_2d_kernel_x<FFTX, fft_size_y, UseSharedMemoryStridedIO, complex_type>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      fft_x_shared_memory);
    CUDA_CHECK_AND_EXIT(error_code);

    // Create workspaces for FFTs
    auto workspace_y = cufftdx::make_workspace<FFTY>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_x = cufftdx::make_workspace<FFTX>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);

    // Synchronize device before execution
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Define 2D FFT execution
    const auto grid_fft_size_y = ((fft_size_x + FFTY::ffts_per_block - 1) / FFTY::ffts_per_block);
    const auto grid_fft_size_x = ((fft_size_y + FFTX::ffts_per_block - 1) / FFTX::ffts_per_block);
    auto fft_2d_execution = [&](cudaStream_t stream) {
        fft_2d_kernel_y<FFTY, complex_type><<<grid_fft_size_y, FFTY::block_dim, FFTY::shared_memory_size, stream>>>(
            cufftdx_input, cufftdx_output, workspace_y);
        CUDA_CHECK_AND_EXIT(cudaGetLastError());
        fft_2d_kernel_x<FFTX, fft_size_y, UseSharedMemoryStridedIO, complex_type>
            <<<grid_fft_size_x, FFTX::block_dim, fft_x_shared_memory, stream>>>(
                cufftdx_output, cufftdx_output, workspace_x);
        CUDA_CHECK_AND_EXIT(cudaGetLastError());
    };

    // Correctness run
    fft_2d_execution(stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    // Copy results to host
    static constexpr size_t flat_fft_size       = fft_size_x * fft_size_y;
    static constexpr size_t flat_fft_size_bytes = flat_fft_size * sizeof(complex_type);
    std::vector<complex_type> output_host(flat_fft_size, {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()});
    CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), output, flat_fft_size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Performance measurements
    auto time = example::measure_execution_ms(
        [&](cudaStream_t stream) {
            fft_2d_execution(stream);
        },
        cufftdx_example_warm_up_runs,
        cufftdx_example_performance_runs,
        stream);

    // Return results
    return example::fft_results<T>{ output_host, (time / cufftdx_example_performance_runs) };
}

// Notes:
// * This examples shows how to use cuFFTDx to run multi-dimensional FFT. Final performance will vary depending on the
// FFT definitions (precision, size, type, ept, fpb) and other user customizations.
// * Best possible performance requires adapting parameters in the sample to particular set of parameters and code customizations.
// * Only FP32 supported in this example.
// * cuFFTDx with enabled shared memory IO usually be the faster cuFFTDx option for larger (>512) sizes.
// * The shared memory IO cuFFTDx has high shared memory requirements and will not work for all possible sizes in X dimension.
template<unsigned int Arch>
void fft_2d() {
    using precision_type                     = float;
    using complex_type                       = cufftdx::complex<precision_type>;

    // FFT Sizes
    static constexpr unsigned int fft_size_y = 1024;
    static constexpr unsigned int fft_size_x = 1024;
    // Kernel Settings
    static constexpr unsigned int ept_y = 16;
    static constexpr unsigned int fpb_y = 1;
    static constexpr unsigned int ept_x = 16;
    static constexpr unsigned int fpb_x = 8;

    // Other recommended configurations to test:
    // 1:
    // static constexpr unsigned int fft_size_y = 16384;
    // static constexpr unsigned int fft_size_x = 4096;
    // static constexpr unsigned int ept_y = 16;
    // static constexpr unsigned int fpb_y = 1;
    // static constexpr unsigned int ept_x = 16;
    // static constexpr unsigned int fpb_x = 2;
    // 2:
    // static constexpr unsigned int fft_size_y = 2048;
    // static constexpr unsigned int fft_size_x = 2048;
    // static constexpr unsigned int ept_y = 16;
    // static constexpr unsigned int fpb_y = 1;
    // static constexpr unsigned int ept_x = 8;
    // static constexpr unsigned int fpb_x = 4;
    // 3:
    // static constexpr unsigned int fft_size_y = 128;
    // static constexpr unsigned int fft_size_x = 128;
    // static constexpr unsigned int ept_y = 16;
    // static constexpr unsigned int fpb_y = 1;
    // static constexpr unsigned int ept_x = 16;
    // static constexpr unsigned int fpb_x = 8;
    // 4:
    // static constexpr unsigned int fft_size_y = 1281;
    // static constexpr unsigned int fft_size_x = 721;
    // static constexpr unsigned int ept_y = 16;
    // static constexpr unsigned int fpb_y = 1;
    // static constexpr unsigned int ept_x = 8;
    // static constexpr unsigned int fpb_x = 4;

    using namespace cufftdx;
    using fft_base = decltype(Block() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                              Precision<precision_type>() + SM<Arch>());
    using fft_y    = decltype(fft_base() + Size<fft_size_y>() + ElementsPerThread<ept_y>() + FFTsPerBlock<fpb_y>());
    using fft_x    = decltype(fft_base() + Size<fft_size_x>() + ElementsPerThread<ept_x>() + FFTsPerBlock<fpb_x>());
    using fft      = fft_y;

    // Host data
    static constexpr size_t flat_fft_size       = fft_size_x * fft_size_y;
    static constexpr size_t flat_fft_size_bytes = flat_fft_size * sizeof(complex_type);
#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_2D
    std::vector<complex_type> input_host(flat_fft_size);
    for (size_t i = 0; i < flat_fft_size; i++) {
        float sign      = (i % 3 == 0) ? -1.0f : 1.0f;
        input_host[i].x = sign * static_cast<float>(i) / flat_fft_size;
        input_host[i].y = sign * static_cast<float>(i) / flat_fft_size;
    }
#else
    auto input_host = example::get_random_complex_data<precision_type>(flat_fft_size, -1, 1);
#endif

    // Device data
    complex_type* input;
    complex_type* output;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&input, flat_fft_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output, flat_fft_size_bytes));

    // Copy host to device
    CUDA_CHECK_AND_EXIT(cudaMemset(output, 0b11111111, flat_fft_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(input, input_host.data(), flat_fft_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    // cuFFTDx 2D
    auto cufftdx_results = cufftdx_fft_2d<fft_x, fft_y, false>(input, output, stream);

    // cuFFTDx 2D
    // * Uses shared memory to speed-up IO in the strided kernel
    auto cufftdx_smemio_results = cufftdx_fft_2d<fft_x, fft_y, true>(input, output, stream);

    // cuFFT as reference
    auto cufft_results = cufft_fft_2d(fft_size_x, fft_size_y, input, output, stream);

    // Destroy created CUDA stream
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    // Free CUDA buffers
    CUDA_CHECK_AND_EXIT(cudaFree(input));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    std::cout << "FFT: (" << fft_size_x << ", " << fft_size_y << ")\n";

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_2D
    std::cout << "cuFFT, cuFFTDx\n";
    for (size_t i = 0; i < 8; i++) {
        std::cout << i << ": ";
        std::cout << "(" << cufft_results.output[i].x << ", " << cufft_results.output[i].y << ")";
        std::cout << ", ";
        std::cout << "(" << cufftdx_results.output[i].x << ", " << cufftdx_results.output[i].y << ")";
        std::cout << "\n";
    }
#endif

    bool success = true;
    std::cout << "Correctness results:\n";
    // Check if cuFFTDx results are correct
    {
        auto fft_error = example::fft_signal_error::calculate_for_complex_values(cufftdx_results.output, cufft_results.output);
        std::cout << "cuFFTDx\n";
        std::cout << "L2 error: " << fft_error.l2_relative_error << "\n";
        std::cout << "Peak error (index: " << fft_error.peak_error_index << "): " << fft_error.peak_error << "\n";
        if(success) {
            success = (fft_error.l2_relative_error < 0.001);
        }
    }
    // Check cuFFTDx with shared memory io
    {
        auto fft_error = example::fft_signal_error::calculate_for_complex_values(cufftdx_smemio_results.output, cufft_results.output);
        std::cout << "cuFFTDx (shared memory IO)\n";
        std::cout << "L2 error: " << fft_error.l2_relative_error << "\n";
        std::cout << "Peak error (index: " << fft_error.peak_error_index << "): " << fft_error.peak_error << "\n";
        if(success) {
            success = (fft_error.l2_relative_error < 0.001);
        }
    }

    // Print performance results
    if(success) {
        std::cout << "\nPerformance results:\n";
        std::cout << std::setw(28) << "cuFFTDx: " << cufftdx_results.avg_time_in_ms << " [ms]\n";
        std::cout << std::setw(28) << "cuFFTDx (shared memory IO): " << cufftdx_smemio_results.avg_time_in_ms << " [ms]\n";
        std::cout << std::setw(28) << "cuFFT: " << cufft_results.avg_time_in_ms << " [ms]\n";
    }

    if (success) {
        std::cout << "\nSuccess\n";
    } else {
        std::cout << "\nFailure\n";
        std::exit(1);
    }
}

template<unsigned int Arch>
struct fft_2d_functor {
    void operator()() { return fft_2d<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<fft_2d_functor>();
}
