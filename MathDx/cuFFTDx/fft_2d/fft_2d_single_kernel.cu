#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>

#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <cufftdx.hpp>
#include <cufft.h>

#include "block_io.hpp"
#include "block_io_strided.hpp"
#include "common.hpp"
#include "random.hpp"

// #define CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_2D
inline constexpr unsigned int cufftdx_example_warm_up_runs = 5;
inline constexpr unsigned int cufftdx_example_performance_runs = 20;

template<class FFTX,
         class FFTY,
         bool UseSharedMemoryStridedIO,
         class ComplexType                = typename FFTX::value_type,
         unsigned int RequiredStorageSize = std::max({FFTX::storage_size, FFTY::storage_size})>
__launch_bounds__(FFTX::max_threads_per_block) __global__
    void fft_2d_kernel(const ComplexType*            input,
                       ComplexType*                  output,
                       typename FFTX::workspace_type workspace_x,
                       typename FFTY::workspace_type workspace_y) {
    using complex_type = ComplexType;

    // Shared memory
    extern __shared__ complex_type shared_mem[];

    // Local array for thread
    complex_type thread_data[RequiredStorageSize];

    // FFTY

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    example::io<FFTY>::load(input, thread_data, local_fft_id);

    // Execute FFTY
    FFTY().execute(thread_data, shared_mem, workspace_y);

    // Save results
    example::io<FFTY>::store(thread_data, output, local_fft_id);

    // Synchronize the whole CUDA Grid
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    grid.sync();

    // FFTX

    static constexpr unsigned int stride = cufftdx::size_of<FFTY>::value;
    // Load data from global memory to registers
    if constexpr (UseSharedMemoryStridedIO) {
        example::io_strided<FFTX>::load_strided<stride>(output, thread_data, shared_mem, local_fft_id);
    } else {
        example::io_strided<FFTX>::load_strided<stride>(output, thread_data, local_fft_id);
    }

    // Execute FFTX
    FFTX().execute(thread_data, shared_mem, workspace_x);

    // Save results
    if constexpr (UseSharedMemoryStridedIO) {
        example::io_strided<FFTX>::store_strided<stride>(thread_data, shared_mem, output, local_fft_id);
    } else {
        example::io_strided<FFTX>::store_strided<stride>(thread_data, output, local_fft_id);
    }
}

template<class T>
example::fft_results<T> cufft_fft_2d(unsigned int fft_size_x,
                                     unsigned int fft_size_y,
                                     T*           input,
                                     T*           output,
                                     cudaStream_t stream) {
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
    auto cufft_execution = [&](cudaStream_t /* stream */) {
        CUFFT_CHECK_AND_EXIT(cufftExecC2C(plan, cufft_input, cufft_output, CUFFT_FORWARD));
    };
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Correctness run
    cufft_execution(stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    // Copy results to host
    const size_t   flat_fft_size       = fft_size_x * fft_size_y;
    const size_t   flat_fft_size_bytes = flat_fft_size * sizeof(complex_type);
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
    return example::fft_results<T> {output_host, (time / cufftdx_example_performance_runs)};
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

    // Checks that FFTX and FFTY can execute in the same kernel
    static_assert((FFTX::block_dim.x == FFTY::block_dim.x) && (FFTX::block_dim.y == FFTY::block_dim.y),
                  "Required block dimensions for FFTX and FFTY must be the same");

    complex_type* cufftdx_input  = reinterpret_cast<complex_type*>(input);
    complex_type* cufftdx_output = reinterpret_cast<complex_type*>(output);

    // Shared memory IO for strided kernel may require more memory than FFTX::shared_memory_size.
    // Note: For some fft_size_x and depending on GPU architecture fft_x_shared_memory_smem_io may exceed max shared
    // memory and cudaFuncSetAttribute will fail.
    const unsigned int fft_shared_memory_smem_io =
        std::max<unsigned>({FFTX::shared_memory_size,
                            FFTY::shared_memory_size,
                            FFTX::ffts_per_block * fft_size_x * sizeof(complex_type),
                            FFTY::ffts_per_block * fft_size_y * sizeof(complex_type)});
    const unsigned int fft_shared_memory =
        UseSharedMemoryStridedIO ? fft_shared_memory_smem_io
                                 : std::max<unsigned>({FFTX::shared_memory_size, FFTY::shared_memory_size});
    const void* kernel = (const void*)fft_2d_kernel<FFTX, FFTY, UseSharedMemoryStridedIO, complex_type>;
    auto error_code    = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, fft_shared_memory);
    CUDA_CHECK_AND_EXIT(error_code);

    // Create workspaces for FFTs
    auto workspace_y = cufftdx::make_workspace<FFTY>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_x = cufftdx::make_workspace<FFTX>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);

    // Synchronize device before execution
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    const dim3 block_dim = FFTX::block_dim;
    const dim3 grid_size = {
        std::max<unsigned>({(fft_size_y / FFTY::ffts_per_block), (fft_size_x / FFTX::ffts_per_block)}), 1, 1};
    typename FFTX::workspace_type workspace_x_device = workspace_x;
    typename FFTY::workspace_type workspace_y_device = workspace_y;
    void*                         args[] = {&cufftdx_input, &cufftdx_output, &workspace_x_device, &workspace_y_device};
    // Define 2D FFT execution
    auto fft_2d_execution = [&](cudaStream_t stream) {
        CUDA_CHECK_AND_EXIT(cudaLaunchCooperativeKernel(kernel, grid_size, block_dim, args, fft_shared_memory, stream));
        CUDA_CHECK_AND_EXIT(cudaGetLastError());
    };

    // Correctness run
    fft_2d_execution(stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    // Copy results to host
    static constexpr size_t   flat_fft_size       = fft_size_x * fft_size_y;
    static constexpr size_t   flat_fft_size_bytes = flat_fft_size * sizeof(complex_type);
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
    return example::fft_results<T> {output_host, (time / cufftdx_example_performance_runs)};
}

// Notes:
// * This example shows how to use cuFFTDx to run multi-dimensional FFT. Final performance will vary depending on the
// FFT definitions (precision, size, type, ept, fpb) and other user customizations.
// * Best possible performance requires adapting parameters in the sample to particular set of parameters and code customizations.
// * Only FP32 supported in this example.
// * The shared memory IO cuFFTDx has high shared memory requirements and will not work for all possible sizes in X dimension.
// * cudaLaunchCooperativeKernel puts restrictions on how big the FFT can be. All batches must be able to execute at the same time
// on the GPU.
// * The best results are for a square FFTs (fft_size_x == fft_size_y).
template<unsigned int Arch>
void fft_2d() {
    using precision_type = float;
    using complex_type   = cufftdx::complex<precision_type>;

    // // FFT Sizes
    // static constexpr unsigned int fft_size_y = 512;
    // static constexpr unsigned int fft_size_x = 512;
    // // Kernel Settings
    // static constexpr unsigned int ept_y = 8;
    // static constexpr unsigned int fpb_y = 8;
    // static constexpr unsigned int ept_x = 8;
    // static constexpr unsigned int fpb_x = fpb_y; // fpb for X and Y dimensions must be the same

    // Other recommended configurations to test:
    // 1:
    // static constexpr unsigned int fft_size_y = 128;
    // static constexpr unsigned int fft_size_x = 128;
    // static constexpr unsigned int ept_y = 8;
    // static constexpr unsigned int fpb_y = 4;
    // static constexpr unsigned int ept_x = 8;
    // static constexpr unsigned int fpb_x = fpb_y;
    // 2:
    static constexpr unsigned int fft_size_y = 256;
    static constexpr unsigned int fft_size_x = 128;
    static constexpr unsigned int ept_y = 16;
    static constexpr unsigned int fpb_y = 4;
    static constexpr unsigned int ept_x = 8;
    static constexpr unsigned int fpb_x = fpb_y;

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
        auto fft_error =
            example::fft_signal_error::calculate_for_complex_values(cufftdx_results.output, cufft_results.output);
        std::cout << "cuFFTDx\n";
        std::cout << "L2 error: " << fft_error.l2_relative_error << "\n";
        std::cout << "Peak error (index: " << fft_error.peak_error_index << "): " << fft_error.peak_error << "\n";
        if (success) {
            success = (fft_error.l2_relative_error < 0.001);
        }
    }
    // Check cuFFTDx with shared memory io
    {
        auto fft_error = example::fft_signal_error::calculate_for_complex_values(cufftdx_smemio_results.output,
                                                                                 cufft_results.output);
        std::cout << "cuFFTDx (shared memory IO)\n";
        std::cout << "L2 error: " << fft_error.l2_relative_error << "\n";
        std::cout << "Peak error (index: " << fft_error.peak_error_index << "): " << fft_error.peak_error << "\n";
        if (success) {
            success = (fft_error.l2_relative_error < 0.001);
        }
    }

    // Print performance results
    if (success) {
        std::cout << "\nPerformance results:\n";
        std::cout << std::setw(28) << "cuFFTDx: " << cufftdx_results.avg_time_in_ms << " [ms]\n";
        std::cout << std::setw(28) << "cuFFTDx (shared memory IO): " << cufftdx_smemio_results.avg_time_in_ms
                  << " [ms]\n";
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
