#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>

#include "block_io.hpp"
#include "common.hpp"
#include "random.hpp"

// #define CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_3D
// #define CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_3D_SIMPLE_IO

template<unsigned int MaxThreadsPerBlock, class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(MaxThreadsPerBlock) __global__
    void cufftdx_3d_fft_single_block_kernel(const ComplexType* input, ComplexType* output) {
    using complex_type = ComplexType;
    static constexpr unsigned int fft_size = cufftdx::size_of<FFT>::value;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // Shared memory use for exchanging data between threads
    extern __shared__ __align__(alignof(float4)) complex_type shared_memory[];

    // Load data from global memory to registers.
    static constexpr unsigned int stride_x = fft_size * fft_size;
    unsigned int                  index    = (threadIdx.x + threadIdx.y * fft_size);
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        thread_data[i] = input[index];
        index += stride_x;
    }

    // Execute FFT in X dimension
    FFT().execute(thread_data);

    // Exchange/transpose via shared memory
    index = (threadIdx.x + threadIdx.y * fft_size);
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        shared_memory[index] = thread_data[i];
        index += stride_x;
    }
    __syncthreads();
    static constexpr unsigned int stride_y = cufftdx::size_of<FFT>::value;
    index                                  = threadIdx.x + threadIdx.y * fft_size * fft_size;
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        thread_data[i] = shared_memory[index];
        index += stride_y;
    }

    // Execute FFT in Y dimension
    FFT().execute(thread_data);

    // Exchange/transpose via shared memory
    index = threadIdx.x + threadIdx.y * fft_size * fft_size;
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        shared_memory[index] = thread_data[i];
        index += stride_y;
    }
    __syncthreads();
    index = (threadIdx.x + threadIdx.y * fft_size) * fft_size;
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        thread_data[i] = shared_memory[index];
        index += 1;
    }

    // Execute FFT in Z dimension
    FFT().execute(thread_data);

    // Save results
#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_3D_SIMPLE_IO
    // Simple IO with poor global memory pattern:
    // Storing the data with stride=1 results in poor global memory
    // write pattern with little or none coalescing
    index = (threadIdx.x + threadIdx.y * fft_size) * fft_size;
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        output[index] = thread_data[i];
        index += 1;
    }
#else
    // Shared memory IO:
    // Exchanging data via shared memory results in a much better global
    // memory patter with good coalescing
    index = (threadIdx.x + threadIdx.y * fft_size) * fft_size;
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        shared_memory[index] = thread_data[i];
        index += 1;
    }
    __syncthreads();
    index = (threadIdx.x + threadIdx.y * fft_size);
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        thread_data[i] = shared_memory[index];
        index += stride_x;
    }

    index = (threadIdx.x + threadIdx.y * fft_size);
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        output[index] = thread_data[i];
        index += stride_x;
    }
#endif
}

void cufft_3d_fft(unsigned int  fft_size_x,
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
    CUFFT_CHECK_AND_EXIT(cufftExecC2C(plan, input, output, CUFFT_FORWARD))
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Clean-up
    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan));
}

template<unsigned int FFTSize>
void cufftdx_3d_fft_single_block(float2* input, float2* output, cudaStream_t stream) {
    using namespace cufftdx;

    static constexpr unsigned int fft_size_x = FFTSize;
    static constexpr unsigned int fft_size_y = FFTSize;
    static constexpr unsigned int fft_size_z = FFTSize;

    using FFT = decltype(Thread() + Size<fft_size_x>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                         Precision<float>());
    using complex_type = typename FFT::value_type;

    constexpr dim3         block_dim             = {fft_size_z, fft_size_y, 1};
    constexpr unsigned int max_threads_per_block = block_dim.x * block_dim.y * block_dim.z;
    const size_t           shared_memory_size    = (fft_size_x * fft_size_y * fft_size_z) * sizeof(complex_type);

    const auto kernel = cufftdx_3d_fft_single_block_kernel<max_threads_per_block, FFT, complex_type>;

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

    complex_type* cufftdx_input  = reinterpret_cast<complex_type*>(input);
    complex_type* cufftdx_output = reinterpret_cast<complex_type*>(output);
    kernel<<<1, block_dim, shared_memory_size, stream>>>(cufftdx_input, cufftdx_output);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// Example showing how to perform a small 3D FFT in a single CUDA block using cuFFTDx thread execution. In this example
// every dimension is the same.
//
// Notes:
// * This example shows how to use cuFFTDx to run a multi-dimensional FFT. Final performance will vary depending on the
// FFT definitions (precision, size, type, ept, fpb) and other user customizations.
// * Best possible performance requires adapting parameters in the sample to particular set of parameters and code customizations.
// * Only FP32 was tested for this example, other types might require adjustments.
int main(int, char**) {
    // 3D FFT where X=Y=Z
    static constexpr unsigned int fft_size = 16;

    // Generate random input data on host
    const unsigned int flat_fft_size = fft_size * fft_size * fft_size;
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

    // Allocate host output for cuFFT and cuFFTDx
    std::vector<float2> cufft_output(flat_fft_size);
    std::vector<float2> cufftdx_output(flat_fft_size);

    // Copy input to the device
    CUDA_CHECK_AND_EXIT(cudaMemcpy(input, host_input.data(), flat_fft_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    cufft_3d_fft(fft_size, fft_size, fft_size, input, output, stream);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(cufft_output.data(), output, flat_fft_size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    cufftdx_3d_fft_single_block<fft_size>(input, output, stream);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(cufftdx_output.data(), output, flat_fft_size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Clean-up
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(input));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    // Check if cuFFTDx results are correct
    auto fft_error = example::fft_signal_error::calculate_for_complex_values(cufftdx_output, cufft_output);

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_3D
    std::cout << "cuFFT, cuFFTDx\n";
    for (size_t i = 0; i < 8; i++) {
        std::cout << i << ": ";
        std::cout << "(" << cufft_output[i].x << ", " << cufft_output[i].y << ")";
        std::cout << ", ";
        std::cout << "(" << cufftdx_output[i].x << ", " << cufftdx_output[i].y << ")";
        std::cout << "\n";
    }
#endif

    std::cout << "Correctness results:\n";
    std::cout << "L2 error: " << fft_error.l2_relative_error << "\n";
    std::cout << "Peak error (index: " << fft_error.peak_error_index << "): " << fft_error.peak_error << "\n";

    if(fft_error.l2_relative_error < 0.001) {
        std::cout << "Success\n";
        return 0;
    } else {
        std::cout << "Failure\n";
        return 1;
    }
}
