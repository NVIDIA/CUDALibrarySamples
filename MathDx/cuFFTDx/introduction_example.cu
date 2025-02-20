#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "common.hpp"


template<class FFT>
__launch_bounds__(FFT::max_threads_per_block)
    __global__ void block_fft_kernel(typename FFT::value_type* data, typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // Local batch id of this FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Global batch id of this FFT in CUDA grid is equal to number of batches per CUDA block (ffts_per_block)
    // times CUDA block id, plus local batch id.
    const unsigned int global_fft_id = (blockIdx.x * FFT::ffts_per_block) + local_fft_id;

    // Load data from global memory to registers
    const unsigned int     offset = cufftdx::size_of<FFT>::value * global_fft_id;
    constexpr unsigned int stride = FFT::stride;
    unsigned int           index  = offset + threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
            thread_data[i] = data[index];
            index += stride;
        }
    }

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_memory[];
    FFT().execute(thread_data, shared_memory, workspace);

    // Save results
    index = offset + threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
            data[index] = thread_data[i];
            index += stride;
        }
    }
}

// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point C2C float precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template<unsigned int Arch>
void introduction_example() {
    using namespace cufftdx;

    // FFT definition
    //
    // Size, precision, type, direction are defined with operators.
    // Block() operator informs that FFT will be executed on block level.
    // Shared memory is required for co-operation between threads.
    //
    // Additionally:
    // * FFTsPerBlock operator defines how many FFTs (batches) are executed in a single CUDA block,
    // * ElementsPerThread operators defines how FFT calculations are mapped into a CUDA block, i.e. how many thread are required, and
    // * SM operator defines targeted CUDA architecture.
    using FFT          = decltype(Size<128>() + Precision<float>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>()
                         + ElementsPerThread<8>() + FFTsPerBlock<2>()
                         + SM<Arch>() + Block());
    using complex_type = typename FFT::value_type;

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    // Allocate managed memory for input/output
    complex_type* data;
    auto          size       = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto          size_bytes = size * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&data, size_bytes));
    // Generate data
    for (size_t i = 0; i < size; i++) {
        data[i] = complex_type {float(i), -float(i)};
    }

    cudaError_t error_code = cudaSuccess;
    auto workspace = make_workspace<FFT>(error_code, stream);
    CUDA_CHECK_AND_EXIT(error_code);

    std::cout << "input [1st FFT]:\n";
    for (size_t i = 0; i < FFT::input_length; i++) {
        std::cout << data[i].x << " " << data[i].y << std::endl;
    }

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFT::shared_memory_size));

    // Invokes kernel with FFT::block_dim threads in CUDA block
    block_fft_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size, stream>>>(data, workspace);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::cout << "output [1st FFT]:\n";
    for (size_t i = 0; i < FFT::output_length; i++) {
        std::cout << data[i].x << " " << data[i].y << std::endl;
    }

    CUDA_CHECK_AND_EXIT(cudaFree(data));
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    std::cout << "Success" << std::endl;
}

template<unsigned int Arch>
struct introduction_example_functor {
    void operator()() { return introduction_example<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<introduction_example_functor>();
}
