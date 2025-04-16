#include <cufftdx.hpp>

#include "../common/common.hpp"

// Include cuFFT-dumped LTOIR database header file
#include "lto_database.hpp.inc"

// This example is a copy of the 02_simple_fft_block/00_simple_fft_block.cu example and
// has been modified to work with cufft lto. It contains only 2 modifications, the inclusion
// of a auto-generated lto header file, lto_database.hpp.inc, and the addition of the
// experimental ltoir CodeType. to the FFT type. The bulk of the modifications made to the
// 02_simple_fft_block example were made in the build scripts which construct and link the
// external lto database as well as create the database header file.

template<class FFT>
__launch_bounds__(FFT::max_threads_per_block)
    __global__ void block_fft_kernel(typename FFT::value_type* data, typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;

    complex_type thread_data[FFT::storage_size];

    const unsigned int local_fft_id = threadIdx.y;
    const unsigned int global_fft_id = (blockIdx.x * FFT::ffts_per_block) + local_fft_id;

    const unsigned int     offset = cufftdx::size_of<FFT>::value * global_fft_id;
    constexpr unsigned int stride = FFT::stride;
    unsigned int           index  = offset + threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
            thread_data[i] = data[index];
            index += stride;
        }
    }

    extern __shared__ __align__(alignof(float4)) complex_type shared_memory[];
    FFT().execute(thread_data, shared_memory, workspace);

    index = offset + threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
            data[index] = thread_data[i];
            index += stride;
        }
    }
}

template<unsigned int Arch>
void introduction_lto_helper() {
    using FFT_without_code_type = decltype(cufftdx::Block() +
                                           cufftdx::Size<128>() +
                                           cufftdx::Type<cufftdx::fft_type::c2c>() +
                                           cufftdx::Direction<cufftdx::fft_direction::forward>() +
                                           cufftdx::Precision<float>() +
                                           cufftdx::ElementsPerThread<8>() +
                                           cufftdx::FFTsPerBlock<2>() +
                                           cufftdx::SM<Arch>());
    // Add experimental ltoir code_type to FFT type
    using FFT = decltype(FFT_without_code_type() + cufftdx::experimental::CodeType<cufftdx::experimental::code_type::ltoir>());

    // Validate that the selected fft is in fact an ltoir kernel
    static_assert(FFT::code == cufftdx::experimental::code_type::ltoir, "Selected implementation code type is not LTO-IR");

    using complex_type = typename FFT::value_type;

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    complex_type* data;
    auto          size       = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto          size_bytes = size * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&data, size_bytes));

    for (size_t i = 0; i < size; i++) {
        data[i] = complex_type {float(i), -float(i)};
    }

    cudaError_t error_code = cudaSuccess;
    auto workspace = cufftdx::make_workspace<FFT>(error_code, stream);
    CUDA_CHECK_AND_EXIT(error_code);

    std::cout << "input [1st FFT]:\n";
    for (size_t i = 0; i < FFT::input_length; i++) {
        std::cout << data[i].x << " " << data[i].y << std::endl;
    }

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFT::shared_memory_size));

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
struct introduction_lto_helper_functor {
    void operator()() { return introduction_lto_helper<Arch>(); }
};

int main() {
    return example::sm_runner<introduction_lto_helper_functor>();
}
