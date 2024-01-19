#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include <cublasdx.hpp>
#include <cufftdx.hpp>

#include "common.hpp"
#include "block_io.hpp"

template<class FFT, class GEMM, class ValueType = typename GEMM::value_type>
void reference(const ValueType* a,
               const ValueType* b,
               ValueType* c,
               const ValueType  alpha,
               const ValueType  beta,
               ValueType*       output,
               cudaStream_t     stream) {
    constexpr auto m = cublasdx::size_of<GEMM>::m;
    constexpr auto n = cublasdx::size_of<GEMM>::n;
    constexpr auto k = cublasdx::size_of<GEMM>::k;

    // Prepare cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK_AND_EXIT(cublasCreate(&handle));
    CUBLAS_CHECK_AND_EXIT(cublasSetStream(handle, stream));
    constexpr bool is_a_transposed = (cublasdx::transpose_mode_of<GEMM>::a_transpose_mode == cublasdx::transpose_mode::transposed);
    constexpr bool is_b_transposed = (cublasdx::transpose_mode_of<GEMM>::b_transpose_mode == cublasdx::transpose_mode::transposed);
    const auto a_transpose = is_a_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto b_transpose = is_b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Prepare cuFFT
    const unsigned int fft_size = cublasdx::size_of<GEMM>::m * cublasdx::size_of<GEMM>::n;
    const unsigned int batch_size = 1;

    cufftHandle plan;
    CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan, fft_size, CUFFT_C2C, batch_size));
    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan, stream));

    // Run cuBLAS
    CUBLAS_CHECK_AND_EXIT(cublasCgemm(handle,
                                      a_transpose,
                                      b_transpose,
                                      m,
                                      n,
                                      k,
                                      reinterpret_cast<const cuFloatComplex*>(&alpha),
                                      reinterpret_cast<const cuFloatComplex*>(a),
                                      is_a_transposed ? k : m,
                                      reinterpret_cast<const cuFloatComplex*>(b),
                                      is_b_transposed ? n : k,
                                      reinterpret_cast<const cuFloatComplex*>(&beta),
                                      reinterpret_cast<cuFloatComplex*>(c),
                                      m));

    // Run cuFFT
    CUFFT_CHECK_AND_EXIT(
        cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(c), reinterpret_cast<cufftComplex*>(output), CUFFT_FORWARD));

    CUBLAS_CHECK_AND_EXIT(cublasDestroy(handle));
    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan));
}

template<class FFT>
inline __device__ unsigned int batch_offset(const unsigned int local_fft_id,
                                            const unsigned int ffts_per_block = blockDim.y) {
    unsigned int global_fft_id = ffts_per_block == 1 ? blockIdx.x : (blockIdx.x * ffts_per_block + local_fft_id);
    return cufftdx::size_of<FFT>::value * global_fft_id;
}

template<class FFT, class GEMM, class ValueType = typename GEMM::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__ void gemm_fft_kernel(const ValueType* a,
                                                                              const ValueType* b,
                                                                              const ValueType* c,
                                                                              const ValueType  alpha,
                                                                              const ValueType  beta,
                                                                              ValueType*       output) {
    #if CUBLASDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using fft_complex_type = example::value_type_t<FFT>;
    using blas_complex_type = example::value_type_t<GEMM>;
    #else
    using blas_complex_type = typename GEMM::value_type;
    using fft_complex_type = typename FFT::value_type;
    #endif

    static_assert(std::is_same_v<fft_complex_type, blas_complex_type>, "BLAS and FFT complex type should match");

    using complex_type = blas_complex_type;
    using value_type = ValueType;
    constexpr unsigned int block_size = GEMM::block_dim.x * GEMM::block_dim.y * GEMM::block_dim.z;

    extern __shared__ complex_type smem[];

    // cuBLASDx

    constexpr auto m = cublasdx::size_of<GEMM>::m;
    constexpr auto n = cublasdx::size_of<GEMM>::n;
    constexpr auto k = cublasdx::size_of<GEMM>::k;

    complex_type* smem_a = smem;
    complex_type* smem_b = smem + GEMM::a_size;
    complex_type* smem_c = smem + GEMM::a_size + GEMM::b_size;

    // Load a, b, c from global to shared memory
    example::io<GEMM>::a_fast_load<block_size>(smem_a, a);
    example::io<GEMM>::b_fast_load<block_size>(smem_b, b);
    example::io<GEMM>::c_fast_load<block_size>(smem_c, c);
    __syncthreads();

    // Execute GEMM
    GEMM().execute(alpha, smem_a, smem_b, beta, smem_c);
    __syncthreads();

    // cuFFTDx

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from shared memory to registers
    const unsigned int     offset = batch_offset<FFT>(local_fft_id, FFT::ffts_per_block);
    constexpr unsigned int stride = FFT::stride;
    unsigned int           index  = offset + threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
            thread_data[i] = smem_c[index];
            index += stride;
        }
    }
    __syncthreads();

    // Execute FFT on registers
    FFT().execute(thread_data, smem);

    // Save results
    index = offset + threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
            output[index] = thread_data[i];
            index += stride;
        }
    }
}

// In this example cuBLASDx and cuFFTDx libraries are combined to perform GEMM and FFT in one pipeline. The kernel
// computes the following operations:
//     1) C = alpha * A * B + beta * C
//     2) 1D FFT(C)
// First the data of matrices A, B, and C is loaded into shared memory. After that the GEMM is executed, and 1D FFT
// is applied to the results (matrix C). The results are stored back to global memory.
//
// Important notes:
// * Results are verified against cuFFT and cuBLAS.
// * FFT and GEMM must be compatible in terms of the requirements on block dimensions. cuFFTDx requirements are more firm, and
//   therefore FFT::block_dim is enforced for GEMM.
// * Shared memory required by the kernel is the max of the amount required by FFT and GEMM.
// * For cuBLASDx GEMM description and cuFFTDx FFT description GEMM::value_type and FFT::value_type are the same types if:
//     * GEMM has complex type set, and
//     * Precision for GEMM and FFT is float or double.
template<unsigned int Arch>
int gemm_fft() {
    using precision_type = float;
    static_assert(!std::is_same_v<precision_type, __half>, "half precision is not supported in this example");

    using FFT          = decltype(cufftdx::Block() + cufftdx::Size<64>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
                         cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::Precision<precision_type>() +
                         cufftdx::ElementsPerThread<2>() + cufftdx::FFTsPerBlock<1>() + cufftdx::SM<Arch>());

    using GEMM =
        decltype(cublasdx::Size<8, 8, 8>() +
                 cublasdx::Precision<precision_type>() +
                 cublasdx::Type<cublasdx::type::complex>() +
                 cublasdx::Function<cublasdx::function::MM>() +
                 cublasdx::TransposeMode<cublasdx::transpose_mode::non_transposed,
                                         cublasdx::transpose_mode::non_transposed>() +
                 cublasdx::Block() +
                 cublasdx::BlockDim<FFT::block_dim.x, FFT::block_dim.y, FFT::block_dim.z>() +
                 cublasdx::SM<Arch>());

    #if CUBLASDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using blas_complex_type = example::value_type_t<GEMM>;
    using fft_complex_type = example::value_type_t<FFT>;
    #else
    using blas_complex_type = typename GEMM::value_type;
    using fft_complex_type = typename FFT::value_type;
    #endif

    static_assert(std::is_same_v<blas_complex_type, fft_complex_type>, "BLAS and FFT complex types should be the same");
    using complex_type = blas_complex_type;

    // Checking that FFT matches GEMM output
    static_assert(cufftdx::size_of<FFT>::value == (cublasdx::size_of<GEMM>::m * cublasdx::size_of<GEMM>::n),
                  "FFT must have the same size as C matrix (MxN)");
    // Checking that block dims match
    static_assert((FFT::block_dim.x == GEMM::block_dim.x) && (FFT::block_dim.y == GEMM::block_dim.y) &&
                      (FFT::block_dim.z == GEMM::block_dim.z),
                  "FFT must require the same CUDA block dimenions as GEMM");
    static_assert(sizeof(fft_complex_type) == sizeof(blas_complex_type), "FFT::value_type matches as GEMM::value_type");

    // Allocate managed memory
    complex_type* buffer;
    complex_type* a;
    complex_type* b;
    complex_type* c;
    complex_type* output;
    complex_type* reference_output;
    auto          size       = (GEMM::a_size + // a
                 GEMM::a_size + // b
                 GEMM::c_size + // c
                 GEMM::c_size + // output
                 GEMM::c_size   // reference_output
    );
    auto          size_bytes = size * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&buffer, size_bytes));
    a                = buffer;
    b                = a + GEMM::a_size;
    c                = b + GEMM::b_size;
    output           = c + GEMM::c_size;
    reference_output = output + GEMM::c_size;

    complex_type alpha = {float(1), float(1)};
    complex_type beta  = {float(1), float(1)};

    // Fill the a, b, c matrices
    {
        float base = cublasdx::size_of<GEMM>::m * cublasdx::size_of<GEMM>::n * cublasdx::size_of<GEMM>::k;
        for (size_t i = 0; i < GEMM::a_size; i++) {
            a[i] = complex_type {float(i) / base, float(i) / base};
        }
        for (size_t i = 0; i < GEMM::b_size; i++) {
            b[i] = complex_type {float(i) / base, float(i) / base};
        }
        for (size_t i = 0; i < GEMM::c_size; i++) {
            c[i] = complex_type {float(1) / base, float(1) / base};
        }
    }

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    // Prefetch memory to device
    {
        int device;
        CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));
        CUDA_CHECK_AND_EXIT(cudaMemPrefetchAsync(buffer, size_bytes, device, stream));
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    }

    // Get max shared memory required by FFT and GEMM
    constexpr auto shared_memory_size = std::max({FFT::shared_memory_size, GEMM::shared_memory_size});
    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        gemm_fft_kernel<FFT, GEMM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size));

    // Invokes cuBLASDx+cuFFTDx kernel with FFT::block_dim threads in CUDA block
    gemm_fft_kernel<FFT, GEMM><<<1, FFT::block_dim, shared_memory_size, stream>>>(a, b, c, alpha, beta, output);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // cuBLAS+cuFFT
    reference<FFT, GEMM>(a, b, c, alpha, beta, reference_output, stream);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::cout << "[cuBLASDx + cuFFTDx]:\n";
    for (size_t i = 0; i < cublasdx::size_of<GEMM>::m; i++) {     // rows
        for (size_t j = 0; j < cublasdx::size_of<GEMM>::n; j++) { // cols
            auto index = i * cublasdx::size_of<GEMM>::n + j;
            std::cout << "[" << output[index].x << ", " << output[index].y << "]\t";
        }
        std::cout << "\n";
    }
    std::cout << "[cuBLAS + cuFFT]:\n";
    for (size_t i = 0; i < cublasdx::size_of<GEMM>::m; i++) {     // rows
        for (size_t j = 0; j < cublasdx::size_of<GEMM>::n; j++) { // cols
            auto index = i * cublasdx::size_of<GEMM>::n + j;
            std::cout << "[" << reference_output[index].x << ", " << reference_output[index].y << "]\t";
        }
        std::cout << "\n";
    }

    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(buffer));

    std::cout << "Success" << std::endl;
    return 0;
}

template<unsigned int Arch>
struct gemm_fft_functor {
    int operator()() { return gemm_fft<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<gemm_fft_functor>();
}
