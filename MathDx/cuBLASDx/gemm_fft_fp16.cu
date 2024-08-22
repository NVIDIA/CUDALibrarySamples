#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include <cublasdx.hpp>
#include <cufftdx.hpp>

#include "common.hpp"
#include "block_io.hpp"
#include "reference.hpp"

// Batch size (number of signals to process)
constexpr unsigned int batch_size = 2;

template <class T, class U>
void copy(T* source, U* destination, unsigned int size) {
    static_assert(example::is_complex<T>() && example::is_complex<U>(), "Expect complex types.");
    for (unsigned int i = 0; i < size; ++i) {
        destination[i].x  = source[i].real();
        destination[i].y  = source[i].imag();
    }
}

template<class FFT, class BLAS, class ValueType = cublasdx::complex<float>>
void reference(const ValueType* a,
               ValueType* b,
               ValueType* c,
               const ValueType  alpha,
               const ValueType  beta,
               ValueType*       output,
               cudaStream_t     stream) {
    constexpr auto m = cublasdx::size_of<BLAS>::m;
    constexpr auto n = cublasdx::size_of<BLAS>::n;
    constexpr auto k = cublasdx::size_of<BLAS>::k;

    // Prepare cuFFT
    const unsigned int fft_size = cublasdx::size_of<BLAS>::k;

    cufftHandle plan;
    CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan, fft_size, CUFFT_C2C, batch_size));
    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan, stream));

    // Run cuFFT
    CUFFT_CHECK_AND_EXIT(
        cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(b), reinterpret_cast<cufftComplex*>(b), CUFFT_FORWARD));

    // Prepare cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK_AND_EXIT(cublasCreate(&handle));
    CUBLAS_CHECK_AND_EXIT(cublasSetStream(handle, stream));
    constexpr bool is_a_transposed = (cublasdx::arrangement_of<BLAS>::a == cublasdx::row_major);
    constexpr bool is_b_transposed = (cublasdx::arrangement_of<BLAS>::b == cublasdx::row_major);
    const auto a_transpose = example::detail::get_cublas_transpose_mode(cublasdx::arrangement_of<BLAS>::a);
    const auto b_transpose = example::detail::get_cublas_transpose_mode(cublasdx::arrangement_of<BLAS>::b);
    static_assert(cublasdx::arrangement_of<BLAS>::c == cublasdx::arrangement::col_major, "Only column-major C matrix supported");

    // Run cuBLAS
    copy(c, output, m * n);
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
                                      reinterpret_cast<cuFloatComplex*>(output),
                                      m));

    CUBLAS_CHECK_AND_EXIT(cublasDestroy(handle));
    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan));
}

template<class FFT, class BLAS, class ValueType = typename example::uniform_value_type_t<BLAS>>
__launch_bounds__(FFT::max_threads_per_block) __global__ void gemm_fft_fp16_kernel(const ValueType* a,
                                                                                   const ValueType* b,
                                                                                   const ValueType* c,
                                                                                   const ValueType  alpha,
                                                                                   const ValueType  beta,
                                                                                   ValueType*       output) {
    #if CUBLASDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using blas_complex_type = example::uniform_value_type_t<BLAS>;
    using fft_complex_type = example::value_type_t<FFT>;
    #else
    using blas_complex_type = example::uniform_value_type_t<BLAS>;
    using fft_complex_type  = typename FFT::value_type;
    #endif

    using complex_type = blas_complex_type;
    using value_type = ValueType;
    constexpr unsigned int block_size = BLAS::block_dim.x * BLAS::block_dim.y * BLAS::block_dim.z;

    extern __shared__ complex_type smem[];

    auto [smem_a, smem_b, smem_c] = BLAS::slice_shared_memory(reinterpret_cast<char*>(smem));

    // Compute FFT(B, axis=0).
    fft_complex_type thread_data[FFT::storage_size];

    // Load data from global memory into registers for FFT, converting to RRII form from RIRI.
    example::load<FFT::elements_per_thread, cufftdx::size_of<FFT>::value, FFT::stride>(b, thread_data);
    __syncthreads();

    // Execute batched FFT on registers.
    FFT().execute(thread_data, reinterpret_cast<fft_complex_type *>(smem));
    __syncthreads();

    // Store register data into smem_b, converting back to RIRI form from RRII.
    example::store<FFT::elements_per_thread, cufftdx::size_of<FFT>::value, FFT::stride>(thread_data, smem_b);

    // Compute C := alpha * A @ FFT(B, axis=0) + beta * C.

    // Load A and C from global to shared memory, B is already in shared memory after the FFT
    auto a_global_tensor = cublasdx::make_tensor(a, BLAS::get_layout_gmem_a());
    auto c_global_tensor = cublasdx::make_tensor(c, BLAS::get_layout_gmem_c());

    auto a_shared_tensor = cublasdx::make_tensor(smem_a, BLAS::get_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, BLAS::get_layout_smem_b());
    auto c_shared_tensor = cublasdx::make_tensor(smem_c, BLAS::get_layout_smem_c());

    using alignment = cublasdx::alignment_of<BLAS>;
    cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<BLAS, alignment::c>(c_global_tensor, c_shared_tensor);
    cublasdx::copy_wait();

    // Execute GEMM: C = alpha * A @ FFT(B, axis=0) + beta * C.
    BLAS().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);
    __syncthreads();

    // Store the results.
    auto out_global_tensor = cublasdx::make_tensor(output, BLAS::get_layout_gmem_c());
    cublasdx::copy<BLAS, alignment::c>(c_shared_tensor, out_global_tensor);
}

// In this example cuBLASDx and cuFFTDx libraries are combined to perform GEMM and FFT in one pipeline for complex half-precision
// data. A key goal is to illustrate the differences in how cuFFTDx and cuBLASDx handle complex half-precision.
//
// The kernel computes the following operations:
//     1) FFT(B, axis=0)
//     2) C = alpha * A @ B + beta * C
//
// This sequence of operations can represent reweighting of signals in the frequency domain, for example.
//
// We first load B from global memory into registers, interleaving data from the two batches in RRII format. The FFT operation is
// implicitly batched in this case, with the two batches being processed in one execution. The results are then stored into shared
// memory in preparation for the GEMM, with the data being converted back to RIRI format. Next the matrices A and C are loaded
// into shared memory, and the GEMM (which is in fact a batched GEMV) is executed. The results are stored back to global memory.
//
// Important notes:
// * Results are verified against cuFFT and cuBLAS.
// * This example is written only for a batch size of 2 and for complex half-precision type.
// * The type used by cuFFTDx is complex<__half2> while the type used by cuBLASDx is complex<half>.
// * Shared memory required by the kernel is the max of the amount required by FFT and GEMM.
template<unsigned int Arch>
int gemm_fft_fp16() {
    using precision_type = __half;
    constexpr unsigned int m = 64;
    constexpr unsigned int n = batch_size;
    constexpr unsigned int k = m;

    static_assert(batch_size == 2, "This example only supports a batch size of 2.");
    static_assert(std::is_same_v<precision_type, __half>, "This example only supports half-precision.");

    using FFT  = decltype(cufftdx::Block() + cufftdx::Size<k>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
                          cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::Precision<precision_type>() +
                          cufftdx::ElementsPerThread<2>() + cufftdx::FFTsPerBlock<batch_size>() + cufftdx::SM<Arch>());

    using BLAS = decltype(cublasdx::Size<m, n, k>() +
                          cublasdx::Precision<precision_type>() +
                          cublasdx::Type<cublasdx::type::complex>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<FFT::block_dim.x>() +
                          cublasdx::SM<Arch>());

    #if CUBLASDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using fft_complex_type = example::value_type_t<FFT>;
    using blas_complex_type = example::uniform_value_type_t<BLAS>;
    #else
    using fft_complex_type = typename FFT::value_type;
    using blas_complex_type = example::uniform_value_type_t<BLAS>;
    #endif
    using complex_type = blas_complex_type;

    // Check that FFT matches GEMM dimensions.
    static_assert(cufftdx::size_of<FFT>::value == cublasdx::size_of<BLAS>::k,
                  "FFT must have the same size as the GEMM k dimension");
    // Checking that block dims match
    static_assert((FFT::block_dim.x == BLAS::block_dim.x) && (FFT::block_dim.y == BLAS::block_dim.y) &&
                  (FFT::block_dim.z == BLAS::block_dim.z),
                  "FFT must require the same CUDA block dimenions as GEMM");

    // Allocate managed memory.
    complex_type* buffer;
    complex_type* a;
    complex_type* b;
    complex_type* c;
    complex_type* output;

    constexpr auto global_a_size = example::global_memory_size_of<BLAS>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<BLAS>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<BLAS>::c_size;

    auto          size = (global_a_size + // a
                          global_b_size + // b
                          global_c_size + // c
                          global_c_size   // output
                         );
    auto          size_bytes = size * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&buffer, size_bytes));
    a                = buffer;
    b                = a + global_a_size;
    c                = b + global_b_size;
    output           = c + global_c_size;

    complex_type alpha = {float(1), float(0)};
    complex_type beta  = {float(0), float(0)};

    // Fill the a, b, c matrices.
    {
        float base = cublasdx::size_of<BLAS>::m * cublasdx::size_of<BLAS>::k;
        for (size_t i = 0; i < global_a_size; i++) {
            a[i] = complex_type {float(i) / base, float(i) / base};
        }
        for (size_t i = 0; i < global_b_size; i++) {
            b[i] = complex_type {float(i) / base, float(i) / base};
        }
        for (size_t i = 0; i < global_c_size; i++) {
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
    constexpr auto shared_memory_size = std::max({FFT::shared_memory_size, BLAS::shared_memory_size});
    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        gemm_fft_fp16_kernel<FFT, BLAS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size));

    // Invokes cuBLASDx+cuFFTDx kernel with FFT::block_dim threads in CUDA block
    gemm_fft_fp16_kernel<FFT, BLAS><<<1, FFT::block_dim, shared_memory_size, stream>>>(a, b, c, alpha, beta, output);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Compute reference results using cuBLAS and cuFFT (with float32 precision).
    using reference_complex_type = cublasdx::complex<float>;
    auto  reference_size_bytes   = size * sizeof(reference_complex_type);

    reference_complex_type* reference_buffer;
    reference_complex_type* reference_a;
    reference_complex_type* reference_b;
    reference_complex_type* reference_c;
    reference_complex_type* reference_output;

    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&reference_buffer, reference_size_bytes));
    reference_a                = reference_buffer;
    reference_b                = reference_a + global_a_size;
    reference_c                = reference_b + global_b_size;
    reference_output           = reference_c + global_c_size;

    reference_complex_type reference_alpha{alpha.real(), alpha.imag()};
    reference_complex_type reference_beta{beta.real(), beta.imag()};

    // Copy a, b, and c to the corresponding reference data buffers.
    copy(a, reference_a, global_a_size);
    copy(b, reference_b, global_b_size);
    copy(c, reference_c, global_c_size);

    // cuBLAS+cuFFT
    reference<FFT, BLAS>(reference_a, reference_b, reference_c, reference_alpha, reference_beta, reference_output, stream);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Print results.
    std::cout << std::fixed << std::showpos << std::setprecision(4);
    std::cout << "[cuBLASDx + cuFFTDx] (float16):\n";
    std::cout << "     Batch 1             Batch 2   \n";
    for (size_t i = 0; i <  m; i++) {
        for (size_t j = 0; j < n; j++) {
            auto index = i + j * m;
            std::cout << "[" << float(output[index].real()) << ", " << float(output[index].imag()) << "]  ";
        }
        std::cout << "\n";
    }
    std::cout << "[cuBLAS + cuFFT] (float32):\n";
    std::cout << "     Batch 1             Batch 2   \n";
    for (size_t i = 0; i <  m; i++) {
        for (size_t j = 0; j < n; j++) {
            auto index = i + j * m;
            std::cout << "[" << float(reference_output[index].x) << ", " << float(reference_output[index].y) << "]  ";
        }
        std::cout << "\n";
    }

    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(buffer));
    CUDA_CHECK_AND_EXIT(cudaFree(reference_buffer));

    std::cout << "Success" << std::endl;
    return 0;
}

template<unsigned int Arch>
struct gemm_fft_fp16_functor {
    int operator()() { return gemm_fft_fp16<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<gemm_fft_fp16_functor>();
}
