#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include <cublasdx.hpp>
#include <cufftdx.hpp>

#include "common.hpp"
#include "block_io.hpp"
#include "reference.hpp"

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

// Uncomment to enable correctness mode and verify correctness of the results
// #define CORRECTNESS 1

#ifdef CORRECTNESS
constexpr unsigned int repeats = 1;
constexpr unsigned int warm_up_repeats = 0;
#else
constexpr unsigned int repeats = 100;
constexpr unsigned int warm_up_repeats = 5;
#endif

template <typename T>
struct custom_unary_op
{
    __host__ __device__
    T operator()(T x) const
    {
        return x * x;
    }
};

template<class FFT, class GEMM, class ValueType = example::uniform_value_type_t<GEMM>>
double measure_cublas_cufft(ValueType*         a,
                            ValueType*         b,
                            ValueType*         c,
                            const ValueType    alpha,
                            const ValueType    beta,
                            ValueType*         output,
                            const unsigned int batch_size,
                            cudaStream_t       stream) {
    constexpr auto m = cublasdx::size_of<GEMM>::m;
    constexpr auto n = cublasdx::size_of<GEMM>::n;
    constexpr auto k = cublasdx::size_of<GEMM>::k;

    // Prepare cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK_AND_EXIT(cublasCreate(&handle));
    CUBLAS_CHECK_AND_EXIT(cublasSetStream(handle, stream));
    constexpr bool is_a_transposed = (cublasdx::arrangement_of<GEMM>::a == cublasdx::row_major);
    constexpr bool is_b_transposed = (cublasdx::arrangement_of<GEMM>::b == cublasdx::row_major);
    const auto a_transpose = example::detail::get_cublas_transpose_mode(cublasdx::arrangement_of<GEMM>::a);
    const auto b_transpose = example::detail::get_cublas_transpose_mode(cublasdx::arrangement_of<GEMM>::b);
    static_assert(cublasdx::arrangement_of<GEMM>::c == cublasdx::arrangement::col_major, "Only column-major C matrix supported");

    constexpr auto global_a_size = example::global_memory_size_of<GEMM>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<GEMM>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<GEMM>::c_size;

    // Prepare cuFFT
    const unsigned int fft_size = global_c_size;

    cufftHandle plan;
    CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan, fft_size, CUFFT_C2C, batch_size));
    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan, stream));

    // Create unary operation for transform
    custom_unary_op<ValueType> unary_op;

    #if (THRUST_VERSION >= 101600)
    auto execution_policy = thrust::cuda::par_nosync.on(stream);
    #else
    auto execution_policy = thrust::cuda::par.on(stream)
    #endif

    double time = example::measure::execution(
        [&](cudaStream_t) {
            // Transform output
            thrust::transform(execution_policy,
                              thrust::device_pointer_cast(a),
                              thrust::device_pointer_cast(a) + global_a_size * batch_size,
                              thrust::device_pointer_cast(a),
                              unary_op);
            // Run cuBLAS
            CUBLAS_CHECK_AND_EXIT(cublasCgemmStridedBatched(handle,
                                                            a_transpose,
                                                            b_transpose,
                                                            m,
                                                            n,
                                                            k,
                                                            reinterpret_cast<const cuFloatComplex*>(&alpha),
                                                            reinterpret_cast<const cuFloatComplex*>(a),
                                                            is_a_transposed ? k : m,
                                                            global_a_size,
                                                            reinterpret_cast<const cuFloatComplex*>(b),
                                                            is_b_transposed ? n : k,
                                                            global_b_size,
                                                            reinterpret_cast<const cuFloatComplex*>(&beta),
                                                            reinterpret_cast<cuFloatComplex*>(c),
                                                            m,
                                                            global_c_size,
                                                            batch_size));
            // Run cuFFT
            CUFFT_CHECK_AND_EXIT(cufftExecC2C(
                plan, reinterpret_cast<cufftComplex*>(c), reinterpret_cast<cufftComplex*>(output), CUFFT_FORWARD));
            // Transform output
            thrust::transform(execution_policy,
                              thrust::device_pointer_cast(output),
                              thrust::device_pointer_cast(output) + global_c_size * batch_size,
                              thrust::device_pointer_cast(output),
                              unary_op);
        },
        warm_up_repeats, repeats, stream);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUBLAS_CHECK_AND_EXIT(cublasDestroy(handle));
    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan));

    return time;
}

// Batch offset in a CUDA block (not global offset)
template<class FFT>
inline __device__ unsigned int batch_offset(const unsigned int local_fft_id,
                                            const unsigned int ffts_per_block = blockDim.y) {
    return ffts_per_block == 1 ? 0 : (cufftdx::size_of<FFT>::value * local_fft_id);
}

template<class FFT, class GEMM, class ValueType = example::uniform_value_type_t<GEMM>>
__launch_bounds__(FFT::max_threads_per_block) __global__ void gemm_fft_kernel(const ValueType* a,
                                                                              const ValueType* b,
                                                                              const ValueType* c,
                                                                              const ValueType  alpha,
                                                                              const ValueType  beta,
                                                                              ValueType*       output) {
    #if CUBLASDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using blas_complex_type = example::uniform_value_type_t<GEMM>;
    using fft_complex_type  = example::value_type_t<FFT>;
    #else
    using blas_complex_type = example::uniform_value_type_t<GEMM>;
    using fft_complex_type  = typename FFT::value_type;
    #endif
    static_assert(std::is_same_v<fft_complex_type, blas_complex_type>, "BLAS and FFT complex type should match");

    using complex_type = blas_complex_type;
    using value_type = ValueType;
    constexpr unsigned int block_size = GEMM::block_dim.x * GEMM::block_dim.y * GEMM::block_dim.z;

    extern __shared__ complex_type smem[];

    constexpr auto global_a_size = example::global_memory_size_of<GEMM>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<GEMM>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<GEMM>::c_size;

    // cuBLASDx
    constexpr auto m = cublasdx::size_of<GEMM>::m;
    constexpr auto n = cublasdx::size_of<GEMM>::n;
    constexpr auto k = cublasdx::size_of<GEMM>::k;

    // Select batch for this CUDA block
    const value_type* batch_a      = a      + (blockIdx.x * global_a_size);
    const value_type* batch_b      = b      + (blockIdx.x * global_b_size);
    const value_type* batch_c      = c      + (blockIdx.x * global_c_size);
    value_type*       batch_output = output + (blockIdx.x * global_c_size);

    auto a_global_tensor = cublasdx::make_tensor(batch_a, GEMM::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(batch_b, GEMM::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(batch_c, GEMM::get_layout_gmem_c());

    auto [smem_a, smem_b, smem_c] = GEMM::slice_shared_memory(reinterpret_cast<char*>(smem));
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
    auto c_shared_tensor = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

    // Load a, b, c from global to shared memory
    using alignment = cublasdx::alignment_of<GEMM>;
    cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy<GEMM, alignment::c>(c_global_tensor, c_shared_tensor);
    cublasdx::copy_wait();

    // Transform A
    custom_unary_op<complex_type> unary_op;
    auto transformer = [unary_op](int i, complex_type v) {
        int r = i % cublasdx::leading_dimension_of<GEMM>::a;
        return r < cublasdx::size_of<GEMM>::m ? unary_op(v) : v;
    };
    example::transform(smem_a, GEMM::a_size, transformer);

    // Execute GEMM
    GEMM().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);
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

    // Transform and save results
    index = offset + threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
            batch_output[index] = unary_op(thread_data[i]);
            index += stride;
        }
    }
}

template<class FFT, class GEMM, class ValueType = typename GEMM::value_type>
double measure_cublasdx_cufftdx(const ValueType*   a,
                                const ValueType*   b,
                                ValueType*         c,
                                const ValueType    alpha,
                                const ValueType    beta,
                                ValueType*         output,
                                const unsigned int batch_size,
                                cudaStream_t       stream) {
    // Get max shared memory required by FFT and GEMM
    constexpr auto shared_memory_size = std::max({FFT::shared_memory_size, GEMM::shared_memory_size});
    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        gemm_fft_kernel<FFT, GEMM>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

    double time = example::measure::execution(
        [&](cudaStream_t stream) {
            gemm_fft_kernel<FFT, GEMM>
                <<<batch_size, FFT::block_dim, shared_memory_size, stream>>>(a, b, c, alpha, beta, output);
        },
        warm_up_repeats, repeats, stream);
    return time;
}

// In this example cuBLASDx and cuFFTDx libraries are combined to perform GEMM and FFT in one pipeline. Additionally, to showcase
// how fusing kernels can improve performance transforms operations are performed before and after GEMM and FFT. The kernel runs
// the following operations:
//     1) Transform(A)
//     2) C = alpha * A * B + beta * C
//     3) 1D FFT(C)
//     4) Transform(C)
//
// First the data of matrices A, B, and C is loaded into shared memory. Matrix A is transformed. After that the GEMM is
// executed, and 1D FFT is applied to the results (matrix C). The results are transformed and stored back to global memory.
//
// Also see example gemm_fft.cu.
template<unsigned int Arch>
int gemm_fft() {
#ifdef CORRECTNESS
    constexpr unsigned int m = 8;
    constexpr unsigned int n = 8;
    constexpr unsigned int k = 8;
    constexpr unsigned int block_size = 32;
#else
    constexpr unsigned int m = 16;
    constexpr unsigned int n = 16;
    constexpr unsigned int k = 16;
    constexpr unsigned int block_size = 128;
#endif
    constexpr unsigned int fft_size = m * n;
    static_assert((fft_size / block_size) >= 2, "Block size is too big");
    constexpr unsigned int fft_ept  = fft_size / block_size;

    using FFT = decltype(cufftdx::Block() + cufftdx::Size<fft_size>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
                         cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::Precision<float>() +
                         cufftdx::ElementsPerThread<fft_ept>() + cufftdx::FFTsPerBlock<1>() + cufftdx::SM<Arch>());

    using GEMM =
        decltype(cublasdx::Size<m, n, k>() +
                 cublasdx::Precision<float>() +
                 cublasdx::Type<cublasdx::type::complex>() +
                 cublasdx::Function<cublasdx::function::MM>() +
                 cublasdx::TransposeMode<cublasdx::transpose_mode::non_transposed,
                                         cublasdx::transpose_mode::non_transposed>() +
                 cublasdx::Block() +
                 cublasdx::BlockDim<FFT::block_dim.x, FFT::block_dim.y, FFT::block_dim.z>() +
                 cublasdx::SM<Arch>());

    #if CUBLASDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using blas_complex_type = example::uniform_value_type_t<GEMM>;
    using fft_complex_type = example::value_type_t<FFT>;
    #else
    using blas_complex_type = example::uniform_value_type_t<GEMM>;
    using fft_complex_type = typename FFT::value_type;
    #endif

    static_assert(std::is_same_v<blas_complex_type, fft_complex_type>, "BLAS and FFT complex types should be the same");
    using complex_type = blas_complex_type;

    // Checking that FFT matches GEMM output
    static_assert(cufftdx::size_of<FFT>::value == (GEMM::c_size),
                  "FFT must have the same size as C matrix (MxN)");
    // Checking that block dims match
    static_assert((FFT::block_dim.x == GEMM::block_dim.x) && (FFT::block_dim.y == GEMM::block_dim.y) &&
                      (FFT::block_dim.z == GEMM::block_dim.z),
                  "FFT must require the same CUDA block dimenions as GEMM");
    static_assert(sizeof(fft_complex_type) == sizeof(blas_complex_type),
                  "FFT::value_type matches as GEMM::value_type");

    constexpr auto global_a_size = example::global_memory_size_of<GEMM>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<GEMM>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<GEMM>::c_size;

    // Get single batch size
    auto single_batch_size = (global_a_size + // a
                              global_b_size + // b
                              global_c_size + // c
                              global_c_size + // output
                              global_c_size   // reference_output
    );
    auto single_batch_size_bytes = single_batch_size * sizeof(complex_type);

    #ifdef CORRECTNESS
    const unsigned int batches = 2;
    #else
    // Calculating parameters for scaling_kernel execution.
    // Get maximum number of running CUDA blocks per multiprocessor.
    int blocks_per_multiprocessor = 0;
    CUDA_CHECK_AND_EXIT(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor,
                                                      gemm_fft_kernel<FFT, GEMM>,
                                                      FFT::block_dim.x * FFT::block_dim.y * FFT::block_dim.y,
                                                      0));
    // Get maximum number of CUDA blocks running on all multiprocessors.
    // This many CUDA blocks will be run for simple_kernel.
    static constexpr unsigned int minimum_input_size_bytes = (1 << 30); // At least 1GB of data
    const unsigned int minimum_batches = (minimum_input_size_bytes - single_batch_size_bytes + 1) / single_batch_size_bytes;
    const unsigned int blocks_per_device = blocks_per_multiprocessor * example::get_multiprocessor_count();
    unsigned int batches = std::max({minimum_batches, blocks_per_device});
    batches = blocks_per_device * ((batches + (blocks_per_device) - 1) / (blocks_per_device));
    // batches = 1; // single batch test
    // batches = example::get_multiprocessor_count(); // few batches test
    #endif

    // Allocate memory for a, b, c
    complex_type* buffer;
    complex_type* a;
    complex_type* b;
    complex_type* c;
    complex_type* output;
    complex_type* reference_output;

    auto size_bytes = batches * single_batch_size_bytes;
    #ifdef CORRECTNESS
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&buffer, size_bytes));
    #else
    CUDA_CHECK_AND_EXIT(cudaMalloc(&buffer, size_bytes));
    #endif
    a                = buffer;
    b                = a + (batches * global_a_size);
    c                = b + (batches * global_b_size);
    output           = c + (batches * global_c_size);
    reference_output = output + (batches * global_a_size);

    // Se alpha and beta for GEMM
    complex_type alpha = example::make_value<complex_type>(1., 1.);
    complex_type beta  = example::make_value<complex_type>(1., 1.);

    #ifdef CORRECTNESS
    // Fill the a, b, c matrices
    {
        float base = global_c_size * cublasdx::size_of<GEMM>::k;
        for (size_t i = 0; i < batches * global_a_size; i++) {
            a[i] = example::make_value<complex_type>(float(i) / base, float(i) / base);
        }
        for (size_t i = 0; i < batches * global_b_size; i++) {
            b[i] = example::make_value<complex_type>(float(i) / base, float(i) / base);
        }
        for (size_t i = 0; i < batches * global_c_size; i++) {
            c[i] = example::make_value<complex_type>(float(1) / base, float(1) / base);
        }
        for (size_t i = 0; i < batches * global_c_size; i++) {
            output[i] = example::make_value<complex_type>(float(-1), float(-1));
        }
        for (size_t i = 0; i < batches * global_c_size; i++) {
            reference_output[i] = example::make_value<complex_type>(float(-1), float(-1));
        }
    }
    #endif

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    #ifdef CORRECTNESS
    // Prefetch memory to device
    {
        int device;
        CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));
        CUDA_CHECK_AND_EXIT(cudaMemPrefetchAsync(a, size_bytes, device, stream));
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    }
    #endif

    double time_cublasdx_cufftdx = measure_cublasdx_cufftdx<FFT, GEMM>(a, b, c, alpha, beta, output, batches, stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    double time_cublas_cufft = measure_cublas_cufft<FFT, GEMM>(a, b, c, alpha, beta, reference_output, batches, stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    #ifdef CORRECTNESS
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
    #endif

    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(buffer));

    // Report results.
    auto report_time_and_performance = [&](std::string name, double time) -> void {
        std::cout << name << std::endl;
        std::cout << "Avg Time [ms_n]: " << time / repeats << std::endl;
        std::cout << "Time (all) [ms_n]: " << time << std::endl;
    };

    std::cout << "GEMM: ";
    std::cout << cublasdx::size_of<GEMM>::m << " x " << cublasdx::size_of<GEMM>::n << " x " << cublasdx::size_of<GEMM>::k << "\n";
    std::cout << "FFT: " << cufftdx::size_of<FFT>::value << "\n";
    std::cout << "Batches: " << batches << "\n";
    report_time_and_performance("cuBLASDx+cuFFTDx", time_cublasdx_cufftdx);
    report_time_and_performance("cuBLAS+cuFFT", time_cublas_cufft);

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
