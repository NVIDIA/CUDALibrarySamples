#include <array>
#include <iostream>
#include <system_error>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "common.hpp"
#include "reference.hpp"

template<class BLAS,
         class GEMMShape,
         class Alpha,
         class AValueType,
         class BValueType,
         class Beta,
         class CValueType>
__launch_bounds__(BLAS::max_threads_per_block, 1)
__global__ void gemm_kernel(GEMMShape  const  gemm_shape,
                            Alpha      const  alpha,
                            AValueType const* a,
                            BValueType const* b,
                            Beta       const  beta,
                            CValueType      * c) {

    extern __shared__ __align__(16) char smem[];

    using alignment = cublasdx::alignment_of<BLAS>;

    // ================================
    // 1. PREPARE GLOBAL MEMORY TENSORS

    // Handle row-major symmetrically to col-major
    auto block_coord = example::get_block_coord<BLAS>();

    auto global_a = example::make_device_gmem_tensor_a<BLAS>(gemm_shape, a);
    auto global_b = example::make_device_gmem_tensor_b<BLAS>(gemm_shape, b);
    auto global_c = example::make_device_gmem_tensor_c<BLAS>(gemm_shape, c);

    // Get a row of tiles from A, containing K / tile_k stages
    auto tile_slice_a_gmem = example::get_block_tile_slice_a<BLAS>(global_a, block_coord);
    // Get a column of tiles from B containing K / tile_k stages
    auto tile_slice_b_gmem = example::get_block_tile_slice_b<BLAS>(global_b, block_coord);
    // Get a single tile from C
    auto tile_c_gmem = example::get_block_tile_c<BLAS>(global_c, block_coord);

    // ================================
    // 2. PREPARE SHARED MEMORY TENSORS

    // Slice shared memory for proper alignment in 2-stage pipelining
    auto [smem_a, smem_b, smem_a_n, smem_b_n] =
        cublasdx::slice_shared_memory_generic<AValueType, BValueType, AValueType, BValueType>(
            smem,
            cute::make_tuple(cublasdx::cosize(BLAS::suggest_layout_smem_a()), cublasdx::alignment_of_v_a<BLAS>),
            cute::make_tuple(cublasdx::cosize(BLAS::suggest_layout_smem_b()), cublasdx::alignment_of_v_b<BLAS>),
            cute::make_tuple(cublasdx::cosize(BLAS::suggest_layout_smem_a()), cublasdx::alignment_of_v_a<BLAS>),
            cute::make_tuple(cublasdx::cosize(BLAS::suggest_layout_smem_b()), cublasdx::alignment_of_v_b<BLAS>)
        );

    auto s_a = cublasdx::make_tensor(smem_a, BLAS::suggest_layout_smem_a());
    auto s_b = cublasdx::make_tensor(smem_b, BLAS::suggest_layout_smem_b());

    auto s_a_n = cublasdx::make_tensor(smem_a_n, BLAS::suggest_layout_smem_a());
    auto s_b_n = cublasdx::make_tensor(smem_b_n, BLAS::suggest_layout_smem_b());

    // ==================================
    // 3. PREPARE 2-STAGE MEMORY PIPELINE

    // Since both slices have an iteration dimension equal to number of necessary
    // GEMM stages, we can just use it for iteration
    const auto k_stages = cute::get<2>(cute::shape(tile_slice_a_gmem.layout()));

    // Schedule first stage into memory pipeline queue
    // cute::Int<X> is a static integer similar to std::integral_constant
    constexpr auto static_first_stage_index = cute::Int<0>{};
    cublasdx::copy<BLAS, alignment::a>(example::get_tile_from_slice(tile_slice_a_gmem, static_first_stage_index), s_a);
    cublasdx::copy<BLAS, alignment::b>(example::get_tile_from_slice(tile_slice_b_gmem, static_first_stage_index), s_b);

    // ==============================================
    // 4. EXECUTE GEMM WITH ACCUMULATION IN REGISTERS

    auto partitioner = BLAS().suggest_partitioner();
    auto c_frag = partitioner.make_accumulator_fragment();
    cublasdx::clear(c_frag);

    #pragma unroll 1
    for(int stage = 1; stage < k_stages; stage++) {
        // Wait for previous stage
        cublasdx::copy_wait();

        // Swap for next iteration
        cublasdx::copy<BLAS, alignment::a>(example::get_tile_from_slice(tile_slice_a_gmem, stage), s_a_n);
        cublasdx::copy<BLAS, alignment::b>(example::get_tile_from_slice(tile_slice_b_gmem, stage), s_b_n);

        // Accumulate results from this stage
        BLAS().execute(s_a, s_b, c_frag);

        example::swap(s_a_n, s_a);
        example::swap(s_b_n, s_b);
    }

    cublasdx::copy_wait();
    BLAS().execute(s_a, s_b, c_frag);

    // ===========
    // 5. EPILOGUE

    auto local_gmem = partitioner.partition_like_C(tile_c_gmem);

    if constexpr(partitioner.is_predicated()) {
        #pragma unroll
        for(int i = 0; i < cublasdx::size(c_frag); ++i) {
            if(partitioner.is_index_in_bounds(i)) {
                local_gmem(i) = alpha * c_frag(i) + beta * local_gmem(i);
            }
        }
    } else {
        cublasdx::axpby(alpha, c_frag, beta, local_gmem);
    }
}

template<class BLAS,
         class GEMMShape,
         class Alpha,
         class AValueType,
         class BValueType,
         class Beta,
         class CValueType>
auto measure_cublasdx(GEMMShape gemm_shape,
                      const Alpha       alpha,
                      const AValueType* a,
                      const BValueType* b,
                      const Beta        beta,
                      CValueType*       c,
                      unsigned int kernel_warm_up_repeats,
                      unsigned int kernel_repeats,
                      cudaStream_t      stream) {


    // Grid size configuration
    const auto m = cute::get<0>(gemm_shape);
    const auto n = cute::get<1>(gemm_shape);

    constexpr auto tile_m = cublasdx::size_of<BLAS>::m;
    constexpr auto tile_n = cublasdx::size_of<BLAS>::n;
    constexpr auto tile_k = cublasdx::size_of<BLAS>::k;

    // m may be dynamic at this point, it's statically checked in main function
    assert(m % tile_m == 0 and n % tile_n == 0);

    std::vector<CValueType> results(m * n);

    std::cout << "tile_m, tile_n, tile_k: " << tile_m << ", " << tile_n << ", " << tile_k << std::endl;
    std::cout << "Alignment: " << cublasdx::alignment_of<BLAS>::a << ", " << cublasdx::alignment_of<BLAS>::b << ", " << cublasdx::alignment_of<BLAS>::c << std::endl;
    std::cout << "ld: " << BLAS::lda << ", " << BLAS::ldb << ", " << BLAS::ldc << std::endl;
    std::cout << "block_size: " << BLAS::max_threads_per_block << std::endl;

    constexpr bool reverse_block_coord = (cublasdx::arrangement_of_v_a<BLAS> == cublasdx::row_major) and
                                         (cublasdx::arrangement_of_v_b<BLAS> == cublasdx::row_major);

    dim3 grid_dim = cute::conditional_return<reverse_block_coord>(dim3{n / tile_n, m / tile_m, 1}, dim3{(m / tile_m), (n / tile_n), 1});

    // Increase max dynamic shared memory for the kernel if needed.
    auto shared_memory_size =
        cublasdx::make_shared_storage_calc()
        .add(cublasdx::alignment_of_v_a<BLAS>, sizeof(AValueType), BLAS::get_layout_smem_a())
        .add(cublasdx::alignment_of_v_b<BLAS>, sizeof(BValueType), BLAS::get_layout_smem_b())
        .add(cublasdx::alignment_of_v_a<BLAS>, sizeof(AValueType), BLAS::get_layout_smem_a())
        .add(cublasdx::alignment_of_v_b<BLAS>, sizeof(BValueType), BLAS::get_layout_smem_b())
        .get();

    auto kernel = gemm_kernel<BLAS, GEMMShape, Alpha, AValueType, BValueType, Beta, CValueType>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

    auto run_cublasdx_gemm = [&](cudaStream_t str) {
        kernel<<<grid_dim, BLAS::block_dim, shared_memory_size, str>>>(gemm_shape, alpha, a, b, beta, c);
    };

    // First run for correctness check
    run_cublasdx_gemm(stream);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaMemcpy(results.data(), c, results.size() * sizeof(CValueType), cudaMemcpyDeviceToHost));

    // Execute kernel.
    double time = example::measure::execution(run_cublasdx_gemm, kernel_warm_up_repeats, kernel_repeats, stream);

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    return std::make_tuple(results, time);
}

template<class Precision, class ... Args>
inline void call_cublas(Args && ... args) {
    if constexpr(std::is_same_v<Precision, __half>) {
        CUBLAS_CHECK_AND_EXIT(cublasHgemm(args ...));
    } else if constexpr(std::is_same_v<Precision, float>) {
        CUBLAS_CHECK_AND_EXIT(cublasSgemm(args ...));
    } else if constexpr(std::is_same_v<Precision, double>) {
        CUBLAS_CHECK_AND_EXIT(cublasDgemm(args ...));
    }
}

template<class ValueType, class GEMMShape>
auto measure_cublas(GEMMShape gemm_shape,
                    cublasdx::arrangement arr_a,
                    cublasdx::arrangement arr_b,
                    cublasdx::arrangement arr_c,
                    ValueType        alpha,
                    ValueType const* a,
                    ValueType const* b,
                    ValueType        beta,
                    ValueType      * c,
                    unsigned int kernel_warm_up_repeats,
                    unsigned int kernel_repeats,
                    cudaStream_t     stream) {

    static_assert((not example::is_complex<ValueType>()) and (std::is_same_v<__half, ValueType> or
                                                             std::is_same_v<float, ValueType> or
                                                             std::is_same_v<double, ValueType>),
                  "only __half, float and double are supported for cuBLAS measurement.");

    const auto m = cute::get<0>(gemm_shape);
    const auto n = cute::get<1>(gemm_shape);
    const auto k = cute::get<2>(gemm_shape);

    std::vector<ValueType> results(m * n);

    cublasHandle_t handle;
    CUBLAS_CHECK_AND_EXIT(cublasCreate(&handle));
    cublasSetStream(handle, stream);

    auto run_cublas_gemm =
        [&](cudaStream_t) {
            bool is_reversed = arr_c == cublasdx::row_major;
            const auto m_dim = is_reversed ? n : m;
            const auto n_dim = is_reversed ? m : n;
            const auto k_dim = k;

            const bool a_col_major = (arr_a == cublasdx::col_major);
            const auto a_transpose = (a_col_major != is_reversed) ? CUBLAS_OP_N : CUBLAS_OP_T;
            const auto lda = a_col_major ? m : k;

            const bool b_col_major = (arr_b == cublasdx::col_major);
            const auto b_transpose = (b_col_major != is_reversed) ? CUBLAS_OP_N : CUBLAS_OP_T;
            const auto ldb = b_col_major ? k : n;

            const auto ldc = m_dim;

            const auto first_transpose = is_reversed ? b_transpose : a_transpose;
            const auto first_ld = is_reversed ? ldb : lda;
            const auto first_ptr = is_reversed ? b : a;

            const auto second_transpose = is_reversed ? a_transpose : b_transpose;
            const auto second_ld = is_reversed ? lda : ldb;
            const auto second_ptr = is_reversed ? a : b;

            // inputs/outputs are row-major, so we flip A&B and M&N
            call_cublas<ValueType>(handle,
                                   first_transpose, second_transpose,
                                   m_dim, n_dim, k_dim,
                                   reinterpret_cast<const ValueType*>(&alpha),
                                   reinterpret_cast<const ValueType*>(first_ptr),
                                   first_ld,
                                   reinterpret_cast<const ValueType*>(second_ptr),
                                   second_ld,
                                   reinterpret_cast<const ValueType*>(&beta),
                                   reinterpret_cast<ValueType*>(c),
                                   ldc);
        };

    // First run for correctness check
    run_cublas_gemm(stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaMemcpy(results.data(), c, results.size() * sizeof(ValueType), cudaMemcpyDeviceToHost));

    double time_cublas = example::measure::execution(run_cublas_gemm, kernel_warm_up_repeats, kernel_repeats, stream);

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    CUBLAS_CHECK_AND_EXIT(cublasDestroy(handle));

    return std::make_tuple(results, time_cublas);
}

template<unsigned int Arch, class GEMMShape>
int device_gemm_performance(GEMMShape gemm_shape) {

    // ==========================================================
    // Tile optimized for big SGEMMs on H100 (fp32), for best
    // results for other precisions it may require a
    // parameter scan over: (tile_m / tile_n / tile_k / threads)
    // and staging C loads / stores through shared memory

    const auto m = cute::get<0>(gemm_shape);
    const auto n = cute::get<1>(gemm_shape);
    const auto k = cute::get<2>(gemm_shape);

    constexpr unsigned int tile_m = 256;
    constexpr unsigned int tile_n = 128;
    constexpr unsigned int tile_k = 16;

    constexpr unsigned int threads = 256;

    const bool divisible = (m % tile_m == 0 and n % tile_n == 0 and k % tile_k == 0);
    if(not divisible) {
        std::cerr << "M, N, K dimensions must be divisible by tile_m, tile_n, tile_k" << std::endl;
        return 1;
    }

    // Compute precision and compute value type
    using compute_precision = float;

    // Decoupled input precision type
    using a_io_value_type = float;
    using b_io_value_type = float;
    using c_io_value_type = float;

    // Global memory arrangements
    constexpr auto arr_a = cublasdx::col_major;
    constexpr auto arr_b = cublasdx::col_major;
    constexpr auto arr_c = cublasdx::col_major;

    constexpr unsigned int cublasdx_alignment = 16;

    // Number type
    constexpr auto type = cublasdx::type::real;
    static_assert(type == cublasdx::type::real, "Only real type reference is supported");

    // Passed to cuBLAS
    using compute_value_type =
        std::conditional_t<type == cublasdx::type::real,
                           compute_precision,
                           cublasdx::complex<compute_precision>>;

    // Scalar multipliers
    compute_value_type alpha = example::make_value<compute_value_type>(1.1);
    compute_value_type beta  = example::make_value<compute_value_type>(1.2);

    // Performance comparison parameters
    const unsigned int kernel_repeats = 100;
    const unsigned int kernel_warm_up_repeats = 10;

    // Test implementation
    a_io_value_type *a_cublasdx;
    b_io_value_type *b_cublasdx;
    c_io_value_type *c_cublasdx;

    compute_value_type *a_cublas;
    compute_value_type *b_cublas;
    compute_value_type *c_cublas;

    CUDA_CHECK_AND_EXIT(cudaMalloc(&a_cublasdx, m * k * sizeof(a_io_value_type)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&a_cublas,   m * k * sizeof(compute_value_type)));

    CUDA_CHECK_AND_EXIT(cudaMalloc(&b_cublasdx, n * k * sizeof(b_io_value_type)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&b_cublas,   n * k * sizeof(compute_value_type)));

    CUDA_CHECK_AND_EXIT(cudaMalloc(&c_cublasdx, n * m * sizeof(c_io_value_type)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&c_cublas,   n * m * sizeof(compute_value_type)));

    // Fill the A, B, C matrices with random values.
    {
        auto host_a_io = example::get_random_data<a_io_value_type>(-0.1f, 0.1f, m * k);
        auto host_b_io = example::get_random_data<b_io_value_type>(-0.1f, 0.1f, k * n);
        auto host_c_io = example::get_random_data<c_io_value_type>(-0.1f, 0.1f, m * n);

        static_assert(std::is_convertible_v<a_io_value_type, compute_value_type>);
        auto host_a_compute = std::vector<compute_value_type>(host_a_io.begin(), host_a_io.end());

        static_assert(std::is_convertible_v<b_io_value_type, compute_value_type>);
        auto host_b_compute = std::vector<compute_value_type>(host_b_io.begin(), host_b_io.end());

        static_assert(std::is_convertible_v<c_io_value_type, compute_value_type>);
        auto host_c_compute = std::vector<compute_value_type>(host_c_io.begin(), host_c_io.end());

        CUDA_CHECK_AND_EXIT(cudaMemcpy(a_cublasdx, host_a_io.data(), m * k * sizeof(a_io_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(cudaMemcpy(b_cublasdx, host_b_io.data(), k * n * sizeof(b_io_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(cudaMemcpy(c_cublasdx, host_c_io.data(), m * n * sizeof(c_io_value_type), cudaMemcpyHostToDevice));

        CUDA_CHECK_AND_EXIT(cudaMemcpy(a_cublas, host_a_compute.data(), m * k * sizeof(compute_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(cudaMemcpy(b_cublas, host_b_compute.data(), k * n * sizeof(compute_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(cudaMemcpy(c_cublas, host_c_compute.data(), m * n * sizeof(compute_value_type), cudaMemcpyHostToDevice));
        // destroy host vectors
    }

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream))

    // cuBLASDx type creation
    using BLAS = decltype(cublasdx::Size<tile_m, tile_n, tile_k>() +
                          cublasdx::Precision<compute_precision>() +
                          cublasdx::Type<type>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<arr_a, arr_b, arr_c>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<threads>() +
                          cublasdx::Alignment<cublasdx_alignment, cublasdx_alignment, cublasdx_alignment>() +
                          cublasdx::experimental::StaticBlockDim() + // Experimental: Runtime block dim is equal to operator block dim
                          cublasdx::SM<Arch>());

    auto [host_dx_results, time_cublasdx] =
        measure_cublasdx<BLAS>(gemm_shape, alpha, a_cublasdx, b_cublasdx, beta, c_cublasdx, kernel_warm_up_repeats, kernel_repeats, stream);

    // Measure cuBLAS performance.
    auto [host_blas_results, time_cublas] =
        measure_cublas(gemm_shape, arr_a, arr_b, arr_c, alpha, a_cublas, b_cublas, beta, c_cublas, kernel_warm_up_repeats, kernel_repeats, stream);

    // Write performance data.
    using cublasdx::size_of;
    std::cout << "m, n, k: " << m << ", " << n << ", " << k
              << std::endl;
    std::cout << "Compute Type: " << example::type_string<compute_value_type>() << std::endl;
    std::cout << "Precision: " << example::precision_string<compute_precision>() << std::endl;
    std::cout << "Dx Input Precision A: " << example::precision_string<a_io_value_type>() << std::endl;
    std::cout << "Dx Input Precision B: " << example::precision_string<b_io_value_type>() << std::endl;
    std::cout << "Dx Input Precision C: " << example::precision_string<c_io_value_type>() << std::endl;

    std::cout << "\ncuBLASDx\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Avg time [ms]  = " << time_cublasdx / kernel_repeats << "\n";

    std::cout << "\ncuBLAS\n";
    std::cout << "Avg time [ms]  = " << time_cublas / kernel_repeats << "\n";

    double error = 0;
    double norm = 0;
    for(int i = 0; i < host_blas_results.size(); i++) {
        double ref = static_cast<double>(host_blas_results[i]);
        double res = static_cast<double>(host_dx_results[i]);
        error += std::norm(ref - res);
        norm += std::norm(ref);
    }
    std::cout << "Error = " << sqrt(error/norm) << "\n";

    std::cout << "cuBLAS / cuBLASDx timings = " << time_cublas / time_cublasdx << "\n";

    // Free resources.
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(a_cublasdx));
    CUDA_CHECK_AND_EXIT(cudaFree(b_cublasdx));
    CUDA_CHECK_AND_EXIT(cudaFree(c_cublasdx));
    CUDA_CHECK_AND_EXIT(cudaFree(a_cublas));
    CUDA_CHECK_AND_EXIT(cudaFree(b_cublas));
    CUDA_CHECK_AND_EXIT(cudaFree(c_cublas));

    return 0;
}

struct device_gemm_performance_functor {

    template<int Arch, class GemmShape>
    int operator()(std::integral_constant<int, Arch>, GemmShape gemm_shape) {
        return device_gemm_performance<Arch>(gemm_shape);
    }
};

int main(int argc, char** argv) {
    std::array<unsigned int, 3> mnk = {8192, 8192, 8192};
    auto usage = []() { std::cerr << "Incorrect usage: ./device_gemm_performance [m n k]" << std::endl; };

    if(argc == 4) {
        std::cout << "Tile optimized for big SGEMMs on H100 (fp32), for best "
                     "results for other precisions it may require a "
                     "parameter scan over: (tile_m / tile_n / tile_k / threads) "
                     "and staging C loads / stores through shared memory" << std::endl;

        try {
            std::transform(argv + 1, argv + argc, mnk.begin(), [&](char* dim_input){ return std::stoul(dim_input);});
        } catch(...) {
            usage(); return 1;
        }
    } else if(argc != 1) {
        usage(); return 1;
    }

    return example::sm_runner(device_gemm_performance_functor{}, cute::make_shape(mnk[0], mnk[1], mnk[2]));
}
