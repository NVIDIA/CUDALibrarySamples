#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cublasdx.hpp>
#include <cute/tensor.hpp>

#include "block_io.hpp"
#include "common.hpp"
#include "reference.hpp"

#if defined(__CUDACC_RELAXED_CONSTEXPR__) && (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 7)))
#error --expt-relaxed-constexpr flag can not be specified with CUDA <11.7 with this example as it triggers NVCC bug
#endif

template <unsigned int Size, unsigned int BlockSize, class TensorFrom, class TensorTo,
          std::enable_if_t<cute::is_tensor<TensorFrom>::value && cute::is_tensor<TensorTo>::value, void> * = nullptr>
__device__ __forceinline__
void copy(TensorFrom &from, TensorTo &to) {
    constexpr unsigned int ept   = (Size + (BlockSize - 1)) / BlockSize;
    unsigned int           index = threadIdx.x;
    #pragma unroll
    for (unsigned int i = 0; i < ept; i++) {
        to(index) = from(index);
        index += BlockSize;
    }
}

// Helper functions to create pairs representing shape, coordinates, etc considering the transpose mode.
template <cublasdx::transpose_mode TransposeMode, class Maker,
          typename std::enable_if<TransposeMode == cublasdx::transpose_mode::non_transposed, bool>::type = true>
__host__ __device__ __forceinline__
cute::tuple<unsigned int, unsigned int> make(unsigned int m, unsigned int n, Maker maker) {
    return maker(m, n);
}

template <cublasdx::transpose_mode TransposeMode, class Maker,
          typename std::enable_if<TransposeMode == cublasdx::transpose_mode::transposed || TransposeMode == cublasdx::transpose_mode::conj_transposed , bool>::type = true>
__host__ __device__ __forceinline__
cute::tuple<unsigned int, unsigned int> make(unsigned int m, unsigned int n, Maker maker) {
    return maker(n, m);
}

template <class BlockMM, class MatrixA, class MatrixB, class MatrixC, class ValueType = typename BlockMM::value_type>
__launch_bounds__(BlockMM::max_threads_per_block) __global__
void block_mm_kernel(MatrixA gA, MatrixB gB, MatrixC gC,
                     ValueType alpha,
                     ValueType beta) {
    using value_type = ValueType;

    // We will partition A along the m-mode into M blocks of size m, B along the n-mode into N blocks of size n,
    // and C along both m- and n-modes into M * N blocks of size m x n. Each of the M * N CUDA blocks computes
    // its block of matrix C: C_mxn = Op(A)_mxk @ Op(B)_kxn, where Op performs transpose or conjugate transpose.

    auto make_shape  = cute::make_shape<unsigned int, unsigned int>;
    auto make_coord  = cute::make_coord<unsigned int, unsigned int>;
    auto make_step   = cute::make_step<unsigned int, unsigned int>;

    constexpr auto a_transpose_mode = cublasdx::transpose_mode_of<BlockMM>::a_transpose_mode;
    constexpr auto b_transpose_mode = cublasdx::transpose_mode_of<BlockMM>::b_transpose_mode;

    // Create the blocks for partitioning A, B, and C.
    auto blockA = make_layout(make<a_transpose_mode>(gridDim.x, 1, make_shape));
    auto blockB = make_layout(make<b_transpose_mode>(1, gridDim.y, make_shape));
    auto blockC = make_layout(cute::make_shape(gridDim.x, gridDim.y));

    constexpr unsigned int m = cublasdx::size_of<BlockMM>::m;
    constexpr unsigned int n = cublasdx::size_of<BlockMM>::n;
    constexpr unsigned int k = cublasdx::size_of<BlockMM>::k;

    auto idxA =  crd2idx(make<a_transpose_mode>(blockIdx.x, 0, make_coord), blockA);
    auto idxB =  crd2idx(make<b_transpose_mode>(0, blockIdx.y, make_coord), blockB);
    auto idxC =  crd2idx(cute::make_coord(blockIdx.x, blockIdx.y), blockC);

    // Get the local partitions of A, B, and C for the current block.
    auto pA = cute::local_partition(gA, blockA, idxA, make<a_transpose_mode>(m, k, make_step));
    auto pB = cute::local_partition(gB, blockB, idxB, make<b_transpose_mode>(k, n, make_step));
    auto pC = cute::local_partition(gC, blockC, idxC, cute::make_step(m, n));

    extern __shared__ value_type shared_mem[];
    value_type *smem_a = shared_mem;
    value_type *smem_b = smem_a + BlockMM::a_size;
    value_type *smem_c = smem_b + BlockMM::b_size;

    // Create shared memory tensors.
    auto sA = cute::make_tensor(cute::make_smem_ptr(smem_a), make<a_transpose_mode>(m, k, make_shape));
    auto sB = cute::make_tensor(cute::make_smem_ptr(smem_b), make<b_transpose_mode>(k, n, make_shape));
    auto sC = cute::make_tensor(cute::make_smem_ptr(smem_c), cute::make_shape(m, n));

    // Copy A, B, and C from global to shared memory.
    copy<BlockMM::a_size, BlockMM::block_dim.x>(pA, sA);
    copy<BlockMM::b_size, BlockMM::block_dim.x>(pB, sB);
    copy<BlockMM::c_size, BlockMM::block_dim.x>(pC, sC);

    __syncthreads();

    // Execute GEMM
    BlockMM().execute(alpha, smem_a, smem_b, beta, smem_c);

    __syncthreads();

    // Copy C from shared to global memory.
    copy<BlockMM::c_size, BlockMM::block_dim.x>(sC, pC);

    __syncthreads();
}

template<class BlockMM, class GlobalMM, unsigned int Arch, unsigned int BlockSize>
int benchmark_multiblock_gemm(const cudaStream_t& stream, bool verbose = false) {
    using namespace cublasdx;

    static constexpr unsigned int kernel_repeats = 5;
    static constexpr unsigned int warm_up_runs   = 5;

    constexpr bool set_block_size{BlockSize > 0};
    using block_mm_type = std::conditional_t<set_block_size, decltype(BlockMM() + BlockDim<BlockSize>()), BlockMM>;
    using value_type = typename block_mm_type::value_type;

    constexpr auto M = cublasdx::size_of<GlobalMM>::m;
    constexpr auto N = cublasdx::size_of<GlobalMM>::n;
    constexpr auto K = cublasdx::size_of<GlobalMM>::k;
    constexpr auto a_size = GlobalMM::a_size;
    constexpr auto b_size = GlobalMM::b_size;
    constexpr auto c_size = GlobalMM::c_size;

    // Allocate device memory for A, B, C.
    value_type* inputs;

    auto inputs_size       = a_size + b_size + c_size;
    auto inputs_size_bytes = inputs_size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&inputs, inputs_size_bytes));

    value_type* a     = inputs;
    value_type* b     = a + a_size;
    value_type* c     = b + b_size;

    value_type  alpha = example::make_value<value_type>(1.f);
    value_type  beta  = example::make_value<value_type>(0.f);

    // Fill the A, B, C matrices with random values.
    auto host_a = example::get_random_data<value_type>(0.1, 1.0, a_size);
    auto host_b = example::get_random_data<value_type>(0.1, 1.0, b_size);
    auto host_c = example::get_random_data<value_type>(0.1, 1.0, c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, host_a.data(), a_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(b, host_b.data(), b_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(c, host_c.data(), c_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    auto make_shape  = cute::make_shape<unsigned int, unsigned int>;

    constexpr auto a_transpose_mode = cublasdx::transpose_mode_of<BlockMM>::a_transpose_mode;
    constexpr auto b_transpose_mode = cublasdx::transpose_mode_of<BlockMM>::b_transpose_mode;

    // Wrap the device memory as CuTe tensors.
    auto tensor_a = cute::make_tensor(cute::make_gmem_ptr(a), make<a_transpose_mode>(M, K, make_shape));
    auto tensor_b = cute::make_tensor(cute::make_gmem_ptr(b), make<b_transpose_mode>(K, N, make_shape));
    auto tensor_c = cute::make_tensor(cute::make_gmem_ptr(c), cute::make_shape(M, N));

    using a_type = decltype(tensor_a);
    using b_type = decltype(tensor_b);
    using c_type = decltype(tensor_c);
    // Increase max dynamic shared memory for the kernel if needed.
    CUDA_CHECK_AND_EXIT(
        cudaFuncSetAttribute(block_mm_kernel<block_mm_type, a_type, b_type, c_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, block_mm_type::shared_memory_size));

    // Create a grid of size (M / m,  N / n).
    dim3 grid{M / cublasdx::size_of<block_mm_type>::m, N / cublasdx::size_of<block_mm_type>::n};
    // Measure performance of N trials.
    double time = example::measure::execution(
        [&](cudaStream_t stream) {
            block_mm_kernel<block_mm_type>
                <<<grid, block_mm_type::block_dim, block_mm_type::shared_memory_size, stream>>>(
                    tensor_a, tensor_b, tensor_c, alpha, beta);
        },
        warm_up_runs,
        kernel_repeats,
        stream);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    double avg_time = time / kernel_repeats;

    double gflops = example::gemm_flops<value_type>(M, N, K) / avg_time / 1000000.;

    // Copy results back to host.
    std::vector<value_type> host_output(c_size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(host_output.data(), c, c_size * sizeof(value_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Free device memory.
    CUDA_CHECK_AND_EXIT(cudaFree(inputs));

    if (verbose) {
        std::cout << "Global M, N, K: " << size_of<GlobalMM>::m << ", " << size_of<GlobalMM>::n << ", " << size_of<GlobalMM>::k
                  << std::endl;
        std::cout << "Block m, n, k: " << size_of<block_mm_type>::m << ", " << size_of<block_mm_type>::n << ", " << size_of<block_mm_type>::k
                  << std::endl;
        std::cout << "Type: " << example::type_string<value_type>() << std::endl;
        std::cout << "Precision: " << example::precision_string<value_type>() << std::endl;
        std::cout << "Block size: " << block_mm_type::block_dim.x << std::endl;
        std::cout << "Grid dimensions: " << grid.x << ", " << grid.y << ", " << grid.z << std::endl;
        std::cout << "Shared memory: " << block_mm_type::shared_memory_size << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Avg time [ms]: " << avg_time << std::endl;
        std::cout << "Time (all) [ms]: " << time << std::endl;
        std::cout << "Performance [GFLOPS]: " << gflops << std::endl;
    } else {
        std::cout << "(" << size_of<GlobalMM>::m << ", " << size_of<GlobalMM>::n << ", " << size_of<GlobalMM>::k << ") "
                  << example::precision_string<value_type>() << " precision " <<  example::type_string<value_type>()
                  << ": " << std::fixed << std::setprecision(4) << gflops << " GFLOPS, " << avg_time << " ms." << std::endl;
    }
    // Calculate reference solution.
    const auto [lda, ldb, ldc] = cublasdx::leading_dimension_of_v<GlobalMM>;
    auto reference_host_output = example::reference_gemm<GlobalMM>(alpha, host_a, lda, host_b, ldb, beta, host_c, ldc);

    // Check against reference (requires beta = 0. to avoid accumulating into C due to the repeated runs on the GPU).
    if (example::check(host_output, reference_host_output)) {
        std::cout << "The results are verified to be correct." << std::endl;
        return 0;
    }
    std::cout << "Failure" << std::endl;
    return 1;
}

// This example illustrates how to use cuBLASDx for matrix multiplication when the matrices do not fit into shared memory.
//
//       C = alpha * op(A) * op(B) + beta * C
//
// Let us denote the "global" matrix sizes along the m-, n-, and k-modes by M, N, and K. We will partition the matrix C along
// the m- and n-modes into blocks of size m and n respectively, resulting in M / m and N / n blocks along each mode. Similarly,
// matrix A will be partitioned along the m-mode into M / m blocks, while matrix B will be partitioned along the n-mode into
// N / n blocks. We will then launch M / m * N / n thread blocks, with each thread block computing a single m x n block of C.
//
// We will use CuTe APIs based on hierarchical layout algebra to elegantly specify the partitioning.
//
template<unsigned int Arch>
int multiblock_gemm() {
    using namespace cublasdx;

    // Parameters M, N, K define the dimensions of matrices A, B, and C. In this example, we choose K to be small enough
    // so that a local block fits into shared memory since we don't partition A and B along the k-mode. Such "tall and
    // skinny" matrices can represent the query, key, or value matrices in an attention operation, as an example.
    constexpr unsigned int M = 2048;
    constexpr unsigned int N = 2048;
    constexpr unsigned int K = 64;

    // Parameters m, n, k define the dimensions of the local blocks of matrices A, B, and C.
    constexpr unsigned int m = 64;
    constexpr unsigned int n = 64;
    constexpr unsigned int k = K;

    static_assert(M % m == 0 && N % n == 0 && K == k, "Error: the global matrix dimensions must be a multiple of the local matrix block dimensions.");

    // Choose block size, or set to 0 to use library-suggested value.
    constexpr unsigned int BlockSize = 0;

    // Choose precision (__half, float, double) and type (real or complex).
    using precision = __half;
    constexpr auto type = cublasdx::type::complex;

    // Choose transpose mode for A and B: non_transposed, transposed, or conj_transposed.
    constexpr auto a_transpose_mode = cublasdx::transpose_mode::non_transposed;
    constexpr auto b_transpose_mode = cublasdx::transpose_mode::conj_transposed;

    // Define the local matrix multiplication operation.
    using BlockMM  = decltype(cublasdx::Size<m, n, k>() +
                       cublasdx::Precision<precision>() +
                       cublasdx::Type<type>() +
                       cublasdx::Function<cublasdx::function::MM>() +
                       cublasdx::TransposeMode<a_transpose_mode, b_transpose_mode>() +
                       cublasdx::Block() +
                       cublasdx::SM<Arch>());
    // The global matrix multiplication operation provides a convenient way to encapsulate data of interest: (M, N, K),
    // matrix sizes, leading dimensions, etc. It cannot be executed since the problem is too large to fit into shared memory.
    using GlobalMM = decltype(cublasdx::Size<M, N, K>() +
                       cublasdx::Precision<precision>() +
                       cublasdx::Type<type>() +
                       cublasdx::Function<cublasdx::function::MM>() +
                       cublasdx::TransposeMode<a_transpose_mode, b_transpose_mode>() +
                       cublasdx::Block());

    bool verbose = true;
    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream))
    int status = benchmark_multiblock_gemm<BlockMM, GlobalMM, Arch, BlockSize>(stream, verbose);
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    return status;
}

template<unsigned int Arch>
struct multiblock_gemm_functor {
    int operator()() { return multiblock_gemm<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<multiblock_gemm_functor>();
}
