#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cublasdx.hpp>
#include <cute/tensor.hpp>

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

template<cublasdx::arrangement Arr>
constexpr auto get_stride_from_arrangement() {
    // This assumes tensor A has shape (m, k) and tensor B has shape (k, n)
    if constexpr(Arr == cublasdx::col_major) {
        return cute::GenColMajor{};
    } else {
        return cute::GenRowMajor{};
    }
}

template <int GridX, int GridY, class BlockMM, class MatrixA, class MatrixB, class MatrixC, class ValueType = typename example::uniform_value_type_t<BlockMM>>
__launch_bounds__(BlockMM::max_threads_per_block) __global__
void block_mm_kernel(MatrixA gA, MatrixB gB, MatrixC gC, ValueType alpha, ValueType beta) {
    using value_type = ValueType;

    // We will partition A along the m-mode into M blocks of size m, B along the n-mode into N blocks of size n,
    // and C along both m- and n-modes into M * N blocks of size m x n. Each of the M * N CUDA blocks computes
    // its block of matrix C: C_mxn = A_mxk @ B_kxn.

    constexpr auto a_arrangement = cublasdx::arrangement_of<BlockMM>::a;
    constexpr auto b_arrangement = cublasdx::arrangement_of<BlockMM>::b;

    // Create the blocks for partitioning A, B, and C.
    cute::Layout blockA = make_layout(cute::make_shape(cute::Int<GridX>{}, cute::_1{}), get_stride_from_arrangement<a_arrangement>());
    cute::Layout blockB = make_layout(cute::make_shape(cute::_1{}, cute::Int<GridY>{}), get_stride_from_arrangement<b_arrangement>());
    cute::Layout blockC = make_layout(cute::make_shape(cute::Int<GridX>{}, cute::Int<GridY>{}));

    constexpr unsigned int m = cublasdx::size_of<BlockMM>::m;
    constexpr unsigned int n = cublasdx::size_of<BlockMM>::n;
    constexpr unsigned int k = cublasdx::size_of<BlockMM>::k;

    auto idxA = crd2idx(cute::make_coord(blockIdx.x,          0), blockA);
    auto idxB = crd2idx(cute::make_coord(0         , blockIdx.y), blockB);
    auto idxC = crd2idx(cute::make_coord(blockIdx.x, blockIdx.y), blockC);

    // Get the local partitions of A, B, and C for the current block.
    cute::Tensor pA = cute::local_partition(gA, blockA, idxA, cute::make_step(cute::Int<m>{}, cute::Int<k>{}));
    cute::Tensor pB = cute::local_partition(gB, blockB, idxB, cute::make_step(cute::Int<k>{}, cute::Int<n>{}));
    cute::Tensor pC = cute::local_partition(gC, blockC, idxC, cute::make_step(cute::Int<m>{}, cute::Int<n>{}));

    extern __shared__ value_type shared_mem[];
    auto [smem_a, smem_b, smem_c] = BlockMM::slice_shared_memory(reinterpret_cast<char*>(shared_mem));

    // Create shared memory tensors.
    cute::Tensor sA = cublasdx::make_tensor(smem_a, BlockMM::get_layout_smem_a());
    cute::Tensor sB = cublasdx::make_tensor(smem_b, BlockMM::get_layout_smem_b());
    cute::Tensor sC = cublasdx::make_tensor(smem_c, BlockMM::get_layout_smem_c());

    // Copy A, B, and C from global to shared memory.
    using alignment = cublasdx::alignment_of<BlockMM>;
    cublasdx::copy<BlockMM, alignment::a>(pA, sA);
    cublasdx::copy<BlockMM, alignment::b>(pB, sB);
    cublasdx::copy<BlockMM, alignment::c>(pC, sC);

    __syncthreads();

    // Execute GEMM
    BlockMM().execute(alpha, sA, sB, beta, sC);

    __syncthreads();

    // Copy C from shared to global memory.
    cublasdx::copy<BlockMM, alignment::c>(sC, pC);

    __syncthreads();
}

template<class BlockMM, class GlobalMM, unsigned int Arch, unsigned int BlockSize>
int benchmark_multiblock_gemm(const cudaStream_t& stream, bool verbose = false) {
    using namespace cublasdx;

    static constexpr unsigned int kernel_repeats = 5;
    static constexpr unsigned int warm_up_runs   = 5;

    constexpr bool set_block_size{BlockSize > 0};
    using block_mm_type = std::conditional_t<set_block_size, decltype(BlockMM() + BlockDim<BlockSize>()), BlockMM>;
    using value_type = typename example::uniform_value_type_t<block_mm_type>;

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

    constexpr auto a_arrangement = cublasdx::arrangement_of<BlockMM>::a;
    constexpr auto b_arrangement = cublasdx::arrangement_of<BlockMM>::b;

    // Wrap the device memory as CuTe tensors.
    auto tensor_a = cute::make_tensor(cute::make_gmem_ptr(a), cute::make_shape(cute::Int<M>{}, cute::Int<K>{}), get_stride_from_arrangement<a_arrangement>());
    auto tensor_b = cute::make_tensor(cute::make_gmem_ptr(b), cute::make_shape(cute::Int<K>{}, cute::Int<N>{}), get_stride_from_arrangement<b_arrangement>());
    auto tensor_c = cute::make_tensor(cute::make_gmem_ptr(c), cute::make_shape(cute::Int<M>{}, cute::Int<N>{}));

    using a_type = decltype(tensor_a);
    using b_type = decltype(tensor_b);
    using c_type = decltype(tensor_c);

    constexpr dim3 grid{M / cublasdx::size_of<block_mm_type>::m, N / cublasdx::size_of<block_mm_type>::n};

    // Increase max dynamic shared memory for the kernel if needed.
    auto kernel = block_mm_kernel<grid.x, grid.y, block_mm_type, a_type, b_type, c_type>;

    CUDA_CHECK_AND_EXIT(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, block_mm_type::shared_memory_size));

    // Create a grid of size (M / m,  N / n).
    // Measure performance of N trials.
    double time = example::measure::execution(
        [&](cudaStream_t stream) {
           kernel<<<grid, block_mm_type::block_dim, block_mm_type::shared_memory_size, stream>>>(
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
    // Since GlobalMM describes global memory we can use leading_dimension here
    // but in a regular problem definition leading_dimension_of would provide
    // shared memory layout information.
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

    // Choose arrangement for A and B: row-major or column-major.
    // Conjugation or other element-wise transformations can be passed to the execute function.
    constexpr auto a_arrangement = cublasdx::col_major;
    constexpr auto b_arrangement = cublasdx::row_major;

    // Define the local matrix multiplication operation.
    using BlockMM  = decltype(
        cublasdx::Size<m, n, k>() +
        cublasdx::Precision<precision>() +
        cublasdx::Type<type>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Arrangement<a_arrangement, b_arrangement>() +
        cublasdx::Block() +
        cublasdx::SM<Arch>()
    );
    // The global matrix multiplication operation provides a convenient way to encapsulate data of interest: (M, N, K),
    // matrix sizes, leading dimensions, etc. It cannot be executed since the problem is too large to fit into shared memory.
    using GlobalMM = decltype(
        cublasdx::Size<M, N, K>() +
        cublasdx::Precision<precision>() +
        cublasdx::Type<type>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Arrangement<a_arrangement, b_arrangement>() +
        cublasdx::Block()
    );

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
