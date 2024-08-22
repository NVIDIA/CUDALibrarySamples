#include <iostream>
#include <vector>
#include <type_traits>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "common.hpp"
#include "block_io.hpp"
#include "reduce.hpp"

template <int M, int LD, class ValueType> __device__ __forceinline__
void scale_rows(ValueType *data, int size, ValueType *scale) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        int r = i % LD;
        if (r < M) {
            data[i] /= scale[r];
        }
    }
}

template<class BLAS1, class BLAS2, class ValueType = example::uniform_value_type_t<BLAS1>>
__launch_bounds__(BLAS1::max_threads_per_block) __global__
void scaled_dot_product_attention_batched(const ValueType* query,
                                          const ValueType* key,
                                          const ValueType* value,
                                          const ValueType* mask,
                                          ValueType*       output) {
    using value_type = ValueType;
    constexpr unsigned int block_size = BLAS1::block_dim.x * BLAS1::block_dim.y * BLAS1::block_dim.z;

    extern __shared__ __align__(16) char smem[];

    static_assert(std::is_same_v<value_type, example::uniform_value_type_t<BLAS2>>, "BLAS1 and BLAS2 must have the same type and precision");
    static_assert((BLAS1::c_dim == BLAS2::a_dim), "The dimensions of the C matrix in BLAS1 must be the same as the dimensions of the A matrix in BLAS2");

    constexpr auto global_a1_size = example::global_memory_size_of<BLAS1>::a_size;
    constexpr auto global_b1_size = example::global_memory_size_of<BLAS1>::b_size;
    constexpr auto global_c1_size = example::global_memory_size_of<BLAS1>::c_size;
    constexpr auto global_b2_size = example::global_memory_size_of<BLAS2>::b_size;
    constexpr auto global_c2_size = example::global_memory_size_of<BLAS2>::c_size;

    // Each block processes one sample from the batch. The inputs and output must be offset to point to the data for the sample.
    query  += blockIdx.x * global_a1_size;
    key    += blockIdx.x * global_b1_size;
    mask   += blockIdx.x * global_c1_size;
    value  += blockIdx.x * global_b2_size;
    output += blockIdx.x * global_c2_size;

    // Matrix C is the first in shared memory, because it's reused in the reduction as well as the 2nd matrix multiplication.
    value_type* smem_c = reinterpret_cast<value_type*>(smem);
    value_type* smem_a = reinterpret_cast<value_type*>(smem) + BLAS1::c_size;
    value_type* smem_b = reinterpret_cast<value_type*>(smem) + BLAS1::c_size + BLAS1::a_size;

    example::io<BLAS1>::a_fast_load<block_size>(smem_a, query);
    example::io<BLAS1>::b_fast_load<block_size>(smem_b, key);
    example::io<BLAS1>::c_fast_load<block_size>(smem_c, mask);
    __syncthreads();

    using cublasdx::size_of;

    // First matrix multiplication C := query @ key.T / sqrt(query.shape[-1]) + mask
    ValueType alpha = rsqrt(ValueType(size_of<BLAS1>::k));  // This can also be precomputed and provided as an argument.
    BLAS1().execute(alpha, smem_a, smem_b, ValueType(1.), smem_c);
    __syncthreads();

    // Compute softmax(C) using the following steps: row-wise reduction, transformation, row-wise reduction, and row scaling.

    constexpr auto ldc = cublasdx::leading_dimension_of<BLAS1>::c;
    value_type* smem_e = smem_c + BLAS1::c_size;
    value_type* smem_w = smem_e + ldc;

    // Find row maximum.
    example::reducers::maximum<ValueType> reducer_max;
    example::reduce_row<size_of<BLAS1>::m, size_of<BLAS1>::n, ldc>(smem_c, reducer_max, smem_w, smem_e);
    __syncthreads();

    // Transform C = exp(C) using row maximum as exponent offset for numerical stability.
    auto transformer = [smem_e, m = size_of<BLAS1>::m, ld=ldc](int i, ValueType v) {
        int r = i % ld;
        return r < m ? example::exp(v - smem_e[r]) : v;
    };
    example::transform(smem_c, BLAS1::c_size, transformer);
    __syncthreads();

    // Calculate E = RowSum(exp(C)).
    example::reducers::addition<ValueType> reducer_add;
    example::reduce_row<size_of<BLAS1>::m, size_of<BLAS1>::n, ldc>(smem_c, reducer_add, smem_w, smem_e);
    __syncthreads();

    // Scale rows to get C = softmax(C): C = C / E[:,None]
    scale_rows<size_of<BLAS1>::m, ldc>(smem_c, BLAS1::c_size, smem_e);
    __syncthreads();

    static_assert((BLAS1::c_size == BLAS2::a_size), "The size of C in BLAS1 must be equal to the size of A in BLAS2");
    value_type* smem_f = smem_c + BLAS2::a_size;
    value_type* smem_g = smem_f + BLAS2::b_size;

    example::io<BLAS2>::b_fast_load<block_size>(smem_f, value);
    __syncthreads();

    // Second matrix multiplication G := C @ value, where C = softmax(query @ key.T / sqrt(query.shape[-1]) + mask)
    BLAS2().execute(ValueType(1.), smem_c, smem_f, ValueType(0.), smem_g);
    __syncthreads();

    example::io<BLAS2>::c_fast_store<block_size>(output, smem_g);
}

template<class BLAS1, class BLAS2, class ValueType = example::uniform_value_type_t<BLAS1>>
double measure_cublasdx(unsigned int kernel_warm_up_repeats,
                        unsigned int kernel_repeats,
                        unsigned int batch_size,
                        const ValueType* query,
                        const ValueType* key,
                        const ValueType* value,
                        const ValueType* mask,
                        ValueType*       output,
                        cudaStream_t     stream) {

    // Increase max dynamic shared memory for the kernel if needed.
    // The memory required is the maximum of the memory required for the row reduction and the two matrix multiplications
    // C := Q @ K.T / sqrt(Q.shape[-1]) + M and R := softmax(C) @ V.
    const size_t redn_smem_size = (BLAS1::c_size + cublasdx::leading_dimension_of<BLAS1>::c + BLAS1::block_dim.x) * sizeof(ValueType);
    const auto shared_memory = std::max<size_t>(std::max<size_t>(BLAS1::shared_memory_size, redn_smem_size), BLAS2::shared_memory_size);
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(scaled_dot_product_attention_batched<BLAS1, BLAS2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

    // Execute kernel. Each sample in the the batch is executed in a different CUDA block.
    double time = example::measure::execution(
        [&](cudaStream_t stream) {
            scaled_dot_product_attention_batched<BLAS1, BLAS2><<<batch_size, BLAS1::block_dim, shared_memory, stream>>>(query, key, value, mask, output);
        },
        kernel_warm_up_repeats, kernel_repeats, stream);

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    return time;
}

// This example illustrates how to compute multi-head attention using batched "scaled dot product attention"
// calculations. The entire computation is implemented in a single fused kernel using cuBLASDx.
//
// The inputs are three matrices "query", "key", and "value" along with a mask, with the following dimensions:
//     query(N, S, E),
//     key(N, S, E),
//     value(N, S, E),
//     mask(N, S, S),
// where N is the batch size, S is the maximum sequence length and E is the embedding size per attention head.
// The product of E with the number of heads is the embedding dimension chosen for the words in the vocabulary.
// Note that the use of batch here encompasses the attention heads as well as the number of samples (it's the
// product of the number of attention heads and the number of samples).
//
// The result r of scaled dot product attention is a matrix of size (N, S, E):
//     r = softmax(query @ key.T / sqrt(query.shape[-1]) + mask) @ value
// where @ represents batched matrix multiplication. This operation essentially reweights the values matrix
// according to the (multiple, one for each head) attention for each sample in the batch.
//
// Notes:
// * The sizes of query, key, and value for N=1  must be such that the data should not only fit within shared
//   memory but also small enough such that a single block is the _optimal_ choice. This is because cuBLASDx is
//   limited to using one block whereas libraries like cuBLAS can use a larger number of blocks.
//
// * Each item in the batch will be processed using a single CUDA block and the items can be processed in
//   parallel (subject to the number of SMs available on the selected device).
//
template<unsigned int Arch>
int scaled_dot_product_attention_batched_performance() {
    // Define the maximum sequence length (number of tokens) S.
    constexpr unsigned int S = 9;

    // Define the embedding dimension E.
    constexpr unsigned int E = 64;

    // Define the batch size N.
    constexpr unsigned int N = 64;

    // Parameters m1, n1, k1 define the dimensions of matrices "query", "key", and "mask".
    constexpr unsigned int m1          = S;
    constexpr unsigned int n1          = S;
    constexpr unsigned int k1          = E;

    // Parameters m2, n2, k2 define the dimensions of matrices "query @ key.T" and "value".
    // Note: (m1, n1) and (m2, k2) must be equal as describe the same matrix.
    constexpr unsigned int m2          = m1;
    constexpr unsigned int n2          = E;
    constexpr unsigned int k2          = n1;

    // Use the same block size for both GEMM operations as well as the reduction, which
    // simplifies the example.
    constexpr unsigned int block_size = 256;

    // Choose the precision (__half, float, double). The data type can only be real.
    using precision = float;

    using BASE        = decltype(
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Precision<precision>() +
                          cublasdx::Type<cublasdx::type::real>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<block_size>() +
                          cublasdx::Alignment<16, 16, 16>() +
                          cublasdx::SM<Arch>());
    using BLAS1_      = decltype(BASE() +
                          cublasdx::Size<m1, n1, k1>() +
                          cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major>());
    // Use cuBLASDx suggested leading dimensions. Since we perform two matrix multiplications, the C matrix from the first has to be
    // consistent (dimensions and padding) with the A matrix of the second. Therefore create the second matrix multiplication
    // descriptor first and use its lda to set the first matrix multiplication's ldc.
    using LD1         = cublasdx::suggested_leading_dimension_of<BLAS1_, Arch>;
    using BLAS2_      = decltype(BASE() +
                          cublasdx::Size<m2, n2, k2>() +
                          cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major>());
    using LD2         = cublasdx::suggested_leading_dimension_of<BLAS2_, Arch>;
    using BLAS2       = decltype(BLAS2_() + typename LD2::type());
    using BLAS1       = decltype(BLAS1_() + cublasdx::LeadingDimension<LD1::lda, LD1::ldb, LD2::lda>());
    using value_type = example::uniform_value_type_t<BLAS1>;

    // Allocate device memory for query, key, value, mask, and output.
    value_type* inputs;
    value_type* output;

    constexpr auto global_a1_size = example::global_memory_size_of<BLAS1>::a_size;
    constexpr auto global_b1_size = example::global_memory_size_of<BLAS1>::b_size;
    constexpr auto global_c1_size = example::global_memory_size_of<BLAS1>::c_size;
    constexpr auto global_b2_size = example::global_memory_size_of<BLAS2>::b_size;
    constexpr auto global_c2_size = example::global_memory_size_of<BLAS2>::c_size;

    auto inputs_size       = N * (global_a1_size + global_b1_size + global_c1_size + global_b2_size);
    auto inputs_size_bytes = inputs_size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&inputs, inputs_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output, N * global_c2_size * sizeof(value_type)));

    value_type* query     = inputs;                      // A matrix for BLAS1
    value_type* key       = query + N * (global_a1_size); // B matrix for BLAS1
    value_type* mask      = key   + N * (global_b1_size); // C matrix for BLAS1
    value_type* value     = mask  + N * (global_c1_size); // B matrix for BLAS2

    // Fill the query, key, and value matrices with random values.
    auto host_query = example::get_random_data<value_type>(0.1, 0.5, N * global_a1_size);
    auto host_key   = example::get_random_data<value_type>(0.1, 0.5, N * global_b1_size);
    auto host_value = example::get_random_data<value_type>(0.1, 0.5, N * global_b2_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(query, host_query.data(), N * global_a1_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(key, host_key.data(), N * global_b1_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(value, host_value.data(), N * global_b2_size * sizeof(value_type), cudaMemcpyHostToDevice));
    // Set the mask to 0. (no mask).
    CUDA_CHECK_AND_EXIT(cudaMemset(mask, 0, N * global_c1_size * sizeof(value_type)));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    const unsigned int kernel_repeats = 100;
    const unsigned int kernel_warm_up_repeats = 1;
    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream))

    // Measure cuBLASDx performance.
    double time_cublasdx =
        measure_cublasdx<BLAS1, BLAS2>(kernel_warm_up_repeats, kernel_repeats, N, query, key, value, mask, output, stream);

    // Write performance data.
    using cublasdx::size_of;
    std::cout << "m1, n1, k1: " << size_of<BLAS1>::m << ", " << size_of<BLAS1>::n << ", " << size_of<BLAS1>::k
              << std::endl;
    const auto [lda1, ldb1, ldc1] = cublasdx::leading_dimension_of_v<BLAS1>;
    std::cout << "Leading dimensions (lda1, ldb1, ldc1): " << lda1 << ", " << ldb1 << ", " << ldc1 << std::endl;
    std::cout << "m2, n2, k2: " << size_of<BLAS2>::m << ", " << size_of<BLAS2>::n << ", " << size_of<BLAS2>::k
              << std::endl;
    const auto [lda2, ldb2, ldc2] = cublasdx::leading_dimension_of_v<BLAS2>;
    std::cout << "Leading dimensions (lda2, ldb2, ldc2): " << lda2 << ", " << ldb2 << ", " << ldc2 << std::endl;
    std::cout << "Type: " << example::type_string<value_type>() << std::endl;
    std::cout << "Precision: " << example::precision_string<value_type>() << std::endl;
    std::cout << "Batch Size: " << N << std::endl;

    std::cout << "\ncuBLASDx (fused kernel for batched scaled dot product attention calculation)\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Avg time [ms]  = " << time_cublasdx / kernel_repeats << "\n";

    // Free resources.
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(inputs));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    return 0;
}

template<unsigned int Arch>
struct scaled_dot_product_attention_batched_performance_functor {
    int operator()() {
        return scaled_dot_product_attention_batched_performance<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner<scaled_dot_product_attention_batched_performance_functor>();
}
