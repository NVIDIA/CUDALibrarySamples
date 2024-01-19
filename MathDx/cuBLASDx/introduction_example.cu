#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "common.hpp"
#include "block_io.hpp"
#include "reference.hpp"

// Naive copy; one thread does all the work
template<class T>
inline __device__ void naive_copy(T* dst, const T* src, unsigned int size) {
    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        for (unsigned int i = 0; i < size; ++i) {
            dst[i] = src[i];
        }
    }
}

template<class GEMM, class ValueType = typename GEMM::value_type>
__global__ void gemm_kernel(const ValueType  alpha,
                            const ValueType* a,
                            const ValueType* b,
                            const ValueType  beta,
                            ValueType* c) {
    using value_type = ValueType;

    extern __shared__ value_type smem[];

    value_type* sa = smem;
    value_type* sb = smem + GEMM::a_size;
    value_type* sc = smem + GEMM::a_size + GEMM::b_size;

    // Load data from global to shared memory
    naive_copy(sa, a, GEMM::a_size);
    naive_copy(sb, b, GEMM::b_size);
    naive_copy(sc, c, GEMM::c_size);
    __syncthreads();

    // Execute GEMM
    GEMM().execute(alpha, sa, sb, beta, sc);
    __syncthreads();

    // Store data to global memory
    naive_copy(c, sc, GEMM::c_size);
}

template<unsigned int Arch>
int introduction_example() {
    using GEMM = decltype(cublasdx::Size<32, 32, 32>()
                  + cublasdx::Precision<double>()
                  + cublasdx::Type<cublasdx::type::real>()
                  + cublasdx::TransposeMode<cublasdx::transpose_mode::non_transposed, cublasdx::transpose_mode::non_transposed>()
                  + cublasdx::Function<cublasdx::function::MM>()
                  + cublasdx::SM<700>()
                  + cublasdx::Block()
                  + cublasdx::BlockDim<256>());
    #if CUBLASDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using value_type = example::value_type_t<GEMM>;
    #else
    using value_type = typename GEMM::value_type;
    #endif

    // Allocate managed memory for A, B, C matrices in one go
    value_type* abc;
    auto        size       = GEMM::a_size + GEMM::b_size + GEMM::c_size;
    auto        size_bytes = size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&abc, size_bytes));
    // Generate data
    for (size_t i = 0; i < size; i++) {
        abc[i] = double(i / size);
    }

    value_type* a = abc;
    value_type* b = abc + GEMM::a_size;
    value_type* c = abc + GEMM::a_size + GEMM::b_size;

    // Invokes kernel with GEMM::block_dim threads in CUDA block
    gemm_kernel<GEMM><<<1, GEMM::block_dim, GEMM::shared_memory_size>>>(1.0, a, b, 1.0, c);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaFree(abc));
    std::cout << "Success" << std::endl;
    return 0;
}

template<unsigned int Arch>
struct introduction_example_functor {
    int operator()() { return introduction_example<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<introduction_example_functor>();
}
