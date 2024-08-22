#include <iostream>
#include <vector>
#include <type_traits>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "common.hpp"
#include "reference.hpp"

template<class BLAS1, class BLAS2, class ValueType = example::uniform_value_type_t<BLAS1>>
__launch_bounds__(BLAS1::max_threads_per_block) //
    __global__                                 //
    void gemm_kernel(const ValueType alpha1,
                     const ValueType* a,
                     const ValueType* b,
                     const ValueType  beta1,
                     const ValueType* c,
                     const ValueType  alpha2,
                     const ValueType* d,
                     const ValueType  beta2,
                     const ValueType* f,
                     ValueType*       output) {
    using value_type = ValueType;
    extern __shared__ __align__(16) char smem[];

    static_assert(std::is_same_v<value_type, example::uniform_value_type_t<BLAS2>>, "BLAS1 and BLAS2 must have the same type and precision");
    static_assert((BLAS1::c_dim == BLAS2::a_dim), "The dimensions of C matrix are different in BLAS1 and BLAS2");

    // Matrix C is the first in shared memory, because it's reused in the 2nd GEMM. Moreover,
    // matrices A and B might have different sizes than F and D.
    value_type* smem_c = reinterpret_cast<value_type*>(smem);
    value_type* smem_a = reinterpret_cast<value_type*>(smem) + BLAS1::c_size;
    value_type* smem_b = reinterpret_cast<value_type*>(smem) + BLAS1::c_size + BLAS1::a_size;

    auto a_global_tensor = cublasdx::make_tensor(a, BLAS1::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, BLAS1::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, BLAS1::get_layout_gmem_c());

    auto a_shared_tensor = cublasdx::make_tensor(smem_a, BLAS1::get_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, BLAS1::get_layout_smem_b());
    auto c_shared_tensor = cublasdx::make_tensor(smem_c, BLAS1::get_layout_smem_c());

    using alignment1 = cublasdx::alignment_of<BLAS1>;
    cublasdx::copy<BLAS1, alignment1::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<BLAS1, alignment1::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy<BLAS1, alignment1::c>(c_global_tensor, c_shared_tensor);
    cublasdx::copy_wait();

    BLAS1().execute(alpha1, a_shared_tensor, b_shared_tensor, beta1, c_shared_tensor);
    __syncthreads();

    static_assert((BLAS1::c_size == BLAS2::a_size), "The sizes of C matrix are different in BLAS1 and BLAS2");
    value_type* smem_d = smem_c + BLAS2::a_size;
    value_type* smem_f = smem_c + BLAS2::a_size + BLAS2::b_size;

    auto d_global_tensor = cublasdx::make_tensor(d, BLAS2::get_layout_gmem_b());
    auto f_global_tensor = cublasdx::make_tensor(f, BLAS2::get_layout_gmem_c());

    auto d_shared_tensor = cublasdx::make_tensor(smem_d, BLAS2::get_layout_smem_b());
    auto f_shared_tensor = cublasdx::make_tensor(smem_f, BLAS2::get_layout_smem_c());

    using alignment2 = cublasdx::alignment_of<BLAS2>;
    cublasdx::copy<BLAS2, alignment2::b>(d_global_tensor, d_shared_tensor);
    cublasdx::copy<BLAS2, alignment2::c>(f_global_tensor, f_shared_tensor);
    cublasdx::copy_wait();

    BLAS2().execute(alpha2, c_shared_tensor, d_shared_tensor, beta2, f_shared_tensor);

    __syncthreads();
    auto out_global_tensor = cublasdx::make_tensor(output, BLAS2::get_layout_gmem_c());
    cublasdx::copy<BLAS2, alignment2::c>(f_shared_tensor, out_global_tensor);
}

// This is an example of two fp16 general matrix-matrix multiplications (GEMM) fused together
// and performed in one kernel in a single CUDA block:
//
//             1) C = alpha1 * (A * B) + beta1 * C
//             2) F = alpha2 * (C * D) + beta2 * F
//
// * A, B, C, D and F are matrices containing real half precision floating-point values.
// * (alpha1, beta1) and (alpha2, beta2) and real half precision floating-point values.
//
// Input data is generated on host using random number generators, and later copied to
// the global memory. Next, kernel with GEMM is executed, and then the matrix F (the result)
// is copied back to host memory. The results are verified against cuBLAS.
//
// In this example the number of threads participating in the GEMM operation is imposed by providing
// BlockDim operator in definition of the GEMM. If BlockDim operator is not used, cuBLASDx automatically
// selects number of threads. Block dimensions are provided via BLAS::block_dim trait.
//
// Notes:
// * Both GEMM operations use the same number of threads, however, it's not a requirement.
// * It's important that the dimensions of the first and the 2nd GEMM are set in such a way that
//   the C matrix has the same dimensions in both operations.
template<unsigned int Arch>
int simple_gemm() {
    // Parameters m1, n1, k1 define the dimensions of matrices A, B, and C
    constexpr unsigned int m1          = 64;
    constexpr unsigned int n1          = 64;
    constexpr unsigned int k1          = 64;

    // Parameters m2, n2, k2 define the dimensions of matrices C, D and F
    // Note: (m1, n1) and (m2, k2) must be equal as describe the same matrix (matrix C)
    constexpr unsigned int m2          = m1;
    constexpr unsigned int n2          = 128;
    constexpr unsigned int k2          = n1;

    // The logical dimensions of matrix A are: [m1, k1] (m rows, k columns)
    // The logical dimensions of matrix B are: [k1, n1]
    // The logical dimensions of matrix C are: [m1, n1]
    constexpr auto a_arrangement = cublasdx::col_major;
    constexpr auto b_arrangement = cublasdx::col_major;
    constexpr auto c_arrangement = cublasdx::col_major;

    // The logical dimensions of matrix C are: [m2, k2] == [m1, n1]
    // The logical dimensions of matrix D are: [k2, n2]
    // The logical dimensions of matrix F are: [m2, n2]
    constexpr auto d_arrangement = cublasdx::col_major;
    constexpr auto f_arrangement = cublasdx::col_major;

    // Use the same block size for both GEMM operations, so BLAS1::block_dim == BLAS2::block_dim which
    // simplifies the example.
    constexpr unsigned int block_size = 128;

    using BLAS1 = decltype(cublasdx::Size<m1, n1, k1>() +
                           cublasdx::Precision<__half>() +
                           cublasdx::Type<cublasdx::type::real>() +
                           cublasdx::Function<cublasdx::function::MM>() +
                           cublasdx::Arrangement<a_arrangement, b_arrangement, c_arrangement>() +
                           cublasdx::Block() +
                           cublasdx::BlockDim<block_size>() +
                           cublasdx::SM<Arch>());
    using BLAS2 = decltype(cublasdx::Size<m2, n2, k2>() +
                           cublasdx::Precision<__half>() +
                           cublasdx::Type<cublasdx::type::real>() +
                           cublasdx::Function<cublasdx::function::MM>() +
                           cublasdx::Arrangement<c_arrangement, d_arrangement, f_arrangement>() +
                           cublasdx::Block() +
                           cublasdx::BlockDim<block_size>() +
                           cublasdx::SM<Arch>());

    using value_type = example::uniform_value_type_t<BLAS1>;

    // alpha and beta for the first GEMM
    value_type alpha1 = 1.0;
    value_type beta1  = 0.0;

    // alpha and beta for the 2nd GEMM
    value_type alpha2 = 1.0;
    value_type beta2  = 1.0;

    // Allocate managed memory for a, b, c, d, f and output
    value_type* inputs;
    value_type* output;

    constexpr auto global_a1_size = example::global_memory_size_of<BLAS1>::a_size;
    constexpr auto global_b1_size = example::global_memory_size_of<BLAS1>::b_size;
    constexpr auto global_c1_size = example::global_memory_size_of<BLAS1>::c_size;
    constexpr auto global_b2_size = example::global_memory_size_of<BLAS2>::b_size;
    constexpr auto global_c2_size = example::global_memory_size_of<BLAS2>::c_size;

    auto inputs_size       = global_a1_size + global_b1_size + global_c1_size + global_b2_size + global_c2_size;
    auto inputs_size_bytes = inputs_size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&inputs, inputs_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output, global_c2_size * sizeof(value_type)));

    value_type* a     = inputs;
    value_type* b     = a + (global_a1_size);
    value_type* c     = b + (global_b1_size); // C matrix for BLAS1, A matrix for BLAS2
    value_type* d     = c + (global_c1_size); // D is B matrix for BLAS2
    value_type* f     = d + (global_b2_size); // F is C matrix for BLAS2

    // Fill the A, B, C matrices with random values
    auto host_a = example::get_random_data<value_type>(0.1, 1.0, global_a1_size);
    auto host_b = example::get_random_data<value_type>(0.1, 1.0, global_b1_size);
    auto host_c = example::get_random_data<value_type>(0.1, 1.0, global_c1_size);
    auto host_d = example::get_random_data<value_type>(1.0, 2.0, global_b2_size);
    auto host_f = example::get_random_data<value_type>(1.0, 10.0, global_c2_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, host_a.data(), global_a1_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(b, host_b.data(), global_b1_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(c, host_c.data(), global_c1_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d, host_d.data(), global_b2_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(f, host_f.data(), global_c2_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Increase max dynamic shared memory for the kernel if needed
    const auto shared_memory = std::max<size_t>(BLAS1::shared_memory_size, BLAS2::shared_memory_size);
    CUDA_CHECK_AND_EXIT(
        cudaFuncSetAttribute(gemm_kernel<BLAS1, BLAS2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

    // Execute kernel
    gemm_kernel<BLAS1, BLAS2><<<1, BLAS1::block_dim, shared_memory>>>(alpha1, a, b, beta1, c, alpha2, d, beta2, f, output);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<value_type> host_output(global_c2_size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(host_output.data(), output, global_c2_size * sizeof(value_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Free device memory
    CUDA_CHECK_AND_EXIT(cudaFree(inputs));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    // Calculate reference
    // 1st GEMM
    auto blas1_reference_host_output = example::reference_gemm<BLAS1>(alpha1, host_a, host_b, beta1, host_c);
    // 2nd GEMM
    std::vector<value_type> blas2_host_c(blas1_reference_host_output.size());
    blas2_host_c.assign(blas1_reference_host_output.begin(), blas1_reference_host_output.end());
    auto reference_host_output = example::reference_gemm<BLAS2>(alpha2, blas2_host_c, host_d, beta2, host_f);

    // Check against reference
    if (example::check(host_output, reference_host_output)) {
        std::cout << "Success" << std::endl;
        return 0;
    }
    std::cout << "Failure" << std::endl;
    return 1;
}

template<unsigned int Arch>
struct simple_gemm_functor {
    int operator()() { return simple_gemm<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<simple_gemm_functor>();
}
