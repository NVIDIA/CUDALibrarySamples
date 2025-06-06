#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "../common/common.hpp"
#include "../reference/reference.hpp"

template<class BLAS,
         class ALayout,
         class BLayout,
         class CLayout,
         class ValueType = typename example::uniform_value_type_t<BLAS>>
__launch_bounds__(BLAS::max_threads_per_block) //
    __global__                                 //
    void gemm_kernel(const ValueType* a,
                     const ValueType* b,
                     const ValueType* c,
                     const ValueType  alpha,
                     const ValueType  beta,
                     ValueType*       output,
                     ALayout a_layout, // Static shape with dynamic strides
                     BLayout b_layout, // Static shape with dynamic strides
                     CLayout c_layout) // Static shape with dynamic strides
{
    using value_type = ValueType;
    extern __shared__ __align__(16) char smem[];

    auto a_global_tensor = cublasdx::make_tensor(a, BLAS::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, BLAS::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, BLAS::get_layout_gmem_c());

    // Pass custom layouts to shared memory slicing for appropriate pointer offsets
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<BLAS>(smem, a_layout, b_layout, c_layout);

    // Define shared memory tensors with custom layouts
    auto a_shared_tensor = cute::make_tensor(cute::make_smem_ptr(smem_a), a_layout);
    auto b_shared_tensor = cute::make_tensor(cute::make_smem_ptr(smem_b), b_layout);
    auto c_shared_tensor = cute::make_tensor(cute::make_smem_ptr(smem_c), c_layout);

    using alignment = cublasdx::alignment_of<BLAS>;
    cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<BLAS, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy<BLAS, alignment::c>(c_global_tensor, c_shared_tensor);
    cublasdx::copy_wait();

    BLAS().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);

    __syncthreads();

    auto out_global_tensor = cublasdx::make_tensor(output, BLAS::get_layout_gmem_c());
    cublasdx::copy<BLAS, alignment::c>(c_shared_tensor, out_global_tensor);
}

// This is an example of fp32 general matrix-matrix multiplication (GEMM) performed
// in a single CUDA block using custom CuTe layouts (cute::Layout concept) for A, B and C matrices:
//
//              C = alpha * A * B + beta * C
//
// * A, B, and C are matrices containing real single precision floating-point values.
// * alpha and beta are real single precision floating-point values.
//
// Input data is generated on host using random number generators, and later copied to
// the global memory. Next, kernel with GEMM is executed, and then the matrix C (the result)
// is copied back to host memory. The results are verified against cuBLAS.
//
// In this example the number of threads participating in the GEMM operation is imposed by providing
// BlockDim operator in definition of the GEMM. If BlockDim operator is not used, cuBLASDx automatically
// selects number of threads. Block dimensions are provided via BLAS::block_dim trait.
template<unsigned int Arch>
int simple_gemm() {
    // Parameters m, n, k define the dimensions of matrices A, B, and C
    constexpr unsigned int m = 8;
    constexpr unsigned int n = 16;
    constexpr unsigned int k = 32;

    // Custom shared memory layout
    auto a_layout = cute::make_layout(cute::Shape<cute::Int<m>, cute::Int<k>>{}, cute::make_stride(2u, 2u*m + 3));
    auto b_layout = cute::make_layout(cute::Shape<cute::Int<k>, cute::Int<n>>{}, cute::make_stride(2u, 2u*k + 4));
    auto c_layout = cute::make_layout(cute::Shape<cute::Int<m>, cute::Int<n>>{}, cute::make_stride(2u, 2u*m + 5));

    // Selected CUDA block size (1D)
    constexpr unsigned int block_size = 256;

    // GEMM definition using cuBLASDx operators:
    // 1. The size, the precision, the type (real or complex) and the alignements are set.
    // 2. The BLAS function is selected: MM (matrix multiplication).
    // 3. Block operator informs that GEMM should be performed on CUDA block level.
    // 4. BlockDim operator sets CUDA block dimensions that the kernel will be executed with.
    // 5. Targeted CUDA compute capability is selected with SM operator.
    using BLAS = decltype(cublasdx::Size<m, n, k>() +
                          cublasdx::Precision<float>() +
                          cublasdx::Type<cublasdx::type::real>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Alignment<16, 16, 16>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<block_size>() +
                          cublasdx::SM<Arch>());

    using value_type = typename example::uniform_value_type_t<BLAS>;

    // Allocate managed memory for a, b, c, and output
    value_type* inputs;
    value_type* output;

    constexpr auto global_a_size = example::global_memory_size_of<BLAS>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<BLAS>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<BLAS>::c_size;

    auto inputs_size       = global_a_size + global_b_size + global_c_size;
    auto inputs_size_bytes = inputs_size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&inputs, inputs_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&output, global_c_size * sizeof(value_type)));

    value_type* a     = inputs;
    value_type* b     = a + (global_a_size);
    value_type* c     = b + (global_b_size);
    value_type  alpha = value_type(1.0);
    value_type  beta  = value_type(2.0);

    // Fill the A, B, C matrices with random values
    auto host_a = example::get_random_data<value_type>(0.1, 1.0, global_a_size);
    auto host_b = example::get_random_data<value_type>(0.1, 1.0, global_b_size);
    auto host_c = example::get_random_data<value_type>(0.1, 1.0, global_c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, host_a.data(), global_a_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(b, host_b.data(), global_b_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(c, host_c.data(), global_c_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Increase max dynamic shared memory for the kernel if needed
    unsigned int shared_memory_size = cublasdx::get_shared_storage_size<BLAS>(a_layout, b_layout, c_layout);
    CUDA_CHECK_AND_EXIT(
        cudaFuncSetAttribute(gemm_kernel<BLAS, decltype(a_layout), decltype(b_layout), decltype(c_layout)>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

    // Execute kernel
    gemm_kernel<BLAS, decltype(a_layout), decltype(b_layout), decltype(c_layout)><<<1, BLAS::block_dim, shared_memory_size>>>(a, b, c, alpha, beta, output, a_layout, b_layout, c_layout);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<value_type> host_output(global_c_size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(host_output.data(), output, global_c_size * sizeof(value_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Free device memory
    CUDA_CHECK_AND_EXIT(cudaFree(inputs));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    // Calculate reference
    auto reference_host_output = example::reference_gemm<BLAS>(alpha, host_a, host_b, beta, host_c);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());

    // Check against reference
    if (example::check_error<BLAS>(host_output, reference_host_output)) {
        std::cout << "Success" << std::endl;
        return 0;
    }
    std::cout << "Failure" << std::endl;
    return 1;
}

struct simple_gemm_functor {
    template<int Arch>
    int operator()(std::integral_constant<int, Arch>) {
        return simple_gemm<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner(simple_gemm_functor{});
}
