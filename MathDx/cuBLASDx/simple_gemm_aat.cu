#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "common.hpp"
#include "block_io.hpp"
#include "reference.hpp"

template<class BLAS, class ValueType = typename example::uniform_value_type_t<BLAS>>
__launch_bounds__(BLAS::max_threads_per_block) //
    __global__                                 //
    void gemm_kernel(const ValueType* a,
                     ValueType*       output) {
    using value_type = ValueType;
    extern __shared__ __align__(16) char smem[];

    auto a_global_tensor = cublasdx::make_tensor(a, BLAS::get_layout_gmem_a());

    ValueType* smem_a = reinterpret_cast<ValueType*>(smem);
    /// A, column major arrangement
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, BLAS::get_layout_smem_a());
    // Same data, but because arrangement for 2nd argument (B) was set row major it will be read as A^T
    auto at_shared_tensor = cublasdx::make_tensor(smem_a, BLAS::get_layout_smem_b());

    using alignment = cublasdx::alignment_of<BLAS>;
    cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy_wait();

    // C = A * A^T
    auto [c_register_fragment, partitioner] = BLAS().execute(a_shared_tensor, at_shared_tensor);

    auto out_global_tensor = cublasdx::make_tensor(output, BLAS::get_layout_gmem_c());
    cublasdx::copy_fragment<alignment::c>(c_register_fragment, out_global_tensor, partitioner);
}

// This example demonstrates a special case of matrix multiplication where an input matrix is multiplied
// by itself transposed:
//
//              C = A * A^T
//
// Using the same matrix for the 2nd argument of the multiplication increases the maximum possible dimensions
// of the input matrix by decreasing the shared memory requirements. However, it is important to know this
// can have performance impact.
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
    // Parameters m, n, k define the dimensions of input matrices and the result matrix
    constexpr unsigned int m = 32;
    constexpr unsigned int n = 32;
    constexpr unsigned int k = 64;

    // Selected CUDA block size (1D)
    constexpr unsigned int block_size = 256;

    // GEMM precision
    using precision = float;

    // GEMM definition using cuBLASDx operators:
    // 1. The size, the precision, and the type (real or complex) are set.
    // 2. The BLAS function is selected: MM (matrix multiplication).
    // 3. Arrangement for A is column-major, so to reuse the same memory for A^T (2nd argument; B) is row-major.
    // 4. Block operator informs that GEMM should be performed on CUDA block level.
    // 5. BlockDim operator sets CUDA block dimensions that the kernel will be executed with.
    // 6. Targeted CUDA compute capability is selected with SM operator.
    using BLAS = decltype(cublasdx::Size<m, n, k>() +
                          cublasdx::Precision<precision>() +
                          cublasdx::Type<cublasdx::type::real>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::col_major>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<block_size>() +
                          cublasdx::SM<Arch>());

    using value_type = typename example::uniform_value_type_t<BLAS>;

    // Allocate managed memory
    value_type* a_matrix; // A
    value_type* output;

    constexpr auto global_a_size = example::global_memory_size_of<BLAS>::a_size;
    constexpr auto global_c_size = example::global_memory_size_of<BLAS>::c_size;

    auto input_size       = global_a_size;
    auto input_size_bytes = input_size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&a_matrix, input_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&output, global_c_size * sizeof(value_type)));

    // Fill the input matrix
    auto host_a = example::get_random_data<value_type>(0.5, 1.0, global_a_size);
    for(size_t i = 0; i < host_a.size(); i++) {
        host_a[i] = i+1;
    }
    CUDA_CHECK_AND_EXIT(cudaMemcpy(a_matrix, host_a.data(), global_a_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Increase max dynamic shared memory for the kernel if needed
    const auto shared_memory_size =
        cublasdx::make_shared_storage_calc()
            .add(cublasdx::alignment_of_v_a<BLAS>, sizeof(value_type), BLAS::get_layout_smem_a())
            .get();
    CUDA_CHECK_AND_EXIT(
        cudaFuncSetAttribute(gemm_kernel<BLAS>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

    // Execute kernel
    gemm_kernel<BLAS><<<1, BLAS::block_dim, shared_memory_size>>>(a_matrix, output);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<value_type> host_output(global_c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(host_output.data(), output, global_c_size * sizeof(value_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Free device memory
    CUDA_CHECK_AND_EXIT(cudaFree(a_matrix));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    // Calculate reference
    const value_type alpha { 1.0 };
    const value_type beta  { 0.0 };
    auto host_c  = std::vector<value_type>(host_a.size(), beta);
    auto reference_host_output = example::reference_gemm<BLAS>(alpha, host_a, host_a, beta, host_c);
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
