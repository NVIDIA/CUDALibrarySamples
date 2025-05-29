#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "../common/common.hpp"
#include "../reference/reference.hpp"

template<class BLAS, class AValueType, class BValueType, class CValueType>
__launch_bounds__(BLAS::max_threads_per_block) //
    __global__                                 //
    void gemm_kernel(const AValueType* a,
                     const BValueType* b,
                     const CValueType* c,
                     const CValueType  alpha,
                     const CValueType  beta,
                           CValueType* output) {

    using a_value_type = AValueType;
    using b_value_type = BValueType;
    using c_value_type = CValueType;
    using c_compute_type = typename BLAS::c_value_type;

    extern __shared__ __align__(16) char smem[];

    // Create global tensors with input types
    auto a_global_tensor = cublasdx::make_tensor(a, BLAS::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, BLAS::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, BLAS::get_layout_gmem_c());

    // Create shared memory tensors with input types
    auto [smem_a, smem_b] = cublasdx::slice_shared_memory_ab<BLAS, AValueType, BValueType>(smem);
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, BLAS::get_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, BLAS::get_layout_smem_b());

    using alignment = cublasdx::alignment_of<BLAS>;

    // Copy input precision data from global to shared memory
    cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<BLAS, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy_wait();

    // Execute GEMM on shared memory tensors
    // 1. Load input precision data from shared to registers
    // 2. Convert in registers to compute precision
    // 3. Perform MMA and accumulate in registers in compute precision
    auto [c_frag, partitioner] =
        BLAS().execute(a_shared_tensor, b_shared_tensor);

    // Create output fragment with input precision
    auto d_frag_io = cublasdx::make_fragment_like<c_value_type>(c_frag);
    // Create accumulator fragment with compute precision
    auto d_frag_compute = partitioner.make_accumulator_fragment();
    // Copy input precision data from global to output fragment
    cublasdx::copy_fragment<alignment::c>(c_global_tensor, d_frag_io, partitioner);
    // Convert in registers to compute precision
    cublasdx::transform(d_frag_io, d_frag_compute, example::converter<c_compute_type>{});
    // Perform AXPBY and accumulate in registers in compute precision
    auto compute_alpha = static_cast<c_compute_type>(alpha);
    auto compute_beta = static_cast<c_compute_type>(beta);
    cublasdx::axpby(compute_alpha, c_frag, compute_beta, d_frag_compute);
    // Convert in registers to output precision
    cublasdx::transform(d_frag_compute, d_frag_io, example::converter<c_value_type>{});
    // Copy output precision data from registers to global memory
    auto out_global_tensor = cublasdx::make_tensor(output, BLAS::get_layout_gmem_c());
    cublasdx::copy_fragment<alignment::c>(d_frag_io, out_global_tensor, partitioner);
}

// This is an example of fp32 general matrix-matrix multiplication (GEMM) with fp16 inputs
// performed in a single CUDA block:
//
//              C = alpha * A * B + beta * C
//
// * A, B, and C are matrices containing real half precision floating-point values.
// * alpha and beta are values convertible to single precison floating-point values.
// all operations happen in fp32 precision, with conversion happening in registers,
// which allows to reduce memory transfers by half.
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

    // Selected CUDA block size (1D)
    constexpr unsigned int block_size = 256;

    // Type of input and output values in global and shared memory
    using data_type = __half;
    // Type in which all mathematical operations and accumulation will happen
    using compute_type = float;

    // GEMM definition using cuBLASDx operators:
    // 1. The size, the precision, and the type (real or complex) are set.
    // 2. The BLAS function is selected: MM (matrix multiplication).
    // 3. Block operator informs that GEMM should be performed on CUDA block level.
    // 4. BlockDim operator sets CUDA block dimensions that the kernel will be executed with.
    // 5. Targeted CUDA compute capability is selected with SM operator.
    //
    // NOTE: The alignment is adjusted to data_type not computation types from cublasdx::Precision
    using BLAS = decltype(cublasdx::Size<m, n, k>() +
                          cublasdx::Precision<compute_type>() +
                          cublasdx::Type<cublasdx::type::real>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Alignment<sizeof(data_type), sizeof(data_type), sizeof(data_type)>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<block_size>() +
                          cublasdx::SM<Arch>());

    // Allocate managed memory for a, b, c, and output
    data_type* inputs;
    data_type* output;

    constexpr auto global_a_size = example::global_memory_size_of<BLAS>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<BLAS>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<BLAS>::c_size;

    auto inputs_size       = global_a_size + global_b_size + global_c_size;
    auto inputs_size_bytes = inputs_size * sizeof(data_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&inputs, inputs_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&output, global_c_size * sizeof(data_type)));

    data_type* a     = inputs;
    data_type* b     = a + (global_a_size);
    data_type* c     = b + (global_b_size);
    data_type  alpha = data_type(1.0);
    data_type  beta  = data_type(2.0);

    // Fill the A, B, C matrices with random values
    auto host_a = example::get_random_data<data_type>(0.01, 0.1, global_a_size);
    auto host_b = example::get_random_data<data_type>(0.01, 0.1, global_b_size);
    auto host_c = example::get_random_data<data_type>(0.01, 0.1, global_c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, host_a.data(), global_a_size * sizeof(data_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(b, host_b.data(), global_b_size * sizeof(data_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(c, host_c.data(), global_c_size * sizeof(data_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Increase max dynamic shared memory for the kernel if needed
    auto shared_memory_size = cublasdx::get_shared_storage_size_ab<BLAS, data_type, data_type>();
    CUDA_CHECK_AND_EXIT(
        cudaFuncSetAttribute(gemm_kernel<BLAS, data_type, data_type, data_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

    // Execute kernel
    gemm_kernel<BLAS><<<1, BLAS::block_dim, shared_memory_size>>>(a, b, c, alpha, beta, output);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<data_type> host_output(global_c_size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(host_output.data(), output, global_c_size * sizeof(data_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Free device memory
    CUDA_CHECK_AND_EXIT(cudaFree(inputs));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    // Calculate reference
    auto reference_host_output = example::reference_gemm<BLAS>(alpha, host_a, host_b, beta, host_c);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());

    // Check against reference
    if (example::check_error<BLAS>(host_output, reference_host_output, true, true)) {
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
