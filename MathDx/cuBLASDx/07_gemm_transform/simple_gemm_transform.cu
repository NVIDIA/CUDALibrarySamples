#include <iostream>
#include <vector>


#include <cuda_runtime_api.h>

#define CUBLASDX_IGNORE_NVBUG_5218000_ASSERT
#include <cublasdx.hpp>

#include "../common/common.hpp"
#include "../reference/reference.hpp"


// Requirements of the transform ops functors:
// * The input type must be the same as the value type of the corresponding matrix.
// * The return type must be convertible to the value type of the corresponding matrix.
struct negate {
    template <class T>
    __device__ __host__ constexpr
    T operator()(const T arg) const {
        return -arg;
    }
};

template<unsigned N>
struct multiply_by {
    template <class T>
    __device__ __host__ constexpr
    T operator()(const T arg) const {
        return static_cast<T>(N) * arg;
    }
};

template<class BLAS,
         class ValueType = typename example::uniform_value_type_t<BLAS>,
         class ALoadOp  = cublasdx::identity,
         class BLoadOp  = cublasdx::identity,
         class CLoadOp  = cublasdx::identity,
         class CStoreOp = cublasdx::identity>
__launch_bounds__(BLAS::max_threads_per_block) //
    __global__                                 //
    void gemm_kernel(const ValueType* a,
                     const ValueType* b,
                     const ValueType* c,
                     const ValueType  alpha,
                     const ValueType  beta,
                     ValueType*       output,
                     const ALoadOp&   a_load_op = {},
                     const BLoadOp&   b_load_op = {},
                     const CLoadOp&   c_load_op = {},
                     const CStoreOp&  c_store_op = {}) {
    using value_type = ValueType;
    extern __shared__ __align__(16) char smem[];

    auto a_global_tensor = cublasdx::make_tensor(a, BLAS::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, BLAS::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, BLAS::get_layout_gmem_c());

    auto [smem_a, smem_b] = cublasdx::slice_shared_memory_ab<BLAS>(smem);
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, BLAS::get_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, BLAS::get_layout_smem_b());

    using alignment = cublasdx::alignment_of<BLAS>;
    cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<BLAS, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy_wait();

    auto [c_frag, partitioner] =
        BLAS().execute(a_shared_tensor, b_shared_tensor, a_load_op, b_load_op);

    auto d_frag = partitioner.make_accumulator_fragment();
    cublasdx::copy_fragment<alignment::c>(c_global_tensor, d_frag, partitioner);
    cublasdx::transform(d_frag, c_load_op);
    cublasdx::axpby(alpha, c_frag, beta, d_frag);
    cublasdx::transform(d_frag, c_store_op);

    auto out_global_tensor = cublasdx::make_tensor(output, BLAS::get_layout_gmem_c());
    cublasdx::copy_fragment<alignment::c>(d_frag, out_global_tensor, partitioner);
}

// This is an example of fp16 general matrix-matrix multiplication (GEMM) performed
// in a single CUDA block:
//
//              C = op_c_store(alpha * op_a(A) * op_b(B) + beta * op_c_load(C))
//
// * A, B, and C are matrices containing real single precision floating-point values.
// * alpha and beta are real single precision floating-point values.
// * op_a, op_b, op_c_load, op_c_store are transform operations (functors) applying element-wisely
//   to the input matrix. cuBLASDx provides two element-wise functors, cublasdx::identity and
//   cublasdx::conjugate, where the later returns the complex conjugate of the input value
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
    constexpr unsigned int n = 4;
    constexpr unsigned int k = 2;

    // Leading dimensions defined for matrices A, B, C
    constexpr unsigned int lda = 16;
    constexpr unsigned int ldb = 16;
    constexpr unsigned int ldc = 16;

    // Transform ops
    auto a_load_op  = negate{};
    auto b_load_op  = multiply_by<2>{};
    auto c_load_op  = multiply_by<2>{};
    auto c_store_op = negate{};

    // Selected CUDA block size (1D)
    constexpr unsigned int block_size = 256;

    // GEMM definition using cuBLASDx operators:
    // 1. The size, the precision, and the type (real or complex) are set.
    // 2. The BLAS function is selected: MM (matrix multiplication).
    // 3. Block operator informs that GEMM should be performed on CUDA block level.
    // 4. BlockDim operator sets CUDA block dimensions that the kernel will be executed with.
    // 5. Targeted CUDA compute capability is selected with SM operator.
    using BLAS = decltype(cublasdx::Size<m, n, k>() +
                          cublasdx::Precision<__half>() +
                          cublasdx::Type<cublasdx::type::real>() +
                          cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>() +
                          cublasdx::LeadingDimension<lda, ldb, ldc>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<block_size>() +
                          cublasdx::SM<Arch>());

    using value_type = typename example::uniform_value_type_t<BLAS>;

    // Allocate managed memory for a, b, c, and output

    constexpr auto global_a_size = example::global_memory_size_of<BLAS>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<BLAS>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<BLAS>::c_size;

    value_type  alpha = value_type(1.0);
    value_type  beta  = value_type(1.0);

    // Fill the A, B, C matrices with random values
    auto host_a = example::get_random_data<value_type>(0.1, 1.0, global_a_size);
    auto host_b = example::get_random_data<value_type>(0.1, 1.0, global_b_size);
    auto host_c = example::get_random_data<value_type>(0.1, 1.0, global_c_size);

    example::device_vector<value_type> a = host_a;
    example::device_vector<value_type> b = host_b;
    example::device_vector<value_type> c = host_c;

    example::device_vector<value_type> output(c.size());

    // Increase max dynamic shared memory for the kernel if needed
    auto shared_size = cublasdx::get_shared_storage_size<BLAS>();
    CUDA_CHECK_AND_EXIT(
        cudaFuncSetAttribute(gemm_kernel<BLAS>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

    // Execute kernel
    gemm_kernel<BLAS><<<1, BLAS::block_dim, shared_size>>>
        (a.data(),
         b.data(),
         c.data(),
         alpha,
         beta,
         output.data(),
         a_load_op, b_load_op, c_load_op, c_store_op);

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<value_type> host_output = output;

    // Calculate reference
    cudaStream_t str = 0;
    auto reference_host_output = example::reference_gemm<BLAS>(alpha, host_a, host_b, beta, host_c, str, a_load_op, b_load_op, c_load_op, c_store_op);

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
