#include "single_gemm_performance.hpp"

// This is an example of testing performance of cuBLASDx device function executing a general matrix multiply (GEMM)
//
//              C = alpha * A * B + beta * C
//
// A, B, and C are matrices. Mixed precisions are supported.
// Note that alpha and beta are expected to have the same precision and type (real or complex value)
// as the matrix C elements.
//
// Input data is generated on host using random number generators, and later copied to
// the global memory. Next, kernel with GEMM is executed, and then the matrix C (the result)
// is copied back to host memory, and verified against cuBLAS.
//
// The measured operation runs multiple times and the average speed is reported.
//
// WARNING: FLOP/s reported by this example are effectively lower than those that can be achieved
// by using those same tiles in fully pipelined and whole-device GEMMs. Please refer to the 
// 11_gemm_device_performance example for more details.

template<unsigned int Arch>
int single_gemm_performance() {
    using namespace cublasdx;

    // Use decoupled precision for input
    using input_type_a = __half;
    using input_type_b = __half;
    using input_type_c = __half;

    // Use decoupled compute precision
    using PA = __half;
    using PB = __half;
    using PC = __half;

    constexpr auto type = cublasdx::type::real;

    // Parameters m, n, k define the dimensions of matrices A, B, and C.
    constexpr unsigned int m = 128;
    constexpr unsigned int n = 128;
    constexpr unsigned int k = 32;

    // Choose block size, or set to 0 to use library-suggested value.
    constexpr unsigned int BlockSize = 128;

    // Flag to use library-suggested leading dimension (potential performance improvement).
    constexpr bool UseSuggestedLD = false;

    // Choose arrangement for A, B, C: row-major or column-major
    constexpr auto a_arrangement = cublasdx::row_major;
    constexpr auto b_arrangement = cublasdx::col_major;
    constexpr auto c_arrangement = cublasdx::row_major;

    // Define the matrix multiplication operation.
    using GEMM = decltype(cublasdx::Size<m, n, k>() +
                          cublasdx::Precision<PA, PB, PC>() +
                          cublasdx::Type<type>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<a_arrangement, b_arrangement, c_arrangement>() +
                          cublasdx::MaxAlignment() +
                          cublasdx::Block() +
                          cublasdx::SM<Arch>());

    bool verbose = true;
    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream))
    int status = benchmark_mixed_precision_gemm<GEMM,
                                                input_type_a,
                                                input_type_b,
                                                input_type_c,
                                                Arch,
                                                BlockSize,
                                                UseSuggestedLD>
                                                (stream, verbose);

    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    return status;
}

struct single_gemm_performance_functor {
    template<int Arch>
    int operator()(std::integral_constant<int, Arch>) {
        return single_gemm_performance<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner(single_gemm_performance_functor{});
}
