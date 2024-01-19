#include "single_gemm_performance.hpp"

template<unsigned int Arch>
int single_gemm_performance() {
    using namespace cublasdx;

    // Parameters m, n, k define the dimensions of matrices A, B, and C.
    constexpr unsigned int m = 32;
    constexpr unsigned int n = 32;
    constexpr unsigned int k = 64;

    // Choose block size, or set to 0 to use library-suggested value.
    constexpr unsigned int BlockSize = 256;

    // Flag to use library-suggested leading dimension (potential performance improvement).
    constexpr bool UseSuggestedLD = true;

    // Choose precision (__half, float, double) and type (real or complex).
    using precision = __half;
    constexpr auto type = cublasdx::type::complex;

    // Choose transpose mode for A and B: non_transposed or transposed.
    constexpr auto a_transpose_mode = cublasdx::transpose_mode::non_transposed;
    constexpr auto b_transpose_mode = cublasdx::transpose_mode::transposed;

    // Define the matrix multiplication operation.
    using GEMM = decltype(cublasdx::Size<m, n, k>() +
                     cublasdx::Precision<precision>() +
                     cublasdx::Type<type>() +
                     cublasdx::Function<cublasdx::function::MM>() +
                     cublasdx::TransposeMode<a_transpose_mode, b_transpose_mode>() +
                     cublasdx::Block() +
                     cublasdx::SM<Arch>());

    bool verbose = true;
    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream))
    int status = benchmark_single_gemm<GEMM, Arch, BlockSize, UseSuggestedLD>(stream, verbose);
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    return status;
}

template<unsigned int Arch>
struct single_gemm_performance_functor {
    int operator()() { return single_gemm_performance<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<single_gemm_performance_functor>();
}
