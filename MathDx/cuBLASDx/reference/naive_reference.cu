#define CUBLASDX_EXAMPLE_NO_THRUST
#include "naive_reference.hpp"

namespace example {

    template<class AccumType,
             class AEngine, class ALayout,
             class BEngine, class BLayout>
    CUTE_HOST_DEVICE
    void dot(AccumType                           & accum,
             cute::Tensor<AEngine, ALayout> const& tensor_a,
             cute::Tensor<BEngine, BLayout> const& tensor_b) {
        static_assert(cute::rank(decltype(tensor_a.layout()){}) == cute::rank(decltype(tensor_b.layout()){}) == 1);
        for (unsigned int k_iter = 0; k_iter < cute::size<0>(tensor_a); ++k_iter) {
            // Cast needed to emulate --> higher += lower * lower
            accum += convert<AccumType>(tensor_a(k_iter)) * convert<AccumType>(tensor_b(k_iter));
        }
    }

    template<class AEngine, class ALayout,
             class BEngine, class BLayout,
             class CEngine, class CLayout,
             class Alpha,   class Beta>
    __global__
    void reference_gemm_naive_kernel(Alpha                                alpha,
                                     cute::Tensor<AEngine, ALayout> const tensor_a,
                                     cute::Tensor<BEngine, BLayout> const tensor_b,
                                     Beta                                 beta,
                                     cute::Tensor<CEngine, CLayout>       tensor_c) {

        using value_type_c = typename CEngine::value_type;

        auto idx_m = blockIdx.x * blockDim.x + threadIdx.x;
        auto idx_n = blockIdx.y * blockDim.y + threadIdx.y;

        if(idx_m < cute::size<0>(tensor_c) and idx_n < cute::size<1>(tensor_c)) {
            value_type_c acc = convert<value_type_c>(0.f);
            dot(acc, tensor_a(idx_m, cute::_), tensor_b(cute::_, idx_n));
            tensor_c(idx_m, idx_n) = alpha * acc + beta * tensor_c(idx_m, idx_n);
        }
    }


    template<class AEngine, class ALayout,
             class BEngine, class BLayout,
             class CEngine, class CLayout,
             class Alpha,   class Beta>
    void reference_gemm_naive_host(Alpha                                 alpha,
                                   cute::Tensor<AEngine, ALayout> const& tensor_a,
                                   cute::Tensor<BEngine, BLayout> const& tensor_b,
                                   Beta                                  beta,
                                   cute::Tensor<CEngine, CLayout>      & tensor_c) {
        assert(cute::size<1>(tensor_a) == cute::size<0>(tensor_b));
        using value_type_c = typename CEngine::value_type;

        for (unsigned int idx_m = 0; idx_m < cute::size<0>(tensor_c); ++idx_m) {
            for (unsigned int idx_n = 0; idx_n < cute::size<1>(tensor_c); ++idx_n) {
                value_type_c acc = convert<value_type_c>(0.f);
                dot(acc, tensor_a(idx_m, cute::_), tensor_b(cute::_, idx_n));
                tensor_c(idx_m, idx_n) = alpha * acc + beta * tensor_c(idx_m, idx_n);
            }
        }
    }


    template<typename ValueType>
    void reference_gemm_naive_device(const unsigned int                 m,
                                     const unsigned int                 n,
                                     const unsigned int                 k,
                                     const ValueType                    alpha,
                                     example::device_vector<ValueType>& A,
                                     const unsigned int                 lda,
                                     cublasdx::arrangement              arr_a,
                                     example::device_vector<ValueType>& B,
                                     const unsigned int                 ldb,
                                     cublasdx::arrangement              arr_b,
                                     const ValueType                    beta,
                                     example::device_vector<ValueType>& C,
                                     const unsigned int                 ldc,
                                     cublasdx::arrangement              arr_c) {
        auto make_tensor = [](auto ptr, auto sx, auto sy, auto ld, bool col_major) {
            return cute::make_tensor(ptr,
                    cute::make_layout(cute::make_shape(sx, sy),
                                      cute::make_stride(col_major ? 1 : ld,
                                                        col_major ? ld : 1)));
        };

        cute::Tensor tensor_a = make_tensor(A.data(), m, k, lda, arr_a == cublasdx::col_major);
        cute::Tensor tensor_b = make_tensor(B.data(), k, n, ldb, arr_b == cublasdx::col_major);
        cute::Tensor tensor_c = make_tensor(C.data(), m, n, ldc, arr_c == cublasdx::col_major);

        // Decide if device or host execution
        const dim3 block_dim = {16, 16, 1};
        const dim3 grid_dim = {cute::ceil_div(m, block_dim.x), cute::ceil_div(n, block_dim.y), 1};
        reference_gemm_naive_kernel<<<grid_dim, block_dim>>>(alpha, tensor_a, tensor_b, beta, tensor_c);
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    }

    #define REFERENCE_FOR_TYPE(Prec)                                                                   \
        template                                                                                       \
        void reference_gemm_naive_device<Prec>                                                         \
                                        (const unsigned int                 m,                         \
                                         const unsigned int                 n,                         \
                                         const unsigned int                 k,                         \
                                         const Prec                         alpha,                     \
                                         example::device_vector<Prec, void>& A,                         \
                                         const unsigned int                 lda,                       \
                                         cublasdx::arrangement              arr_a,                     \
                                         example::device_vector<Prec, void>& B,                         \
                                         const unsigned int                 ldb,                       \
                                         cublasdx::arrangement              arr_b,                     \
                                         const Prec                         beta,                      \
                                         example::device_vector<Prec, void>& C,                         \
                                         const unsigned int                 ldc,                       \
                                         cublasdx::arrangement              arr_c);                    \
                                                                                                       \
        template                                                                                       \
        void reference_gemm_naive_device<cublasdx::complex<Prec>>                                      \
                                        (const unsigned int                 m,                         \
                                         const unsigned int                 n,                         \
                                         const unsigned int                 k,                         \
                                         const cublasdx::complex<Prec>      alpha,                     \
                                         example::device_vector<cublasdx::complex<Prec>, void>& A,           \
                                         const unsigned int                 lda,                       \
                                         cublasdx::arrangement              arr_a,                     \
                                         example::device_vector<cublasdx::complex<Prec>, void>& B,           \
                                         const unsigned int                 ldb,                       \
                                         cublasdx::arrangement              arr_b,                     \
                                         const cublasdx::complex<Prec>      beta,                      \
                                         example::device_vector<cublasdx::complex<Prec>, void>& C,           \
                                         const unsigned int                 ldc,                       \
                                         cublasdx::arrangement              arr_c);

    REFERENCE_FOR_TYPE(double)
    REFERENCE_FOR_TYPE(int64_t)
    REFERENCE_FOR_TYPE(uint64_t)

} // namespace example
