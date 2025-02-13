#ifndef CUBLASDX_EXAMPLE_NAIVE_REFERENCE_HPP
#define CUBLASDX_EXAMPLE_NAIVE_REFERENCE_HPP

#include <type_traits>
#include "../common.hpp"

namespace example {
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
                                     cublasdx::arrangement              arr_c);
} // namespace example

#endif // CUBLASDX_EXAMPLE_NAIVE_REFERENCE_HPP
