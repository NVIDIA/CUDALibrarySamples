#ifndef CUBLASDX_EXAMPLE_NAIVE_REFERENCE_HPP
#define CUBLASDX_EXAMPLE_NAIVE_REFERENCE_HPP

#include <type_traits>
#include "../common/common.hpp"

namespace example {
    using unsigned_tuple = cute::tuple<unsigned, unsigned, unsigned>;
    using arr_tuple = cute::tuple<cublasdx::arrangement, cublasdx::arrangement, cublasdx::arrangement>;

    template<typename ValueType>
    void reference_gemm_naive_device(unsigned_tuple           const& gemm_shape,
                                     arr_tuple                const& gemm_arr,
                                     unsigned_tuple           const& gemm_ld,
                                     ValueType                const& alpha,
                                     device_vector<ValueType> const& A,
                                     device_vector<ValueType> const& B,
                                     ValueType                const& beta,
                                     device_vector<ValueType>      & C);
} // namespace example

#endif // CUBLASDX_EXAMPLE_NAIVE_REFERENCE_HPP
