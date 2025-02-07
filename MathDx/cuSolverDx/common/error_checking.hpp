#ifndef CUSOLVERDX_EXAMPLE_COMMON_ERROR_CHECKING_HPP
#define CUSOLVERDX_EXAMPLE_COMMON_ERROR_CHECKING_HPP

#include <cmath>
#include <iostream>

#include <type_traits>

#include <cusolverdx.hpp>

#include "numeric.hpp"

namespace common {

    template<typename ResultType, typename ReferenceType>
    double check_error(const ResultType* data, const ReferenceType* reference, const std::size_t n, bool print = false, bool verbose = false);

    template<typename T>
    bool is_error_acceptable(double tot_rel_err) {
        constexpr bool is_non_float_non_double_a_b_c =
            (!std::is_same_v<T, float> && !std::is_same_v<T, double>) || (!std::is_same_v<T, cusolverdx::complex<float>> && !std::is_same_v<T, cusolverdx::complex<double>>);

        if (is_non_float_non_double_a_b_c) {
            if (tot_rel_err > 1e-2) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        } else { // A,B,C are either float or double
            if (tot_rel_err > 1e-3) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        }
        return std::isfinite(tot_rel_err);
    }
} // namespace common


#endif // CUSOLVERDX_TEST_COMMON_ERROR_CHECKING_HPP
