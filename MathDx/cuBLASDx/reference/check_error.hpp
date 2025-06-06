#ifndef CUBLASDX_EXAMPLE_CHECK_ERROR_HPP
#define CUBLASDX_EXAMPLE_CHECK_ERROR_HPP

#define CUBLASDX_EXAMPLE_NO_THRUST
#include <type_traits>
#include "../common/common.hpp"

namespace example {
    template<typename TC, typename TA = TC, typename TB = TC>
    bool is_error_acceptable(double tot_rel_err) {
        if (!std::isfinite(tot_rel_err)) {
            return false;
        }

        constexpr bool is_fp8_a_b_c =  (commondx::is_floating_point_v<TA> and not is_complex<TA>() and sizeof(TA) == 1) ||
                                       (commondx::is_floating_point_v<TB> and not is_complex<TB>() and sizeof(TB) == 1) ||
                                       (commondx::is_floating_point_v<TC> and not is_complex<TC>() and sizeof(TC) == 1);

        constexpr bool is_fp8_a_b_c_complex =  (commondx::is_floating_point_v<TA> and is_complex<TA>() and sizeof(TA) == 2) ||
                                               (commondx::is_floating_point_v<TB> and is_complex<TB>() and sizeof(TB) == 2) ||
                                               (commondx::is_floating_point_v<TC> and is_complex<TC>() and sizeof(TC) == 2);

        constexpr bool is_bf16_a_b_c = std::is_same_v<TA, __nv_bfloat16> || std::is_same_v<TB, __nv_bfloat16> ||
                                       std::is_same_v<TC, __nv_bfloat16>;

        constexpr bool is_bf16_a_b_c_complex =
            std::is_same_v<TA, cublasdx::complex<__nv_bfloat16>> ||
            std::is_same_v<TB, cublasdx::complex<__nv_bfloat16>> ||
            std::is_same_v<TC, cublasdx::complex<__nv_bfloat16>>;

        constexpr bool is_integral =
            commondx::is_integral_v<TA> and
            commondx::is_integral_v<TB> and
            commondx::is_integral_v<TC>;

        constexpr bool is_non_float_non_double_a_b_c =
            (!std::is_same_v<TA, float> && !std::is_same_v<TA, double>) ||
            (!std::is_same_v<TB, float> && !std::is_same_v<TB, double>) ||
            (!std::is_same_v<TC, float> && !std::is_same_v<TC, double>) ||
            (!std::is_same_v<TA, cublasdx::complex<float>> && !std::is_same_v<TA, cublasdx::complex<double>>) ||
            (!std::is_same_v<TB, cublasdx::complex<float>> && !std::is_same_v<TB, cublasdx::complex<double>>) ||
            (!std::is_same_v<TC, cublasdx::complex<float>> && !std::is_same_v<TC, cublasdx::complex<double>>);

        if constexpr(is_integral) {
            if (tot_rel_err != 0.0) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        } else if (is_fp8_a_b_c) {
            if (tot_rel_err > 7e-2) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        } else if (is_fp8_a_b_c_complex) {
            if (tot_rel_err > 1e-1) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        } else if (is_bf16_a_b_c_complex) {
            if (tot_rel_err > 6e-2) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        } else if (is_bf16_a_b_c) {
            if (tot_rel_err > 5e-2) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        } else if (is_non_float_non_double_a_b_c) {
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
        return true;
    }

    template<class T>
    constexpr bool is_reference_type() {
        return std::is_same_v<T, detail::get_reference_value_type_t<T>>;
    }

    template<typename T1, typename T2>
    std::enable_if_t<is_reference_type<T1>() and is_reference_type<T2>() and std::is_same_v<T1, T2>, double>
    calculate_error(const std::vector<T1>& data, const std::vector<T2>& reference, bool verbose = false, bool print = false);

    template<typename T1, typename T2>
    std::enable_if_t<not is_reference_type<T1>() or not std::is_same_v<T1, T2>, double>
    calculate_error(const std::vector<T1>& data, const std::vector<T2>& reference, bool verbose = false, bool print = false) {
        using ref_t = detail::get_reference_value_type_t<T2>;
        std::vector<ref_t> input_upcasted;
        std::transform(std::cbegin(data), std::cend(data), std::back_inserter(input_upcasted), converter<ref_t>{});

        // if only the input data required conversion, run comparison
        if constexpr(is_reference_type<T2>()) {
            return calculate_error(input_upcasted, reference, verbose, print);
        }
        // else, if the reference was also calculated in lower precision,
        // also upcast it and only then run the comparison
        else {
            std::vector<ref_t> reference_upcasted;
            std::transform(std::cbegin(reference), std::cend(reference), std::back_inserter(reference_upcasted), converter<ref_t>{});
            return calculate_error(input_upcasted, reference_upcasted, verbose, print);
        }
    }

    template<class TA, class TB, class TC, typename ResT, typename RefT>
    bool check_error_custom(const std::vector<ResT>& results, const std::vector<RefT>& reference, bool verbose = false, bool print = false) {
        [[ maybe_unused ]] constexpr bool is_floating = commondx::is_floating_point_v<RefT>;
        [[ maybe_unused ]] constexpr bool is_integral = commondx::is_integral_v<RefT>;

        auto ret = false;

        if constexpr (is_floating) {
             double error = calculate_error(results, reference, verbose, print);
             ret = is_error_acceptable<TC, TA, TB>(error);
        } else if constexpr (is_integral) {
            // If the input was integral, then we want absolute equality
            if(print) {
                std::cout << "Ref\tRes\n";
            }
            ret = std::equal(reference.cbegin(), reference.cend(), results.cbegin(), [print](auto ref, auto res) {
                if(print) {
                    if constexpr(is_complex<decltype(ref)>()) {
                        std::cout << ref.real() << "," << ref.imag() << "\t"
                                  << res.real() << "," << res.imag() << "\n";
                    } else {
                        std::cout << ref << "\t" << res << "\n";
                    }
                }
                if constexpr(is_complex<decltype(ref)>()) {
                    return ref.real() == res.real() and
                           ref.imag() == res.imag();
                } else {
                    return ref == res;
                }
            });
        } else {
            static_assert(is_floating or is_integral, "Reference and result must either both be integral or floating point.");
        }

        return ret;
    }
    template<class BLAS, typename ResT, typename RefT>
    bool check_error(const std::vector<ResT>& results, const std::vector<RefT>& reference, bool verbose = false, bool print = false) {
        using a_prec_t = typename cublasdx::precision_of<BLAS>::a_type;
        using b_prec_t = typename cublasdx::precision_of<BLAS>::b_type;
        using c_prec_t = typename cublasdx::precision_of<BLAS>::c_type;
        return check_error_custom<a_prec_t, b_prec_t, c_prec_t, ResT, RefT>(results, reference, verbose, print);
    }

    template<typename ResT, typename RefT>
    bool check_error(const std::vector<ResT>& results, const std::vector<RefT>& reference, bool verbose = false, bool print = false) {
        return check_error_custom<ResT, ResT, ResT>(results, reference, verbose, print);
    }
}

#endif // CUBLASDX_EXAMPLE_CHECK_ERROR_HPP
