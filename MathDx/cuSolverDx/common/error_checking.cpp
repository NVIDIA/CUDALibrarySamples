#include "error_checking.hpp"

namespace common {

    // Copied from CuTe examples and slightly modified
    template<typename T1, typename T2>
    double check_error(const T1* data, const T2* reference, const std::size_t n, bool print, bool verbose) {

        // Use either double or complex<double> for error computation
        using error_type = std::conditional_t<is_complex<T2>(), cusolverdx::complex<double>, double>;

        if (print && verbose) {
            std::cout << "Idx:\t"
                      << "Val\t"
                      << "RefVal\t"
                      << "RelError"
                      << "\n";
        }

        double eps = 1e-200;

        double tot_error_sq    = 0;
        double tot_norm_sq     = 0;
        double tot_ind_rel_err = 0;
        double max_ind_rel_err = 0;
        for (std::size_t i = 0; i < n; ++i) {
            error_type val = convert<error_type>(data[i]);
            error_type ref = convert<error_type>(reference[i]);

            double aref      = detail::abs(ref);
            double diff      = detail::abs(ref - error_type(val));
            double rel_error = diff / (aref + eps);

            // Individual relative error
            tot_ind_rel_err += rel_error;

            // Maximum relative error
            max_ind_rel_err = std::max(max_ind_rel_err, rel_error);

            // Total relative error
            tot_error_sq += diff * diff;
            tot_norm_sq += aref * aref;

            if (print && verbose) {
                if constexpr (is_complex<error_type>()) {
                    std::cout << i << ":\t" << '<' << val.real() << ',' << val.imag() << '>' << "\t" << '<' << ref.real() << ',' << ref.imag() << '>' << "\t" << rel_error << "\n";
                } else {
                    std::cout << i << ":\t" << val << "\t" << ref << "\t" << rel_error << "\n";
                }
            }
        }
        if (print)
            printf("Vector reference  norm: [%.5e]\n", sqrt(tot_norm_sq));

        double tot_rel_err = sqrt(tot_error_sq / (tot_norm_sq + eps));
        if (print)
            printf("Vector  relative error: [%.5e]\n", tot_rel_err);

        double ave_rel_err = tot_ind_rel_err / double(n);
        if (print)
            printf("Average relative error: [%.5e]\n", ave_rel_err);

        if (print)
            printf("Maximum relative error: [%.5e]\n", max_ind_rel_err);

        return tot_rel_err;
    }

#define CUSOLVERDX_DETAIL_CHECK_ERROR_SELF(type) template double check_error<type, type>(const type* data, const type* reference, const std::size_t n, bool print, bool verbose);

#define CUSOLVERDX_DETAIL_CHECK_ERROR(type)                                                                                              \
    template double check_error<type, double>(const type* data, const double* reference, const std::size_t n, bool print, bool verbose); \
    template double check_error<cusolverdx::complex<type>, cusolverdx::complex<double>>(                                                 \
        const cusolverdx::complex<type>* data, const cusolverdx::complex<double>* reference, const std::size_t n, bool print, bool verbose);

    CUSOLVERDX_DETAIL_CHECK_ERROR_SELF(float)
    CUSOLVERDX_DETAIL_CHECK_ERROR_SELF(cusolverdx::complex<float>)
    CUSOLVERDX_DETAIL_CHECK_ERROR_SELF(double)
    CUSOLVERDX_DETAIL_CHECK_ERROR_SELF(cusolverdx::complex<double>)

    CUSOLVERDX_DETAIL_CHECK_ERROR(float)

#undef CUSOLVERDX_DETAIL_CHECK_ERROR_SELF
#undef CUSOLVERDX_DETAIL_CHECK_ERROR


} // namespace common
