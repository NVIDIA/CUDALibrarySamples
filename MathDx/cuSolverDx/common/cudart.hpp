#ifndef CUSOLVERDX_EXAMPLE_COMMON_CUDART_HPP
#define CUSOLVERDX_EXAMPLE_COMMON_CUDART_HPP

#include "macros.hpp"
#include <cuComplex.h>
#include <cusolverDn.h>

namespace common {
    inline unsigned int get_cuda_device_arch() {
        int device;
        CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));

        int major = 0;
        int minor = 0;
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

        return static_cast<unsigned>(major) * 100 + static_cast<unsigned>(minor) * 10;
    }

    inline unsigned int get_multiprocessor_count(int device) {
        int multiprocessor_count = 0;
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
        return multiprocessor_count;
    }

    inline unsigned int get_multiprocessor_count() {
        int device = 0;
        CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));
        return get_multiprocessor_count(device);
    }

    //============================
    // type traits
    //============================
    template<typename T>
    struct traits;

    template<>
    struct traits<float> {
        // scalar type
        typedef float T;
        typedef T     S;

        static constexpr T            zero           = 0.f;
        static constexpr cudaDataType cuda_data_type = CUDA_R_32F;
#if CUDART_VERSION >= 11000
        static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_R_32F;
#endif

        inline static S abs(T val) {
            return fabs(val);
        }

        template<typename RNG>
        inline static T rand(RNG& gen) {
            return (S)gen();
        }

        inline static T add(T a, T b) {
            return a + b;
        }

        inline static T mul(T v, S f) {
            return v * f;
        }
    };

    template<>
    struct traits<double> {
        // scalar type
        typedef double T;
        typedef T      S;

        static constexpr T            zero           = 0.;
        static constexpr cudaDataType cuda_data_type = CUDA_R_64F;
#if CUDART_VERSION >= 11000
        static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_R_64F;
#endif

        inline static S abs(T val) {
            return fabs(val);
        }

        template<typename RNG>
        inline static T rand(RNG& gen) {
            return (S)gen();
        }

        inline static T add(T a, T b) {
            return a + b;
        }

        inline static T mul(T v, S f) {
            return v * f;
        }
    };

    template<>
    struct traits<cuFloatComplex> {
        // scalar type
        typedef float          S;
        typedef cuFloatComplex T;

        static constexpr T            zero           = {0.f, 0.f};
        static constexpr cudaDataType cuda_data_type = CUDA_C_32F;
#if CUDART_VERSION >= 11000
        static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_C_32F;
#endif

        inline static S abs(T val) {
            return cuCabsf(val);
        }

        template<typename RNG>
        inline static T rand(RNG& gen) {
            return make_cuFloatComplex((S)gen(), (S)gen());
        }

        inline static T add(T a, T b) {
            return cuCaddf(a, b);
        }
        inline static T add(T a, S b) {
            return cuCaddf(a, make_cuFloatComplex(b, 0.f));
        }

        inline static T mul(T v, S f) {
            return make_cuFloatComplex(v.x * f, v.y * f);
        }
    };

    template<>
    struct traits<cuDoubleComplex> {
        // scalar type
        typedef double          S;
        typedef cuDoubleComplex T;

        static constexpr T            zero           = {0., 0.};
        static constexpr cudaDataType cuda_data_type = CUDA_C_64F;
#if CUDART_VERSION >= 11000
        static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_C_64F;
#endif

        inline static S abs(T val) {
            return cuCabs(val);
        }

        template<typename RNG>
        inline static T rand(RNG& gen) {
            return make_cuDoubleComplex((S)gen(), (S)gen());
        }

        inline static T add(T a, T b) {
            return cuCadd(a, b);
        }
        inline static T add(T a, S b) {
            return cuCadd(a, make_cuDoubleComplex(b, 0.));
        }

        inline static T mul(T v, S f) {
            return make_cuDoubleComplex(v.x * f, v.y * f);
        }
    };
} // namespace common

#endif // CUSOLVERDX_EXAMPLE_COMMON_CUDART_HPP
