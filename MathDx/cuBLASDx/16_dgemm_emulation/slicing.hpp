#pragma once

#include <iostream>
#include <utility>
#include <cassert>
#include <cuda_runtime.h>

union double_structure {
    double d;
    struct float64 {
        unsigned int mantissa_lo : 32;
        unsigned int mantissa_hi : 20;
        unsigned int exponent : 11;
        unsigned int sign : 1;
    } s;
};

static constexpr int bias = 1023;

/*
 * Signed magnitudes of length N only allow for (N-1) of effective storage.
 */
template<class T>
__device__ __host__ constexpr int get_width() {
    if constexpr (std::is_signed<T>()) {
        return 8 * sizeof(T) - 1;
    } else {
        return 8 * sizeof(T);
    };
}

inline __host__ __device__ int64_t div_up(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

inline __device__ __host__ int rz_width(const double_structure& em) {
    return em.s.exponent + 1 - bias;
}
// Length of a bits before the decimal point. i.e., bit width if casted to infinite-length int type.
inline __device__ __host__ int rz_width(const double d) {
    double_structure em {d};
    return rz_width(em);
}

static __device__ __host__ constexpr int64_t ipow_p(int64_t base, int exp, int64_t ans = 1) {
    return exp < 1 ? ans : ipow_p(base * base, exp / 2, (exp % 2) ? ans * base : ans);
}

static __device__ __host__ constexpr double ipow(int base, int exp) {
    return exp > 0 ? ipow_p(base, exp) : 1.0 / ipow_p(base, -exp);
}

template<class T>
__device__ __host__ constexpr int max_exponent();

// scale to numbers no bigger than 256
template<>
__device__ __host__ constexpr int max_exponent<uint8_t>() {
    return 8;
}

// scale to numbers no bigger than 128
template<>
__device__ __host__ constexpr int max_exponent<int8_t>() {
    return 7;
}

int32_t get_exponent(double val) {
    double_structure em = {val};

    int em_exponent = (em.s.exponent + 1 - bias);

    if (em.s.mantissa_hi & (63 << 14))
        em_exponent++;

    return em_exponent;
}

// An implementation of ldexp() to be used in scaling double-precision numbers obtained from unpacking slices
// in the epilogue. The resulting double values must be finite and normalized, and so the fast path should
// simply adjust the exponent field of the value, so long as the result is also finite and normalized.
inline __device__ __host__ void epilogue_ldexp(double_structure& em, int exp) {
    static constexpr int exp_max             = bias - 1;
    int                  previous_exp_biased = static_cast<int>(em.s.exponent);
    if (0 < previous_exp_biased && 0 < previous_exp_biased + exp && previous_exp_biased + exp <= exp_max + bias) {
        em.s.exponent += exp;
        return;
    }
    em.d = ldexp(em.d, exp);
}

/*
 * Returns the exponent shift to be applied to a row/colum
 * based on the max(abs()) on that row/column.
 * Naively, this scaling factor would be just the exponent
 * of max(abs()) but we do some other tricks to account for
 * the encoding of the signed magnitude only on the leading
 * slice among other things..
 */
inline __device__ __host__ int32_t max_to_exponent_shift(double row_col_max) {
    static constexpr int scale_max_exponent = max_exponent<int8_t>();

    double_structure em = {row_col_max};

    int32_t em_exponent = (em.s.exponent + 1 - bias);

    if (em.s.mantissa_hi & (63 << 14))
        em_exponent++;

    return (scale_max_exponent - em_exponent);
}

/*
 * slices up the input value "val" in "nslices" of type "SliceValueType".
 * Before slicing the number, the exponent of "val" is shifted based on
 * the value of "exponent_shift" which has been computed using the
 * max_to_exponent_shift function based on the max(abs()) of the relevant
 * row/column of A/B.
 *
 * On exit, the first value of the returned array contains the most
 * significant slice.
 */
template<class SliceValueType, unsigned nslices>
__device__ __host__ std::array<SliceValueType, nslices> slices_from_fp64(double val, int32_t exponent_shift) {
    static_assert(std::is_integral<SliceValueType>());
    static_assert(std::is_signed<SliceValueType>());
    static_assert(nslices >= 0);

    std::array<SliceValueType, nslices> slices     = {0};
    std::array<SliceValueType, nslices> slices_out = {0};

    static constexpr double normalization_factor = 0x1.0p52;
    // Normalise denormalised numbers, but store the effective exponent in its own variable,
    // allowing for representation of fp64 denorms as normalised numbers.

    int     skip_slices = 0;
    int64_t r           = 0;

    uint8_t reg_pack = 0;

    double_structure r0                  = {val};
    int              denorm_compensation = 0;
    if (r0.s.exponent == 0) {
        if (r0.d == 0.0) {
            skip_slices = nslices;
            r           = 0;
        }
        /* round to nearest is the default behavior on CPU, afaik... */
        r0.d                = (r0.d * normalization_factor);
        denorm_compensation = -52;
    }
    int exp = r0.s.exponent + exponent_shift + denorm_compensation - bias;
    exp += (nslices - 1) * get_width<uint8_t>(); // Use all 8 bits.
                                                 // Adjust casting range.
    int extra_width = (exp + 1) - 63;
    extra_width     = extra_width > 0 ? extra_width : 0;
    skip_slices     = div_up(extra_width, get_width<uint8_t>());
    exp -= skip_slices * get_width<uint8_t>();
    // End: Adjust casting range.
    // Handle exp outside of double range
    if (exp < 0) {
        r = 0;
    } else {
        r0.s.exponent = (unsigned int)(exp + bias);
        r             = static_cast<int64_t>(r0.d);
    }

    for (int64_t _i = 0; _i < nslices; _i++) {
        int64_t i = nslices - 1 - _i;

        if (_i < skip_slices) {
            reg_pack = 0;
        } else {
            reg_pack  = static_cast<uint8_t>(r);
            slices[i] = static_cast<int8_t>(reg_pack);
            r         = (r >> get_width<uint8_t>()) + (reg_pack >> get_width<int8_t>());
        }
    }

    return slices;
}

/*
 * Recovers fp64 value from slices and exponent shifts.
 *
 * T  : slice type
 * nslices : number of slices
 * BT : base slice type. This is the data type that was used to slice up the
 *      fp64 values in the first place.
 *
 * slices : array of "nslices" values of type T as computed by "slices_from_fp64".
 * exponent_shift : exponent shift (see slices_from_fp64).
 *
 * Why do we need T and BT? The original or base (BT) slicing type defines the
 * offset between consecutive slices. The slice type (T) defines the type of
 * the current storage container. In the algorithm we find cases there the
 * storage is different to the base type, for example, inputs to IMMA are
 * int8 and outputs are int32. In this case, if we wanted to reconstruct
 * the FP64 value, we'll call this routine with T=int32 and BT=int8.
 */
template<typename T, unsigned nslices, typename BT>
__device__ __host__ __forceinline__ double slices_to_fp64(std::array<T, nslices> pack_c, int32_t exponent_shift) {
    static_assert(std::is_integral<T>());
    static_assert(std::is_signed<T>());
    static_assert(std::is_integral<BT>());
    static_assert(std::is_signed<BT>());
    static_assert(nslices > 1);

    /* In some instances, we use the unsigned value type to leverage all bits for storage */
    using UT  = std::make_unsigned_t<T>;
    using UBT = std::make_unsigned_t<BT>;

    double ko      = {1};
    double value_i = {0};

#ifdef __CUDA_ARCH__
#    pragma unroll
#endif
    for (unsigned diag = 0; diag < nslices; diag++) {
        value_i += ko * static_cast<double>(pack_c[diag]);
        ko *= pow(2.0, -get_width<UBT>());
    }
    double_structure value = {value_i};
    epilogue_ldexp(value, -exponent_shift);

    return value.d;
}


/*
 * This function is a building block to reconstruct an FP64 number from the slices.
 * Instead of receiving the set of slices and adding them to an FP64 number,
 * this function gets a single slice (the NTH-slice) and returns it as an FP64 value.
 *
 * In this way, one could use this function to compute and accumulate the contributions
 * from the slices separately.
 *
 * Remark: when reconstructing an FP64 number accumulate the least significant
 * diagonals first to avoid catastrophic cancellation.
 */
template<typename T, typename BT>
__forceinline__ __device__ __host__ double nth_slice_to_fp64(int32_t nth, T nth_slice, int32_t exponent_shift) {
    static_assert(std::is_integral<T>());
    static_assert(std::is_signed<T>());
    static_assert(std::is_integral<BT>());
    static_assert(std::is_signed<BT>());
    assert(nth >= 0);

    /* In some instances, we use the unsigned value type to leverage all bits for storage */
    using UT  = std::make_unsigned_t<T>;
    using UBT = std::make_unsigned_t<BT>;

    double ko = pow(2.0, -get_width<UBT>() * nth);

    double           value_i = ko * static_cast<double>(nth_slice);
    double_structure value   = {value_i};
    epilogue_ldexp(value, -exponent_shift);
    return value.d;
}
