// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <iterator>

namespace mathdx_examples {

// Computes ||actual - reference||_2 / ||reference||_2 over [actual_first,
// actual_last). For interleaved complex storage, pass iterators over all scalar
// components.
template <class ActualIterator, class ReferenceIterator>
double relative_l2_error(ActualIterator actual_first, ActualIterator actual_last, ReferenceIterator reference_first) {
    double error_l2_squared = 0.0;
    double reference_l2_squared = 0.0;
    for (; actual_first != actual_last; ++actual_first, ++reference_first) {
        const double actual_value = static_cast<double>(*actual_first);
        const double reference_value = static_cast<double>(*reference_first);
        const double error = actual_value - reference_value;
        error_l2_squared += error * error;
        reference_l2_squared += reference_value * reference_value;
    }

    // A zero reference has no meaningful relative error. Return its absolute
    // L2 error instead, which remains well-defined and useful to callers.
    return reference_l2_squared == 0.0 ? std::sqrt(error_l2_squared)
                                       : std::sqrt(error_l2_squared / reference_l2_squared);
}

template <class ActualRange, class ReferenceRange>
double relative_l2_error(const ActualRange& actual, const ReferenceRange& reference) {
    using std::begin;
    using std::end;
    return relative_l2_error(begin(actual), end(actual), begin(reference));
}

} // namespace mathdx_examples
