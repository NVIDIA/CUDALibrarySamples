// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cufft.h>
#include <cufftXt.h>

#include <cstddef>
#include <vector>

#include "cufft_lto_store_multiply.h"
#include "device_buffer.hpp"
#include "error_checks.hpp"

// cuFFT reference for the convolution examples. The frequency-domain product
// stays on device, like in the kernels, through an LTO store callback.
namespace mathdx_examples::cufft {

// Real samples as complex values.
inline std::vector<cufftComplex> complex_from_real(const std::vector<float>& values) {
    std::vector<cufftComplex> result(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        result[index] = { values[index], 0.0f };
    }
    return result;
}

// Interleaved {real, imaginary} samples as complex values.
inline std::vector<cufftComplex> complex_from_interleaved(const std::vector<float>& values) {
    std::vector<cufftComplex> result(values.size() / 2);
    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = { values[2 * index], values[2 * index + 1] };
    }
    return result;
}

// The scaled real parts.
inline std::vector<float> real_parts(const std::vector<cufftComplex>& values, float scale) {
    std::vector<float> result(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        result[index] = values[index].x * scale;
    }
    return result;
}

// The scaled samples, interleaved {real, imaginary}.
inline std::vector<float> interleaved(const std::vector<cufftComplex>& values, float scale) {
    std::vector<float> result(2 * values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        result[2 * index] = values[index].x * scale;
        result[2 * index + 1] = values[index].y * scale;
    }
    return result;
}

// Convolves `signal` with `filter`, the product formed by the signal plan's store
// callback. The transforms are `batch` contiguous FFTs of sizes `n`. Unnormalized.
inline std::vector<cufftComplex> convolve(const std::vector<cufftComplex>& signal,
                                          const std::vector<cufftComplex>& filter,
                                          std::vector<int> n,
                                          int batch = 1) {
    device_buffer<cufftComplex> device_signal(signal.size());
    device_buffer<cufftComplex> device_filter(filter.size());
    device_signal.copy_from_host(signal.data(), signal.size());
    device_filter.copy_from_host(filter.data(), filter.size());
    cufftComplex* device_filter_pointer = device_filter.data();

    cufftHandle filter_plan = 0;
    cufftHandle signal_plan = 0;
    cufftHandle inverse_plan = 0;
    CUFFT_CHECK(cufftCreate(&filter_plan));
    CUFFT_CHECK(cufftCreate(&signal_plan));
    CUFFT_CHECK(cufftCreate(&inverse_plan));
    // cuFFT dereferences the user-data pointer at execution time; JIT callbacks
    // are set before the plan is made.
    CUFFT_CHECK(cufftXtSetJITCallback(signal_plan,
                                      "cufftdx_tile_store_multiply",
                                      cufft_lto_store_multiply,
                                      sizeof(cufft_lto_store_multiply),
                                      CUFFT_CB_ST_COMPLEX,
                                      reinterpret_cast<void**>(&device_filter_pointer)));
    size_t work_size = 0;
    for (const cufftHandle plan : { filter_plan, signal_plan, inverse_plan }) {
        CUFFT_CHECK(cufftMakePlanMany(plan,
                                      static_cast<int>(n.size()),
                                      n.data(),
                                      nullptr /*inembed*/,
                                      1 /*istride*/,
                                      0 /*idist*/,
                                      nullptr /*onembed*/,
                                      1 /*ostride*/,
                                      0 /*odist*/,
                                      CUFFT_C2C,
                                      batch,
                                      &work_size));
    }

    CUFFT_CHECK(cufftExecC2C(filter_plan, device_filter.data(), device_filter.data(), CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(signal_plan, device_signal.data(), device_signal.data(), CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(inverse_plan, device_signal.data(), device_signal.data(), CUFFT_INVERSE));

    std::vector<cufftComplex> output(signal.size());
    device_signal.copy_to_host(output.data(), output.size());
    CUFFT_CHECK(cufftDestroy(filter_plan));
    CUFFT_CHECK(cufftDestroy(signal_plan));
    CUFFT_CHECK(cufftDestroy(inverse_plan));
    return output;
}

} // namespace mathdx_examples::cufft
