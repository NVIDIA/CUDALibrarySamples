/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <type_traits>
#include <cmath>
#include <cuda_fp16.h>

#include "nvcomp.hpp"
#include "nvcomp/bitcomp.h"
#include "nvcomp/native/bitcomp.h"

namespace nvcomp
{

namespace
{

template <typename T>
void verifyLossyCompressionInternal(const T* input, const T* output, const size_t num_elements, const double delta)
{
  for (size_t j = 0; j < num_elements; j++) {
    const T original = input[j];
    const T reconstructed = output[j];
    const double original_comp = static_cast<double>(original);
    const double reconstructed_comp = static_cast<double>(reconstructed);

    // Skip special values (NaN, infinity) as quantization doesn't apply to them
    const bool is_nan = std::isnan(original_comp);
    const bool is_inf = std::isinf(original_comp);
    if (is_nan || is_inf) {
      if ((is_nan && !std::isnan(reconstructed_comp)) || (is_inf && !std::isinf(reconstructed_comp))) {
        throw nvcomp::NVCompException(
            nvcompErrorInvalidValue, "Lossy bitcomp compression produced invalid special value");
      }
      continue;
    }

    const double diff = std::fabs(original_comp - reconstructed_comp);
    const double error_bound = std::is_same<T, half>::value ? static_cast<double>(__hdiv(__double2half(delta), __double2half(2.0))) : delta / 2.0;
    if (diff > error_bound) {
      std::cout << std::setprecision(std::numeric_limits<T>::max_digits10)
                << "Original: " << original_comp
                << ", Quantized: " << reconstructed_comp << ", Delta: " << delta
                << ", Diff: " << diff << std::endl;
      throw nvcomp::NVCompException(nvcompErrorInvalidValue, "Lossy bitcomp compression error exceeds allowed delta");
    }
  }
}

bitcompDataType_t getLossyDataType(const int fp_bits)
{
  switch (fp_bits) {
  case 16:
    return BITCOMP_FP16_DATA;
  case 32:
    return BITCOMP_FP32_DATA;
  case 64:
    return BITCOMP_FP64_DATA;
  default:
    throw nvcomp::NVCompException(nvcompErrorInvalidValue, "Unsupported data type for lossy bitcomp compression");
  }
}

void verifyLossyCompression(
    const void* input,
    const void* output,
    const size_t num_bytes,
    const double delta,
    const bitcompDataType_t data_type)
{
  switch (data_type) {
  case BITCOMP_FP16_DATA:
    assert(num_bytes % sizeof(half) == 0);
    verifyLossyCompressionInternal(
        reinterpret_cast<const half*>(input),
        reinterpret_cast<const half*>(output),
        num_bytes / sizeof(half),
        delta);
    break;
  case BITCOMP_FP32_DATA:
    assert(num_bytes % sizeof(float) == 0);
    verifyLossyCompressionInternal(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(output),
        num_bytes / sizeof(float),
        delta);
    break;
  case BITCOMP_FP64_DATA:
    assert(num_bytes % sizeof(double) == 0);
    verifyLossyCompressionInternal(
        reinterpret_cast<const double*>(input),
        reinterpret_cast<const double*>(output),
        num_bytes / sizeof(double),
        delta);
    break;
  default:
    throw nvcomp::NVCompException(nvcompErrorInvalidValue, "Unsupported data type for lossy bitcomp compression");
  }
}

} // namespace
} // namespace nvcomp
