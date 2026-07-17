/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// This example demonstrates the nvCOMP Bitcomp *Native API*, which exposes
// capabilities not available through the standard low-level batched API,
// most notably error-bounded lossy compression of floating-point data.
//
// The flow is:
//   1. Create a Bitcomp plan describing the data type, compression mode
//      (lossy FP32 -> signed integers), and algorithm.
//   2. Compress on the GPU with a chosen quantization delta.
//   3. Decompress with the same plan.
//   4. Verify that the maximum reconstruction error is <= delta / 2.
//
// Note: The Native APIs are experimental and subject to change.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "nvcomp/native/bitcomp.h"
#include "util.h"

// Error-check macro for Bitcomp Native API calls (they return bitcompResult_t).
#define BTCHK(func)                                                                                                    \
  do                                                                                                                   \
  {                                                                                                                    \
    bitcompResult_t rt = (func);                                                                                       \
    if (rt != BITCOMP_SUCCESS)                                                                                         \
    {                                                                                                                  \
      std::cerr << "Bitcomp call failure \"" #func "\" with code " << rt << " at " << __FILE__ << ":" << __LINE__      \
                << std::endl;                                                                                          \
      std::exit(1);                                                                                                    \
    }                                                                                                                  \
  } while (0)

int main()
{
  // The quantization delta sets the accuracy/ratio trade-off. The maximum
  // reconstruction error is guaranteed to be <= delta / 2. Bitcomp rounds
  // delta down to the nearest power of two, so 1.0f is used as is.
  const float delta = 1.0f;

  // Generate some smooth floating-point data with fractional values so the
  // lossy quantization is observable (and bounded).
  const size_t num_values = 1u << 20; // 1M floats
  const size_t in_bytes = num_values * sizeof(float);

  std::vector<float> h_input(num_values);
  for (size_t i = 0; i < num_values; ++i)
  {
    h_input[i] = 100.0f * std::sin(static_cast<float>(i) * 0.001f);
  }

  // Allocate device buffers. cudaMalloc satisfies Bitcomp's 8-byte alignment requirements implicitly.
  float *d_input = nullptr;
  void *d_comp = nullptr;
  float *d_output = nullptr;
  const size_t max_comp_bytes = bitcompMaxBuflen(in_bytes);

  CUDA_CHECK(cudaMallocSafe(&d_input, in_bytes));
  CUDA_CHECK(cudaMallocSafe(&d_comp, max_comp_bytes));
  CUDA_CHECK(cudaMallocSafe(&d_output, in_bytes));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaMemcpyAsync(d_input, h_input.data(), in_bytes, cudaMemcpyHostToDevice, stream));

  // 1. Create a plan for lossy FP32 -> signed compression with the default
  //    algorithm. The same plan can be used for compression and decompression.
  bitcompHandle_t plan;
  BTCHK(bitcompCreatePlan(&plan, in_bytes, BITCOMP_FP32_DATA, BITCOMP_LOSSY_FP_TO_SIGNED, BITCOMP_DEFAULT_ALGO));
  BTCHK(bitcompSetStream(plan, stream));

  // 2. Compress on the GPU.
  BTCHK(bitcompCompressLossy_fp32(plan, d_input, d_comp, delta));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Query the resulting compressed size.
  size_t comp_bytes = 0;
  BTCHK(bitcompGetCompressedSize(d_comp, &comp_bytes));

  // 3. Decompress with the same plan.
  BTCHK(bitcompUncompress(plan, d_comp, d_output));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // 4. Copy back and verify the error bound.
  std::vector<float> h_output(num_values);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, in_bytes, cudaMemcpyDeviceToHost));

  float max_error = 0.0f;
  for (size_t i = 0; i < num_values; ++i)
  {
    max_error = std::max(max_error, std::fabs(h_output[i] - h_input[i]));
  }

  std::cout << "Bitcomp Native API lossy FP32 compression\n"
            << "  uncompressed bytes : " << in_bytes << "\n"
            << "  compressed bytes   : " << comp_bytes << "\n"
            << "  compression ratio  : " << static_cast<double>(in_bytes) / static_cast<double>(comp_bytes) << "\n"
            << "  delta              : " << delta << "\n"
            << "  max error (bound)  : " << max_error << " (<= " << delta / 2.0f << ")\n";

  const bool within_bound = max_error <= delta / 2.0f;
  std::cout << (within_bound ? "SUCCESS: error within delta/2 bound." : "FAILURE: error exceeds delta/2 bound.")
            << std::endl;

  BTCHK(bitcompDestroyPlan(plan));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_comp));
  CUDA_CHECK(cudaFree(d_output));

  return within_bound ? 0 : 1;
}
