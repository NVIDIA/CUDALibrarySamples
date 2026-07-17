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

#include <iostream>
#include <vector>

#include "benchmark_template_chunked.cuh"
#include "nvcomp/cascaded.h"

NVBENCH_REGISTER_CRITERION(total_time_criterion);

static nvcompBatchedCascadedCompressOpts_t nvcompBatchedCascadedCompressOpts = {4096, NVCOMP_TYPE_UINT, 2, 1, 1, {0}};
static nvcompBatchedCascadedDecompressOpts_t nvcompBatchedCascadedDecompressOpts =
  nvcompBatchedCascadedDecompressDefaultOpts;

const std::vector<parameter_type> custom_params = {
  {"t",
   "type",
   "Cascaded data type. Must be one of (u)char, (u)short, (u)int, or (u)longlong.",
   "uint",
   [](const char *arg) -> bool {
     bool valid;
     nvcompBatchedCascadedCompressOpts.type = string_to_data_type(arg, valid);
     return valid;
   }},
  {"r",
   "num_rles",
   "Number of RLE layers (max. 7).",
   "2",
   [](const char *arg) -> bool {
     int n = atoi(arg);
     if (n < 0 || n > 7)
     {
       std::cerr << "ERROR: num_rles must be between 0 and 7, but it is " << n << std::endl;
       return false;
     }
     nvcompBatchedCascadedCompressOpts.num_RLEs = n;
     return true;
   }},
  {"nd",
   "num_deltas",
   "Number of delta layers (max. 3).",
   "1",
   [](const char *arg) -> bool {
     int n = atoi(arg);
     if (n < 0 || n > 3)
     {
       std::cerr << "ERROR: num_deltas must be between 0 and 3, but it is " << n << std::endl;
       return false;
     }
     nvcompBatchedCascadedCompressOpts.num_deltas = n;
     return true;
   }},
  {"bp", "use_bp", "Whether to use bit-packing (0 or 1).", "1", [](const char *arg) -> bool {
     int n = atoi(arg);
     if (n < 0 || n > 1)
     {
       std::cerr << "ERROR: use_bp can only be 0 or 1, but it is " << n << std::endl;
       return false;
     }
     nvcompBatchedCascadedCompressOpts.use_bp = n;
     return true;
   }}
};

static bool isCascadedInputValid(
  const std::vector<std::vector<char>> &data,
  bool compressed_inputs,
  [[maybe_unused]] const nvcompBatchedCascadedCompressOpts_t compress_opts,
  [[maybe_unused]] const nvcompBatchedCascadedDecompressOpts_t decompress_opts
)
{
  // Find the type size, to check that all chunk sizes are a multiple of it.
  size_t typeSize = 1;
  auto type = nvcompBatchedCascadedCompressOpts.type;
  switch (type)
  {
    case NVCOMP_TYPE_CHAR:
    case NVCOMP_TYPE_UCHAR:
      // Type size is 1 byte, so chunk sizes are always a multiple of it.
      return true;
    case NVCOMP_TYPE_SHORT:
    case NVCOMP_TYPE_USHORT:
      typeSize = sizeof(uint16_t);
      break;
    case NVCOMP_TYPE_INT:
    case NVCOMP_TYPE_UINT:
      typeSize = sizeof(uint32_t);
      break;
    case NVCOMP_TYPE_LONGLONG:
    case NVCOMP_TYPE_ULONGLONG:
      typeSize = sizeof(uint64_t);
      break;
    default:
      std::cerr << "ERROR: Cascaded data type must be 0-7 (CHAR, UCHAR, SHORT, "
                   "USHORT, INT, UINT, LONGLONG, or ULONGLONG), "
                   "but it is "
                << int(type) << std::endl;
      return false;
  }

  if (!compressed_inputs)
  {
    for (const auto &chunk : data)
    {
      if ((chunk.size() % typeSize) != 0)
      {
        std::cerr << "ERROR: Input data must have a length and chunk size that "
                     "are a multiple of "
                  << typeSize << ", the size of the specified data type." << std::endl;
        return false;
      }
    }
  }
  return true;
}

template <bool DO_COMPRESSION>
void run_benchmark(nvbench::state &state)
{
  if constexpr (DO_COMPRESSION)
  {
    run_compression(
      nvcompBatchedCascadedCompressGetTempSizeAsync,
      nvcompBatchedCascadedCompressGetTempSizeSync,
      nvcompBatchedCascadedCompressGetMaxOutputChunkSize,
      nvcompBatchedCascadedCompressAsync,
      nvcompBatchedCascadedCompressGetRequiredAlignments,
      isCascadedInputValid,
      nvcompBatchedCascadedCompressOpts,
      nvcompBatchedCascadedDecompressOpts,
      state
    );
  }
  else
  {
    run_decompression(
      nvcompBatchedCascadedDecompressGetTempSizeAsync,
      nvcompBatchedCascadedDecompressGetTempSizeSync,
      nvcompBatchedCascadedDecompressAsync,
      nvcompBatchedCascadedGetDecompressSizeAsync,
      nvcompBatchedCascadedDecompressGetRequiredAlignments,
      nvcompBatchedCascadedDecompressOpts,
      state
    );
  }
}

static void run_benchmark_compress(nvbench::state &state) { run_benchmark<true>(state); }
static void run_benchmark_decompress(nvbench::state &state) { run_benchmark<false>(state); }
NVBENCH_BENCH(run_benchmark_compress)
  .set_name("Cascaded Chunked Compression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
NVBENCH_BENCH(run_benchmark_decompress)
  .set_name("Cascaded Chunked Decompression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
