/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "benchmark_template_chunked.cuh"
#include "nvcomp.h"

NVBENCH_REGISTER_CRITERION(total_time_criterion);

static nvcompBatchedBitcompCompressOpts_t nvcompBatchedBitcompCompressOpts = nvcompBatchedBitcompCompressDefaultOpts;
static nvcompBatchedBitcompDecompressOpts_t nvcompBatchedBitcompDecompressOpts =
  nvcompBatchedBitcompDecompressDefaultOpts;
static bool bitcomp_delta_was_specified = false;

const std::vector<parameter_type> custom_params = {
  {"t",
   "type",
   "Bitcomp data type. Must be an integral type, float16, float32, or float64.",
   "uchar",
   [](const char *arg) -> bool {
     bool valid;
     nvcompBatchedBitcompCompressOpts.data_type = string_to_data_type(arg, valid);
     return valid;
   }},
  {"alg",
   "algorithm",
   "Bitcomp algorithm to use (0 or 1).",
   "0",
   [](const char *arg) -> bool {
     int algorithm_type = atoi(arg);
     if (algorithm_type < 0 || algorithm_type > 1)
     {
       std::cerr << "ERROR: Bitcomp algorithm must be 0 or 1, but it is " << algorithm_type << std::endl;
       return false;
     }
     nvcompBatchedBitcompCompressOpts.algorithm = algorithm_type;
     return true;
   }},
  {"",
   "mode",
   "Bitcomp mode: lossless, lossy-signed, or lossy-unsigned. Lossy modes require --delta.",
   "lossless",
   [](const char *arg) -> bool {
     if (strcmp(arg, "lossless") == 0)
     {
       nvcompBatchedBitcompCompressOpts.mode = BITCOMP_LOSSLESS;
       nvcompBatchedBitcompCompressOpts.delta = 0.0;
       return true;
     }
     if (strcmp(arg, "lossy-signed") == 0)
     {
       nvcompBatchedBitcompCompressOpts.mode = BITCOMP_LOSSY_FP_TO_SIGNED;
       return true;
     }
     if (strcmp(arg, "lossy-unsigned") == 0)
     {
       nvcompBatchedBitcompCompressOpts.mode = BITCOMP_LOSSY_FP_TO_UNSIGNED;
       return true;
     }
     std::cerr << "ERROR: Bitcomp mode must be lossless, lossy-signed, or lossy-unsigned." << std::endl;
     return false;
   }},
  {"", "delta", "Lossy compression delta (required for lossy modes).", "", [](const char *arg) -> bool {
     nvcompBatchedBitcompCompressOpts.delta = atof(arg);
     bitcomp_delta_was_specified = true;
     return true;
   }}
};

static bool isBitcompInputValid(
  const std::vector<std::vector<char>> &data,
  bool compressed_inputs,
  const nvcompBatchedBitcompCompressOpts_t compress_opts,
  [[maybe_unused]] const nvcompBatchedBitcompDecompressOpts_t decompress_opts
)
{
  if (compressed_inputs)
  {
    return true;
  }

  const auto type = compress_opts.data_type;
  const bool integral_type = type == NVCOMP_TYPE_BITS || (type >= NVCOMP_TYPE_CHAR && type <= NVCOMP_TYPE_ULONGLONG);
  const bool floating_point_type = type == NVCOMP_TYPE_FLOAT16 || type == NVCOMP_TYPE_FLOAT32 ||
                                   type == NVCOMP_TYPE_FLOAT64;
  // Lossy modes require a floating-point type; lossless additionally accepts integral types.
  const bool valid_mode_and_type = floating_point_type || (compress_opts.mode == BITCOMP_LOSSLESS && integral_type);
  if (!valid_mode_and_type)
  {
    std::cerr << "ERROR: Lossy Bitcomp requires a floating-point type." << std::endl;
    return false;
  }

  const size_t type_size = nvcomp::sizeOfnvcompType(type);
  for (const auto &chunk : data)
  {
    if (!nvcomp::ensure_length_type_aligned(chunk.size(), type_size))
    {
      return false;
    }
  }
  return true;
}

template <bool DO_COMPRESSION>
void run_benchmark(nvbench::state &state)
{
  benchmark_assert(
    nvcompBatchedBitcompCompressOpts.mode == BITCOMP_LOSSLESS || bitcomp_delta_was_specified,
    "Lossy Bitcomp mode requires --delta."
  );

  if constexpr (DO_COMPRESSION)
  {
    run_compression(
      nvcompBatchedBitcompCompressGetTempSize,
      nvcompBatchedBitcompCompressGetTempSizeSync,
      nvcompBatchedBitcompCompressGetMaxOutputChunkSize,
      nvcompBatchedBitcompCompressAsync,
      nvcompBatchedBitcompCompressGetRequiredAlignments,
      isBitcompInputValid,
      nvcompBatchedBitcompCompressOpts,
      nvcompBatchedBitcompDecompressOpts,
      state
    );
  }
  else
  {
    run_decompression(
      nvcompBatchedBitcompDecompressGetTempSize,
      nvcompBatchedBitcompDecompressGetTempSizeSync,
      nvcompBatchedBitcompDecompressAsync,
      nvcompBatchedBitcompGetDecompressSizeAsync,
      nvcompBatchedBitcompDecompressGetRequiredAlignments,
      nvcompBatchedBitcompDecompressOpts,
      state,
      nvbench::exec_tag::sync,
      nvcompBatchedBitcompCompressOpts.mode == BITCOMP_LOSSLESS ? 0.0 : nvcompBatchedBitcompCompressOpts.delta,
      nvcompBatchedBitcompCompressOpts.data_type
    );
  }
}

static void run_benchmark_compress(nvbench::state &state) { run_benchmark<true>(state); }
static void run_benchmark_decompress(nvbench::state &state) { run_benchmark<false>(state); }
NVBENCH_BENCH(run_benchmark_compress)
  .set_name("Bitcomp Chunked Compression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
NVBENCH_BENCH(run_benchmark_decompress)
  .set_name("Bitcomp Chunked Decompression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
