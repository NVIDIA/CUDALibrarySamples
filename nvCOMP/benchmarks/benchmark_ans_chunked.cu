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
#include "nvcomp/ans.h"

NVBENCH_REGISTER_CRITERION(total_time_criterion);

static nvcompBatchedANSCompressOpts_t nvcompBatchedANSCompressOpts = nvcompBatchedANSCompressDefaultOpts;
static nvcompBatchedANSDecompressOpts_t nvcompBatchedANSDecompressOpts = nvcompBatchedANSDecompressDefaultOpts;

const std::vector<parameter_type> custom_params = {
  {"t",
   "type",
   "ANS data type to use. Must be one of (u)char, float16, or float8_e4m3.",
   "char",
   [](const char *arg) -> bool {
     bool valid;
     auto type = string_to_data_type(arg, valid);
     if (!valid)
     {
       return false;
     }
     switch (type)
     {
       case NVCOMP_TYPE_CHAR:
       case NVCOMP_TYPE_UCHAR:
       case NVCOMP_TYPE_FLOAT16:
       case NVCOMP_TYPE_FLOAT8_E4M3:
         nvcompBatchedANSCompressOpts.data_type = type;
         // Set the decompress data_type so it uses the type-specialized kernel.
         nvcompBatchedANSDecompressOpts.data_type = type;
         return true;
       default:
         std::cerr << "ERROR: ANS data type must be (u)char, float16, or float8_e4m3, "
                      "but it is "
                   << arg << std::endl;
         return false;
     }
   }},
  {"s",
   "max-sub-chunk-count",
   "Maximum sub-chunk count. Must be a power-of-2 in [4, 64] or zero (default).",
   "0",
   [](const char *arg) -> bool {
     int count = std::atoi(arg);
     if (count == 0)
     {
       return true;
     }
     if (count < 4 || count > 64 || (count & (count - 1)) != 0)
     {
       std::cerr << "ERROR: --max-sub-chunk-count must be a power-of-2 in [4, 64] or zero (default), got " << count
                 << std::endl;
       return false;
     }
     nvcompBatchedANSCompressOpts.max_sub_chunk_count = static_cast<uint8_t>(count);
     nvcompBatchedANSDecompressOpts.max_sub_chunk_count = static_cast<uint8_t>(count);
     return true;
   }}
};

static bool isANSInputValid(
  const std::vector<std::vector<char>> &data,
  bool compressed_inputs,
  [[maybe_unused]] const nvcompBatchedANSCompressOpts_t compress_opts,
  [[maybe_unused]] const nvcompBatchedANSDecompressOpts_t decompress_opts
)
{
  for (const auto &chunk : data)
  {
    if (chunk.size() > nvcompANSCompressionMaxAllowedChunkSize)
    {
      std::cerr << "ERROR: ANS doesn't support chunk sizes larger than "
                   "2^32-1 bytes."
                << std::endl;
      return false;
    }

    if (nvcompBatchedANSCompressOpts.data_type == NVCOMP_TYPE_FLOAT16 && chunk.size() % 2 != 0)
    {
      std::cerr << "Error: chunk size must be a multiple of 2 when using "
                   "ANS on float16 data."
                << std::endl;
      return false;
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
      nvcompBatchedANSCompressGetTempSizeAsync,
      nvcompBatchedANSCompressGetTempSizeSync,
      nvcompBatchedANSCompressGetMaxOutputChunkSize,
      nvcompBatchedANSCompressAsync,
      nvcompBatchedANSCompressGetRequiredAlignments,
      isANSInputValid,
      nvcompBatchedANSCompressOpts,
      nvcompBatchedANSDecompressOpts,
      state
    );
  }
  else
  {
    run_decompression(
      nvcompBatchedANSDecompressGetTempSizeAsync,
      nvcompBatchedANSDecompressGetTempSizeSync,
      nvcompBatchedANSDecompressAsync,
      nvcompBatchedANSGetDecompressSizeAsync,
      nvcompBatchedANSDecompressGetRequiredAlignments,
      nvcompBatchedANSDecompressOpts,
      state
    );
  }
}

static void run_benchmark_compress(nvbench::state &state) { run_benchmark<true>(state); }
static void run_benchmark_decompress(nvbench::state &state) { run_benchmark<false>(state); }
NVBENCH_BENCH(run_benchmark_compress)
  .set_name("ANS Chunked Compression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
NVBENCH_BENCH(run_benchmark_decompress)
  .set_name("ANS Chunked Decompression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
