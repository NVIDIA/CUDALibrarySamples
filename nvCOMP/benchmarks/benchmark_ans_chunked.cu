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
#include "nvcomp.hpp"

NVBENCH_REGISTER_CRITERION(total_time_criterion);

constexpr uint8_t ANS_BENCH_AUTO_STATES_PER_LANE = 2;

static nvcompBatchedANSCompressOpts_t nvcompBatchedANSCompressOpts = nvcompBatchedANSCompressDefaultOpts;
// Set states_per_lane to the default value of 2 in order to launch the specialized kernel if the user
// only sets the data type.
static nvcompBatchedANSDecompressOpts_t nvcompBatchedANSDecompressOpts =
  {NVCOMP_DECOMPRESS_BACKEND_DEFAULT, NVCOMP_TYPE_BITS, 0, ANS_BENCH_AUTO_STATES_PER_LANE, 0, {0}};

const std::vector<parameter_type> custom_params = {
  {"t",
   "type",
   "ANS data type to use. Must be one of (u)char, float16, float32, or float8_e4m3.",
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
       case NVCOMP_TYPE_FLOAT32:
       case NVCOMP_TYPE_FLOAT8_E4M3:
         nvcompBatchedANSCompressOpts.data_type = type;
         // Set the decompress data_type so it uses the type-specialized kernel.
         nvcompBatchedANSDecompressOpts.data_type = type;
         return true;
       default:
         std::cerr << "ERROR: ANS data type must be (u)char, float16, float32, or float8_e4m3, "
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
   }},
  {"n",
   "states-per-lane",
   "Interleaved rANS states per lane: 0 (auto, default), 1 (single stream), or 2 (two streams).",
   "0",
   [](const char *arg) -> bool {
     const int states = std::atoi(arg);
     if (states != 0 && states != 1 && states != 2)
     {
       std::cerr << "ERROR: --states-per-lane must be 0 (auto), 1, or 2, got " << states << std::endl;
       return false;
     }
     nvcompBatchedANSCompressOpts.states_per_lane = static_cast<uint8_t>(states);
     nvcompBatchedANSDecompressOpts.states_per_lane = (states != 0) ? static_cast<uint8_t>(states)
                                                                    : ANS_BENCH_AUTO_STATES_PER_LANE;
     return true;
   }},
  {"S",
   "histogram-reduction-log2",
   "Right-shift of each warp's histogram slice: 0 (exact, default) through " +
     std::to_string(nvcompANSMaxHistogramReductionLog2) + " (keep 1/" +
     std::to_string(1u << nvcompANSMaxHistogramReductionLog2) + ").",
   "0",
   [](const char *arg) -> bool {
     const int shift = std::atoi(arg);
     if (shift < 0 || shift > static_cast<int>(nvcompANSMaxHistogramReductionLog2))
     {
       std::cerr << "ERROR: --histogram-reduction-log2 must be 0 through "
                 << static_cast<int>(nvcompANSMaxHistogramReductionLog2) << ", got " << shift << std::endl;
       return false;
     }
     nvcompBatchedANSCompressOpts.histogram_reduction_log2 = static_cast<uint8_t>(shift);
     return true;
   }},
  {"k",
   "skip-validate",
   "Skip per-chunk header/request checks during decompress: 0 (default) or 1 (trusted input).",
   "0",
   [](const char *arg) -> bool {
     const int skip = std::atoi(arg);
     if (skip != 0 && skip != 1)
     {
       std::cerr << "ERROR: --skip-validate must be 0 or 1, got " << skip << std::endl;
       return false;
     }
     nvcompBatchedANSDecompressOpts.skip_validate = static_cast<uint8_t>(skip);
     return true;
   }}
};

static bool isANSInputValid(
  const std::vector<std::vector<char>> &data,
  bool compressed_inputs,
  const nvcompBatchedANSCompressOpts_t compress_opts,
  [[maybe_unused]] const nvcompBatchedANSDecompressOpts_t decompress_opts
)
{
  const size_t max_chunk_size = compressed_inputs ? nvcompANSDecompressionMaxAllowedChunkSize
                                                  : nvcompANSCompressionMaxAllowedChunkSize;
  for (const auto &chunk : data)
  {
    if (chunk.size() > max_chunk_size)
    {
      std::cerr << "ERROR: ANS doesn't support " << (compressed_inputs ? "compressed" : "uncompressed")
                << " chunk sizes larger than " << max_chunk_size << " bytes." << std::endl;
      return false;
    }

    if (!compressed_inputs && chunk.size() % nvcomp::sizeOfnvcompType(compress_opts.data_type) != 0)
    {
      std::cerr << "ERROR: chunk size must be a multiple of the selected ANS data type size." << std::endl;
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
      nvcompBatchedANSCompressGetTempSize,
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
      nvcompBatchedANSDecompressGetTempSize,
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
