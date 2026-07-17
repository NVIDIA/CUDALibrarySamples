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

#include "benchmark_template_chunked.cuh"
#include "nvcomp/lz4.h"

NVBENCH_REGISTER_CRITERION(total_time_criterion);

static nvcompBatchedLZ4CompressOpts_t nvcompBatchedLZ4CompressOpts = nvcompBatchedLZ4CompressDefaultOpts;
static nvcompBatchedLZ4DecompressOpts_t nvcompBatchedLZ4DecompressOpts = nvcompBatchedLZ4DecompressDefaultOpts;

const std::vector<parameter_type> custom_params = {
  {"t",
   "type",
   "LZ4 data type. Must be one of (u)char, (u)short, (u)int, or bits.",
   "char",
   [](const char *arg) -> bool {
     bool valid;
     nvcompBatchedLZ4CompressOpts.data_type = string_to_data_type(arg, valid);
     nvcompBatchedLZ4DecompressOpts.data_type = nvcompBatchedLZ4CompressOpts.data_type;
     return valid;
   }},
  {"bs", "bitshuffle", "Bitshuffle mode (0, 1, or 2).", "0", [](const char *arg) -> bool {
     int bitshuffle_mode = atoi(arg);
     if (bitshuffle_mode < 0 || bitshuffle_mode > 2)
     {
       std::cerr << "ERROR: Bitshuffle mode must be 0, 1, or 2, but it is " << bitshuffle_mode << std::endl;
       return false;
     }
     nvcompBatchedLZ4CompressOpts.bitshuffle_mode = static_cast<nvcompBitshuffleMode_t>(bitshuffle_mode);
     nvcompBatchedLZ4DecompressOpts.bitshuffle_mode = static_cast<nvcompBitshuffleMode_t>(bitshuffle_mode);
     return true;
   }}
};

static bool isLZ4InputValid(
  const std::vector<std::vector<char>> &data,
  bool compressed_inputs,
  [[maybe_unused]] const nvcompBatchedLZ4CompressOpts_t compress_opts,
  [[maybe_unused]] const nvcompBatchedLZ4DecompressOpts_t decompress_opts
)
{
  // Find the type size, to check that all chunk sizes are a multiple of it.
  size_t typeSize = 1;
  auto type = nvcompBatchedLZ4CompressOpts.data_type;
  switch (type)
  {
    case NVCOMP_TYPE_BITS:
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
    default:
      std::cerr << "ERROR: LZ4 data type must be 0-5 or 255 (CHAR, UCHAR, SHORT, "
                   "USHORT, INT, UINT, or BITS), "
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
      nvcompBatchedLZ4CompressGetTempSizeAsync,
      nvcompBatchedLZ4CompressGetTempSizeSync,
      nvcompBatchedLZ4CompressGetMaxOutputChunkSize,
      nvcompBatchedLZ4CompressAsync,
      nvcompBatchedLZ4CompressGetRequiredAlignments,
      isLZ4InputValid,
      nvcompBatchedLZ4CompressOpts,
      nvcompBatchedLZ4DecompressOpts,
      state
    );
  }
  else
  {
    run_decompression(
      nvcompBatchedLZ4DecompressGetTempSizeAsync,
      nvcompBatchedLZ4DecompressGetTempSizeSync,
      nvcompBatchedLZ4DecompressAsync,
      nvcompBatchedLZ4GetDecompressSizeAsync,
      nvcompBatchedLZ4DecompressGetRequiredAlignments,
      nvcompBatchedLZ4DecompressOpts,
      state
    );
  }
}

static void run_benchmark_compress(nvbench::state &state) { run_benchmark<true>(state); }
static void run_benchmark_decompress(nvbench::state &state) { run_benchmark<false>(state); }
NVBENCH_BENCH(run_benchmark_compress)
  .set_name("LZ4 Chunked Compression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
NVBENCH_BENCH(run_benchmark_decompress)
  .set_name("LZ4 Chunked Decompression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
