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
#include "nvcomp/bitcomp.h"

NVBENCH_REGISTER_CRITERION(total_time_criterion);

static nvcompBatchedBitcompCompressOpts_t nvcompBatchedBitcompCompressOpts = nvcompBatchedBitcompCompressDefaultOpts;
static nvcompBatchedBitcompDecompressOpts_t nvcompBatchedBitcompDecompressOpts =
  nvcompBatchedBitcompDecompressDefaultOpts;

const std::vector<parameter_type> custom_params = {
  {"t",
   "type",
   "Bitcomp data type. Must be one of (u)char, (u)short, (u)int, or (u)longlong.",
   "uchar",
   [](const char *arg) -> bool {
     bool valid;
     nvcompBatchedBitcompCompressOpts.data_type = string_to_data_type(arg, valid);
     return valid;
   }},
  {"alg", "algorithm", "Bitcomp algorithm to use (0 or 1).", "0", [](const char *arg) -> bool {
     int algorithm_type = atoi(arg);
     if (algorithm_type < 0 || algorithm_type > 1)
     {
       std::cerr << "ERROR: Bitcomp algorithm must be 0 or 1, but it is " << algorithm_type << std::endl;
       return false;
     }
     nvcompBatchedBitcompCompressOpts.algorithm = algorithm_type;
     return true;
   }}
};

static bool isBitcompInputValid(
  const std::vector<std::vector<char>> &data,
  bool compressed_inputs,
  [[maybe_unused]] const nvcompBatchedBitcompCompressOpts_t compress_opts,
  [[maybe_unused]] const nvcompBatchedBitcompDecompressOpts_t decompress_opts
)
{

  // Find the type size, to check that all chunk sizes are a multiple of it.
  size_t typeSize = 1;
  auto type = nvcompBatchedBitcompCompressOpts.data_type;
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
      std::cerr << "ERROR: Bitcomp data type must be 0-7 (CHAR, UCHAR, SHORT, "
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
      nvcompBatchedBitcompCompressGetTempSizeAsync,
      nvcompBatchedBitcompCompressGetTempSizeSync,
      nvcompBatchedBitcompCompressGetMaxOutputChunkSize,
      nvcompBatchedBitcompCompressAsync,
      nvcompBatchedBitcompCompressGetRequiredAlignments,
      isBitcompInputValid,
      nvcompBatchedBitcompCompressOpts,
      nvcompBatchedBitcompDecompressOpts,
      state,
      nvbench::exec_tag::sync
    );
  }
  else
  {
    run_decompression(
      nvcompBatchedBitcompDecompressGetTempSizeAsync,
      nvcompBatchedBitcompDecompressGetTempSizeSync,
      nvcompBatchedBitcompDecompressAsync,
      nvcompBatchedBitcompGetDecompressSizeAsync,
      nvcompBatchedBitcompDecompressGetRequiredAlignments,
      nvcompBatchedBitcompDecompressOpts,
      state,
      nvbench::exec_tag::sync
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
