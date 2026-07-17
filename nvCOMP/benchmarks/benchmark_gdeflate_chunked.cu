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

#include <limits>

#include "benchmark_template_chunked.cuh"
#include "nvcomp/gdeflate.h"

#include <stdint.h>

NVBENCH_REGISTER_CRITERION(total_time_criterion);

static nvcompBatchedGdeflateCompressOpts_t nvcompBatchedGdeflateCompressOpts = nvcompBatchedGdeflateCompressDefaultOpts;
static nvcompBatchedGdeflateDecompressOpts_t nvcompBatchedGdeflateDecompressOpts =
  nvcompBatchedGdeflateDecompressDefaultOpts;

const std::vector<parameter_type> custom_params = {
  {"alg", "algorithm", "Gdeflate algorithm to use (0-5).", "1", [](const char *arg) -> bool {
     int algorithm_type = atoi(arg);
     if (algorithm_type < 0 || algorithm_type > 5)
     {
       std::cerr << "ERROR: Gdeflate algorithm must be 0, 1, 2, 3, 4 or 5, but it is " << algorithm_type << std::endl;
       return false;
     }
     nvcompBatchedGdeflateCompressOpts.algorithm = algorithm_type;
     return true;
   }}
};

static bool isGdeflateInputValid(
  const std::vector<std::vector<char>> &data,
  bool compressed_inputs,
  [[maybe_unused]] const nvcompBatchedGdeflateCompressOpts_t compress_opts,
  [[maybe_unused]] const nvcompBatchedGdeflateDecompressOpts_t decompress_opts
)
{
  (void)compressed_inputs;
  for (const auto &chunk : data)
  {
    if (chunk.size() > nvcompGdeflateCompressionMaxAllowedChunkSize)
    {
      std::cerr << "ERROR: Gdeflate doesn't support chunk sizes larger than "
                   "2GB."
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
      nvcompBatchedGdeflateCompressGetTempSizeAsync,
      nvcompBatchedGdeflateCompressGetTempSizeSync,
      nvcompBatchedGdeflateCompressGetMaxOutputChunkSize,
      nvcompBatchedGdeflateCompressAsync,
      nvcompBatchedGdeflateCompressGetRequiredAlignments,
      isGdeflateInputValid,
      nvcompBatchedGdeflateCompressOpts,
      nvcompBatchedGdeflateDecompressOpts,
      state,
      nvbench::exec_tag::no_batch | nvbench::exec_tag::sync
    );
  }
  else
  {
    run_decompression(
      nvcompBatchedGdeflateDecompressGetTempSizeAsync,
      nvcompBatchedGdeflateDecompressGetTempSizeSync,
      nvcompBatchedGdeflateDecompressAsync,
      nvcompBatchedGdeflateGetDecompressSizeAsync,
      nvcompBatchedGdeflateDecompressGetRequiredAlignments,
      nvcompBatchedGdeflateDecompressOpts,
      state
    );
  }
}

static void run_benchmark_compress(nvbench::state &state) { run_benchmark<true>(state); }
static void run_benchmark_decompress(nvbench::state &state) { run_benchmark<false>(state); }
NVBENCH_BENCH(run_benchmark_compress)
  .set_name("Gdeflate Chunked Compression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
NVBENCH_BENCH(run_benchmark_decompress)
  .set_name("Gdeflate Chunked Decompression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
