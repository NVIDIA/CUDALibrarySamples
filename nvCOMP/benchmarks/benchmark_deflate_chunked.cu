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
#include "nvcomp/deflate.h"

#include <stdint.h>

NVBENCH_REGISTER_CRITERION(total_time_criterion);

static nvcompBatchedDeflateCompressOpts_t nvcompBatchedDeflateCompressOpts = nvcompBatchedDeflateCompressDefaultOpts;
static nvcompBatchedDeflateDecompressOpts_t nvcompBatchedDeflateDecompressOpts =
  nvcompBatchedDeflateDecompressDefaultOpts;

const std::vector<parameter_type> custom_params = {
  {"alg", "algorithm", "Deflate algorithm to use (0-5).", "1", [](const char *arg) -> bool {
     int algorithm_type = atoi(arg);
     if (algorithm_type < 0 || algorithm_type > 5)
     {
       std::cerr << "ERROR: Deflate algorithm must be 0, 1, 2, 3, 4, or 5, but it is " << algorithm_type << std::endl;
       return false;
     }
     nvcompBatchedDeflateCompressOpts.algorithm = algorithm_type;
     return true;
   }}
};

static bool isDeflateInputValid(
  const std::vector<std::vector<char>> &data,
  bool compressed_inputs,
  [[maybe_unused]] const nvcompBatchedDeflateCompressOpts_t compress_opts,
  [[maybe_unused]] const nvcompBatchedDeflateDecompressOpts_t decompress_opts
)
{
  (void)compressed_inputs;
  for (const auto &chunk : data)
  {
    if (chunk.size() > nvcompDeflateCompressionMaxAllowedChunkSize)
    {
      std::cerr << "ERROR: Deflate doesn't support chunk sizes larger than "
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
      nvcompBatchedDeflateCompressGetTempSizeAsync,
      nvcompBatchedDeflateCompressGetTempSizeSync,
      nvcompBatchedDeflateCompressGetMaxOutputChunkSize,
      nvcompBatchedDeflateCompressAsync,
      nvcompBatchedDeflateCompressGetRequiredAlignments,
      isDeflateInputValid,
      nvcompBatchedDeflateCompressOpts,
      nvcompBatchedDeflateDecompressOpts,
      state,
      nvbench::exec_tag::no_batch | nvbench::exec_tag::sync
    );
  }
  else
  {
    run_decompression(
      nvcompBatchedDeflateDecompressGetTempSizeAsync,
      nvcompBatchedDeflateDecompressGetTempSizeSync,
      nvcompBatchedDeflateDecompressAsync,
      nvcompBatchedDeflateGetDecompressSizeAsync,
      nvcompBatchedDeflateDecompressGetRequiredAlignments,
      nvcompBatchedDeflateDecompressOpts,
      state
    );
  }
}

static void run_benchmark_compress(nvbench::state &state) { run_benchmark<true>(state); }
static void run_benchmark_decompress(nvbench::state &state) { run_benchmark<false>(state); }
NVBENCH_BENCH(run_benchmark_compress)
  .set_name("Deflate Chunked Compression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
NVBENCH_BENCH(run_benchmark_decompress)
  .set_name("Deflate Chunked Decompression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
