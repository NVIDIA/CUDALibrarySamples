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

#include "benchmark_template_chunked.cuh"
#include "nvcomp/gzip.h"

NVBENCH_REGISTER_CRITERION(total_time_criterion);

static nvcompBatchedGzipCompressOpts_t nvcompBatchedGzipCompressOpts = nvcompBatchedGzipCompressDefaultOpts;

static nvcompBatchedGzipDecompressOpts_t nvcompBatchedGzipDecompressOpts = nvcompBatchedGzipDecompressDefaultOpts;

const std::vector<parameter_type> custom_params = {
  {"alg",
   "algorithm",
   "Gzip compression algorithm to use (0-5).",
   "1",
   [](const char *arg) -> bool {
     int algorithm_type = atoi(arg);
     if (algorithm_type < 0 || algorithm_type > 5)
     {
       std::cerr << "ERROR: Gzip algorithm must be 0, 1, 2, 3, 4, or 5, but it is " << algorithm_type << std::endl;
       return false;
     }
     nvcompBatchedGzipCompressOpts.algorithm = algorithm_type;
     return true;
   }},
  {"da", "decompress-algorithm", "Gzip decompression algorithm: 0=naive, 1=lookahead.", "0", [](const char *arg) -> bool {
     int algorithm_type = atoi(arg);
     if (algorithm_type != 0 && algorithm_type != 1)
     {
       std::cerr << "ERROR: Gzip decompress algorithm must be 0 (naive) or 1 (lookahead), but it is " << algorithm_type
                 << std::endl;
       return false;
     }
     nvcompBatchedGzipDecompressOpts.algorithm = static_cast<nvcompBatchedGzipDecompressAlgorithm_t>(algorithm_type);
     return true;
   }}
};

static bool isGzipInputValid(
  const std::vector<std::vector<char>> &data,
  bool compressed_inputs,
  [[maybe_unused]] const nvcompBatchedGzipCompressOpts_t compress_opts,
  const nvcompBatchedGzipDecompressOpts_t decompress_opts
)
{
  // The naive and lookahead decompressors each have their own maximum chunk size.
  const size_t max_decompress_chunk_size = decompress_opts.algorithm == NVCOMP_GZIP_DECOMPRESS_ALGORITHM_NAIVE
                                             ? nvcompGzipNaiveDecompressionMaxAllowedChunkSize
                                             : nvcompGzipLookaheadDecompressionMaxAllowedChunkSize;

  const size_t max_chunk_size = compressed_inputs
                                  ? max_decompress_chunk_size
                                  : std::min(nvcompGzipCompressionMaxAllowedChunkSize, max_decompress_chunk_size);
  for (const auto &chunk : data)
  {
    if (chunk.size() > max_chunk_size)
    {
      std::cerr << "ERROR: Gzip doesn't support chunk sizes larger than " << max_chunk_size << " bytes." << std::endl;
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
      nvcompBatchedGzipCompressGetTempSizeAsync,
      nvcompBatchedGzipCompressGetTempSizeSync,
      nvcompBatchedGzipCompressGetMaxOutputChunkSize,
      nvcompBatchedGzipCompressAsync,
      nvcompBatchedGzipCompressGetRequiredAlignments,
      isGzipInputValid,
      nvcompBatchedGzipCompressOpts,
      nvcompBatchedGzipDecompressOpts,
      state,
      nvbench::exec_tag::no_batch | nvbench::exec_tag::sync
    );
  }
  else
  {
    run_decompression(
      nvcompBatchedGzipDecompressGetTempSizeAsync,
      nvcompBatchedGzipDecompressGetTempSizeSync,
      nvcompBatchedGzipDecompressAsync,
      nvcompBatchedGzipGetDecompressSizeAsync,
      nvcompBatchedGzipDecompressGetRequiredAlignments,
      nvcompBatchedGzipDecompressOpts,
      state
    );
  }
}

static void run_benchmark_compress(nvbench::state &state) { run_benchmark<true>(state); }
static void run_benchmark_decompress(nvbench::state &state) { run_benchmark<false>(state); }
NVBENCH_BENCH(run_benchmark_compress)
  .set_name("Gzip Chunked Compression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
NVBENCH_BENCH(run_benchmark_decompress)
  .set_name("Gzip Chunked Decompression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
