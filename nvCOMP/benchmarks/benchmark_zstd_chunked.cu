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
#include "nvcomp/zstd.h"

NVBENCH_REGISTER_CRITERION(total_time_criterion);

const std::vector<parameter_type> custom_params;

static bool isZstdInputValid(
  const std::vector<std::vector<char>> &data,
  bool compressed_inputs,
  [[maybe_unused]] const nvcompBatchedZstdCompressOpts_t compress_opts,
  [[maybe_unused]] const nvcompBatchedZstdDecompressOpts_t decompress_opts
)
{
  (void)compressed_inputs;
  for (const auto &chunk : data)
  {
    if (chunk.size() > nvcompZstdCompressionMaxAllowedChunkSize)
    {
      std::cerr << "ERROR: Zstd doesn't support chunk sizes larger than " << nvcompZstdCompressionMaxAllowedChunkSize
                << " bytes." << std::endl;
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
      nvcompBatchedZstdCompressGetTempSizeAsync,
      nvcompBatchedZstdCompressGetTempSizeSync,
      nvcompBatchedZstdCompressGetMaxOutputChunkSize,
      nvcompBatchedZstdCompressAsync,
      nvcompBatchedZstdCompressGetRequiredAlignments,
      isZstdInputValid,
      nvcompBatchedZstdCompressDefaultOpts,
      nvcompBatchedZstdDecompressDefaultOpts,
      state
    );
  }
  else
  {
    run_decompression(
      nvcompBatchedZstdDecompressGetTempSizeAsync,
      nvcompBatchedZstdDecompressGetTempSizeSync,
      nvcompBatchedZstdDecompressAsync,
      nvcompBatchedZstdGetDecompressSizeAsync,
      nvcompBatchedZstdDecompressGetRequiredAlignments,
      nvcompBatchedZstdDecompressDefaultOpts,
      state
    );
  }
}

static void run_benchmark_compress(nvbench::state &state) { run_benchmark<true>(state); }
static void run_benchmark_decompress(nvbench::state &state) { run_benchmark<false>(state); }
NVBENCH_BENCH(run_benchmark_compress)
  .set_name("Zstd Chunked Compression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
NVBENCH_BENCH(run_benchmark_decompress)
  .set_name("Zstd Chunked Decompression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
