/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "nvcomp/snappy.h"

static bool handleCommandLineArgument(
    const std::string& arg,
    const char* const* additionalArgs,
    size_t& additionalArgsUsed)
{
  // Snappy has no options.
  return false;
}

void run_benchmark(
    const std::vector<std::vector<char>>& data,
    const bool warmup,
    const size_t count,
    const bool csv_output,
    const nvcompDecompressBackend_t decompress_backend,
    const bool tab_separator,
    const size_t duplicate_count,
    const size_t num_files,
    const bool compressed_inputs,
    const bool single_output_buffer,
    const std::string& output_compressed_filename,
    const std::string& output_decompressed_filename)
{
  run_benchmark_template(
      nvcompBatchedSnappyCompressGetTempSizeAsync,
      nvcompBatchedSnappyCompressGetMaxOutputChunkSize,
      nvcompBatchedSnappyCompressAsync,
      nvcompBatchedSnappyCompressGetRequiredAlignments,
      nvcompBatchedSnappyDecompressGetTempSizeAsync,
      nvcompBatchedSnappyDecompressGetTempSizeSync,
      nvcompBatchedSnappyDecompressAsync,
      nvcompBatchedSnappyGetDecompressSizeAsync,
      nvcompBatchedSnappyDecompressGetRequiredAlignments,
      inputAlwaysValid,
      nvcompBatchedSnappyCompressDefaultOpts,
      nvcompBatchedSnappyDecompressDefaultOpts,
      data,
      warmup,
      count,
      csv_output,
      decompress_backend,
      tab_separator,
      duplicate_count,
      num_files,
      compressed_inputs,
      single_output_buffer,
      output_compressed_filename,
      output_decompressed_filename);
}