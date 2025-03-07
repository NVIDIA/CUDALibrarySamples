/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#include "benchmark_template_chunked.cuh"
#include "nvcomp/zstd.h"

static nvcompBatchedZstdOpts_t nvcompBatchedZstdTestOpts{};

static bool handleCommandLineArgument(
    const std::string& arg,
    const char* const* additionalArgs,
    size_t& additionalArgsUsed)
{
  // Zstd has no options.
  return false;
}

static bool isZstdInputValid(const std::vector<std::vector<char>>& data,
                             bool compressed_inputs)
{
  (void)compressed_inputs;
  for (const auto& chunk : data) {
    if (chunk.size() > nvcompZstdCompressionMaxAllowedChunkSize) {
      std::cerr << "ERROR: Zstd doesn't support chunk sizes larger than "
                << nvcompZstdCompressionMaxAllowedChunkSize << " bytes."
                << std::endl;
      return false;
    }
  }

  return true;
}

void run_benchmark(
    const std::vector<std::vector<char>>& data,
    const bool warmup,
    const size_t count,
    const bool csv_output,
    const bool tab_separator,
    const size_t duplicate_count,
    const size_t num_files,
    const bool compressed_inputs,
    const bool single_output_buffer,
    const std::string& output_compressed_filename,
    const std::string& output_decompressed_filename)
{
  run_benchmark_template(
      nvcompBatchedZstdCompressGetTempSize,
      nvcompBatchedZstdCompressGetMaxOutputChunkSize,
      nvcompBatchedZstdCompressAsync,
      nvcompBatchedZstdCompressGetRequiredAlignments,
      nvcompBatchedZstdDecompressGetTempSize,
      nvcompBatchedZstdDecompressAsync,
      nvcompBatchedZstdGetDecompressSizeAsync,
      nvcompBatchedZstdDecompressRequiredAlignments,
      isZstdInputValid,
      nvcompBatchedZstdTestOpts,
      data,
      warmup,
      count,
      csv_output,
      tab_separator,
      duplicate_count,
      num_files,
      compressed_inputs,
      single_output_buffer,
      output_compressed_filename,
      output_decompressed_filename);
}
