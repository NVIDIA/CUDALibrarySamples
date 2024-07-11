/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES.
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
static std::string filename;
static bool do_output;

static bool handleCommandLineArgument(
    const std::string& arg,
    const char* const* additionalArgs,
    size_t& additionalArgsUsed)
{
  if (arg == "--output-file" || arg == "-o") {
    const char* const typeArg = *additionalArgs;
    additionalArgsUsed = 1;
    filename = typeArg;
    do_output = true;
    return true;
  }
  return false; // Any other parameters means that we took in an invalid argument
}

static bool isZstdInputValid(const std::vector<std::vector<char>>& data)
{
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
    const bool single_output_buffer)
{
  run_benchmark_template(
      nvcompBatchedZstdCompressGetTempSize,
      nvcompBatchedZstdCompressGetMaxOutputChunkSize,
      nvcompBatchedZstdCompressAsync,
      nvcompBatchedZstdDecompressGetTempSize,
      nvcompBatchedZstdDecompressAsync,
      nvcompBatchedZstdGetDecompressSizeAsync,
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
      do_output,
      filename);
}
