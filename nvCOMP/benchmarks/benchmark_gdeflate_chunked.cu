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
#include "nvcomp/gdeflate.h"

static nvcompBatchedGdeflateOpts_t nvcompBatchedGdeflateOpts = {1};

static bool handleCommandLineArgument(
    const std::string& arg,
    const char* const* additionalArgs,
    size_t& additionalArgsUsed)
{
  if (arg == "--algorithm" || arg == "-a") {
    int algorithm_type = atoi(*additionalArgs);
    additionalArgsUsed = 1;
    if (algorithm_type < 0 || algorithm_type > 5) {
      std::cerr << "ERROR: Gdeflate algorithm must be 0, 1, 2, 3, 4 or 5, but it is "
                << algorithm_type << std::endl;
      return false;
    }
    nvcompBatchedGdeflateOpts.algo = algorithm_type;
    return true;
  }
  return false;
}

static bool isGdeflateInputValid(const std::vector<std::vector<char>>& data)
{
  for (const auto& chunk : data) {
    if (chunk.size() > 65536) {
      std::cerr << "ERROR: Gdeflate doesn't support chunk sizes larger than "
                   "65536 bytes."
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
      nvcompBatchedGdeflateCompressGetTempSize,
      nvcompBatchedGdeflateCompressGetMaxOutputChunkSize,
      nvcompBatchedGdeflateCompressAsync,
      nvcompBatchedGdeflateDecompressGetTempSize,
      nvcompBatchedGdeflateDecompressAsync,
      nvcompBatchedGdeflateGetDecompressSizeAsync,
      isGdeflateInputValid,
      nvcompBatchedGdeflateOpts,
      data,
      warmup,
      count,
      csv_output,
      tab_separator,
      duplicate_count,
      num_files,
      compressed_inputs,
      single_output_buffer);
}
