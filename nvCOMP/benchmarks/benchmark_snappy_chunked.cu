/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES.
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
    const bool tab_separator,
    const size_t duplicate_count,
    const size_t num_files,
    const bool compressed_inputs,
    const bool single_output_buffer)
{
  run_benchmark_template(
      nvcompBatchedSnappyCompressGetTempSize,
      nvcompBatchedSnappyCompressGetMaxOutputChunkSize,
      nvcompBatchedSnappyCompressAsync,
      nvcompBatchedSnappyDecompressGetTempSize,
      nvcompBatchedSnappyDecompressAsync,
      nvcompBatchedSnappyGetDecompressSizeAsync,
      inputAlwaysValid,
      nvcompBatchedSnappyDefaultOpts,
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
