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
#include "nvcomp/lz4.h"

static nvcompBatchedLZ4Opts_t nvcompBatchedLZ4TestOpts{NVCOMP_TYPE_CHAR};

static bool handleCommandLineArgument(
    const std::string& arg,
    const char* const* additionalArgs,
    size_t& additionalArgsUsed)
{
  if (arg == "--type" || arg == "-t") {
    const char* const typeArg = *additionalArgs;
    additionalArgsUsed = 1;
    bool valid;
    nvcompBatchedLZ4TestOpts.data_type = string_to_data_type(typeArg, valid);
    return valid;
  }
  return false;
}

static bool isLZ4InputValid(const std::vector<std::vector<char>>& data)
{
  // Find the type size, to check that all chunk sizes are a multiple of it.
  size_t typeSize = 1;
  auto type = nvcompBatchedLZ4TestOpts.data_type;
  switch (type) {
  case NVCOMP_TYPE_BITS:
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
  default:
    std::cerr << "ERROR: LZ4 data type must be 0-5 or 255 (CHAR, UCHAR, SHORT, "
                 "USHORT, INT, UINT, or BITS), "
                 "but it is "
              << int(type) << std::endl;
    return false;
  }

  for (const auto& chunk : data) {
    if ((chunk.size() % typeSize) != 0) {
      std::cerr << "ERROR: Input data must have a length and chunk size that "
                   "are a multiple of "
                << typeSize << ", the size of the specified data type."
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
      nvcompBatchedLZ4CompressGetTempSize,
      nvcompBatchedLZ4CompressGetMaxOutputChunkSize,
      nvcompBatchedLZ4CompressAsync,
      nvcompBatchedLZ4DecompressGetTempSize,
      nvcompBatchedLZ4DecompressAsync,
      nvcompBatchedLZ4GetDecompressSizeAsync,
      isLZ4InputValid,
      nvcompBatchedLZ4TestOpts,
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
