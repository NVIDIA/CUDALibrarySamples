/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "nvcomp/cascaded.h"

#include <iostream>
#include <vector>

static nvcompBatchedCascadedCompressOpts_t nvcompBatchedCascadedCompressOpts =
  {4096, NVCOMP_TYPE_UINT, 2, 1, 1, {0}};
static nvcompBatchedCascadedDecompressOpts_t nvcompBatchedCascadedDecompressOpts =
  nvcompBatchedCascadedDecompressDefaultOpts;

static bool handleCommandLineArgument(
    const std::string& arg,
    const char* const* additionalArgs,
    size_t& additionalArgsUsed)
{
  if (arg == "--type" || arg == "-t") {
    const char* const typeArg = *additionalArgs;
    additionalArgsUsed = 1;
    bool valid;
    nvcompBatchedCascadedCompressOpts.type = string_to_data_type(typeArg, valid);
    return valid;
  }
  if (arg == "--num_rles" || arg == "-r") {
    int n = atoi(*additionalArgs);
    additionalArgsUsed = 1;
    if (n < 0) {
      std::cerr << "ERROR: num_rles can't be negative, but it is " << n
                << std::endl;
      return false;
    }
    nvcompBatchedCascadedCompressOpts.num_RLEs = n;
    return true;
  }
  if (arg == "--num_deltas" || arg == "-d") {
    int n = atoi(*additionalArgs);
    additionalArgsUsed = 1;
    if (n < 0) {
      std::cerr << "ERROR: num_deltas can't be negative, but it is " << n
                << std::endl;
      return false;
    }
    nvcompBatchedCascadedCompressOpts.num_deltas = n;
    return true;
  }
  if (arg == "--num_bps" || arg == "-b") {
    int n = atoi(*additionalArgs);
    additionalArgsUsed = 1;
    if (n < 0 || n > 1) {
      std::cerr << "ERROR: num_bps can only be 0 or 1, but it is " << n
                << std::endl;
      return false;
    }
    nvcompBatchedCascadedCompressOpts.use_bp = n;
    return true;
  }
  return false;
}

static bool isCascadedInputValid(const std::vector<std::vector<char>>& data,
                                 bool compressed_inputs)
{
  // Find the type size, to check that all chunk sizes are a multiple of it.
  size_t typeSize = 1;
  auto type = nvcompBatchedCascadedCompressOpts.type;
  switch (type) {
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
  case NVCOMP_TYPE_LONGLONG:
  case NVCOMP_TYPE_ULONGLONG:
    typeSize = sizeof(uint64_t);
    break;
  default:
    std::cerr << "ERROR: Cascaded data type must be 0-7 (CHAR, UCHAR, SHORT, "
                 "USHORT, INT, UINT, LONGLONG, or ULONGLONG), "
                 "but it is "
              << int(type) << std::endl;
    return false;
  }

  if(!compressed_inputs) {
    for (const auto& chunk : data) {
      if ((chunk.size() % typeSize) != 0) {
        std::cerr << "ERROR: Input data must have a length and chunk size that "
                    "are a multiple of "
                  << typeSize << ", the size of the specified data type."
                  << std::endl;
        return false;
      }
    }
  }
  return true;
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
      nvcompBatchedCascadedCompressGetTempSizeAsync,
      nvcompBatchedCascadedCompressGetMaxOutputChunkSize,
      nvcompBatchedCascadedCompressAsync,
      nvcompBatchedCascadedCompressGetRequiredAlignments,
      nvcompBatchedCascadedDecompressGetTempSizeAsync,
      nvcompBatchedCascadedDecompressGetTempSizeSync,
      nvcompBatchedCascadedDecompressAsync,
      nvcompBatchedCascadedGetDecompressSizeAsync,
      nvcompBatchedCascadedDecompressGetRequiredAlignments,
      isCascadedInputValid,
      nvcompBatchedCascadedCompressOpts,
      nvcompBatchedCascadedDecompressOpts,
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
