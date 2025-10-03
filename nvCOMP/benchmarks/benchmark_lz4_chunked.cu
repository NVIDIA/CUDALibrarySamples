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
#include "nvcomp/lz4.h"

static nvcompBatchedLZ4CompressOpts_t nvcompBatchedLZ4CompressOpts =
  nvcompBatchedLZ4CompressDefaultOpts;
static nvcompBatchedLZ4DecompressOpts_t nvcompBatchedLZ4DecompressOpts =
  nvcompBatchedLZ4DecompressDefaultOpts;

static bool handleCommandLineArgument(
    const std::string& arg,
    const char* const* additionalArgs,
    size_t& additionalArgsUsed)
{
  if (arg == "--type" || arg == "-t") {
    const char* const typeArg = *additionalArgs;
    additionalArgsUsed = 1;
    bool valid;
    nvcompBatchedLZ4CompressOpts.data_type = string_to_data_type(typeArg, valid);
    return valid;
  }
  return false;
}

static bool isLZ4InputValid(const std::vector<std::vector<char>>& data,
                            bool compressed_inputs)
{
  // Find the type size, to check that all chunk sizes are a multiple of it.
  size_t typeSize = 1;
  auto type = nvcompBatchedLZ4CompressOpts.data_type;
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

  if (!compressed_inputs) {
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
      nvcompBatchedLZ4CompressGetTempSizeAsync,
      nvcompBatchedLZ4CompressGetMaxOutputChunkSize,
      nvcompBatchedLZ4CompressAsync,
      nvcompBatchedLZ4CompressGetRequiredAlignments,
      nvcompBatchedLZ4DecompressGetTempSizeAsync,
      nvcompBatchedLZ4DecompressGetTempSizeSync,
      nvcompBatchedLZ4DecompressAsync,
      nvcompBatchedLZ4GetDecompressSizeAsync,
      nvcompBatchedLZ4DecompressGetRequiredAlignments,
      isLZ4InputValid,
      nvcompBatchedLZ4CompressOpts,
      nvcompBatchedLZ4DecompressOpts,
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