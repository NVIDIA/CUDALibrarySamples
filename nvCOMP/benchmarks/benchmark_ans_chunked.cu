/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "nvcomp/ans.h"

static nvcompBatchedANSCompressOpts_t nvcompBatchedANSCompressOpts =
  nvcompBatchedANSCompressDefaultOpts;
static nvcompBatchedANSDecompressOpts_t nvcompBatchedANSDecompressOpts =
  nvcompBatchedANSDecompressDefaultOpts;

static bool handleCommandLineArgument(
    const std::string& arg,
    const char* const* additionalArgs,
    size_t& additionalArgsUsed)
{
  if (arg == "--type" || arg == "-t") {
    const char* const typeArg = *additionalArgs;
    additionalArgsUsed = 1;
    bool valid;

    auto type = string_to_data_type(typeArg, valid);
    switch(type){
    case NVCOMP_TYPE_CHAR:
    case NVCOMP_TYPE_UCHAR:
    case NVCOMP_TYPE_FLOAT16:
      nvcompBatchedANSCompressOpts.data_type = type;
      break;
    default:
      std::cerr << "ERROR: ANS data type must be (NVCOMP_TYPE_(U)CHAR or NVCOMP_TYPE_FLOAT16), "
	                 "but it is " << typeArg << std::endl;
      valid = false;
    }
    return valid;
  }
  return false;
}

static bool isANSInputValid(const std::vector<std::vector<char>>& data,
                            bool compressed_inputs)
{
  (void)compressed_inputs;
  for (const auto& chunk : data) {
    if (chunk.size() > (1ULL << 32) - 1) {
      std::cerr << "ERROR: ANS doesn't support chunk sizes larger than "
                   "2^32-1 bytes."
                << std::endl;
      return false;
    }

    if(nvcompBatchedANSCompressOpts.data_type == NVCOMP_TYPE_FLOAT16 && chunk.size() % 2 != 0){
      std::cerr << "Error: chunk size must be a multiple of 2 when using "
	           "ANS on float16 data."
                << std::endl;
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
      nvcompBatchedANSCompressGetTempSizeAsync,
      nvcompBatchedANSCompressGetMaxOutputChunkSize,
      nvcompBatchedANSCompressAsync,
      nvcompBatchedANSCompressGetRequiredAlignments,
      nvcompBatchedANSDecompressGetTempSizeAsync,
      nvcompBatchedANSDecompressGetTempSizeSync,
      nvcompBatchedANSDecompressAsync,
      nvcompBatchedANSGetDecompressSizeAsync,
      nvcompBatchedANSDecompressGetRequiredAlignments,
      isANSInputValid,
      nvcompBatchedANSCompressOpts,
      nvcompBatchedANSDecompressOpts,
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