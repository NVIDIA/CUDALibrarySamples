/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
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
#include "nvcomp/ans.h"

static nvcompBatchedANSOpts_t nvcompBatchedANSOpts = {nvcomp_rANS, uint8};

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
    case NVCOMP_TYPE_UINT8:
      nvcompBatchedANSOpts.data_type = uint8;
      break;
    case NVCOMP_TYPE_FLOAT16:
      nvcompBatchedANSOpts.data_type = float16;
      break;
    default:
      std::cerr << "ERROR: ANS data type must be (uint8 or float16) "
	           "but it is "
                << typeArg << std::endl;
      valid = false;
    }

    return valid;
  }
  return false;
}

static bool isANSInputValid(const std::vector<std::vector<char>>& data)
{
  for (const auto& chunk : data) {
    if (chunk.size() > (1ULL << 32) - 1) {
      std::cerr << "ERROR: ANS doesn't support chunk sizes larger than "
                   "2^32-1 bytes."
                << std::endl;
      return false;
    }

    if(nvcompBatchedANSOpts.data_type == float16 && chunk.size() % 2 != 0){
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
    const bool tab_separator,
    const size_t duplicate_count,
    const size_t num_files,
    const bool compressed_inputs,
    const bool single_output_buffer)
{
  run_benchmark_template(
      nvcompBatchedANSCompressGetTempSize,
      nvcompBatchedANSCompressGetMaxOutputChunkSize,
      nvcompBatchedANSCompressAsync,
      nvcompBatchedANSDecompressGetTempSize,
      nvcompBatchedANSDecompressAsync,
      nvcompBatchedANSGetDecompressSizeAsync,
      isANSInputValid,
      nvcompBatchedANSOpts,
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
