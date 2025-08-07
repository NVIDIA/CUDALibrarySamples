/*
 * Copyright (c) 2022-2025 NVIDIA CORPORATION AND AFFILIATES. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the NVIDIA CORPORATION nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
