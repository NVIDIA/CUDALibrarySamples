/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include "nvcomp/bitcomp.h"
#include "nvcomp/native/bitcomp.h"
#include "benchmark_template_chunked.cuh"
#include "benchmark_common.h"


static nvcompBatchedBitcompCompressOpts_t nvcompBatchedBitcompCompressOpts =
    nvcompBatchedBitcompCompressDefaultOpts;
static nvcompBatchedBitcompDecompressOpts_t nvcompBatchedBitcompDecompressOpts =
    nvcompBatchedBitcompDecompressDefaultOpts;
static double nvcomp_lossy_delta = 1.0;
static int nvcomp_lossy_fp_bits = 16;

static nvcompStatus_t bitcompBatchLossyCompressAsync(
    const void* const* in_ptrs,
    const size_t* in_sizes,
    size_t /* max_in_size */,
    size_t num_chunks,
    void* /* temp_ptr */,
    size_t /* temp_bytes */,
    void* const* out_ptrs,
    size_t* out_sizes,
    nvcompBatchedBitcompCompressOpts_t compression_opts,
    nvcompStatus_t* statuses,
    cudaStream_t stream)
{
  const auto algo = static_cast<bitcompAlgorithm_t>(compression_opts.algorithm);

  bitcompHandle_t plan;
  bitcompMode_t mode = BITCOMP_LOSSY_FP_TO_SIGNED;
  bitcompDataType_t lossy_dtype = getLossyDataType(nvcomp_lossy_fp_bits);
  BTCHK(bitcompCreateBatchPlan(&plan, num_chunks, lossy_dtype, mode, algo));

  BTCHK(bitcompSetStream(plan, stream));

  switch(lossy_dtype) {
    case BITCOMP_FP16_DATA:
      BTCHK(bitcompBatchCompressLossyScalar_fp16(
            plan,
            reinterpret_cast<const half* const*>(in_ptrs),
            out_ptrs,
            in_sizes,
            out_sizes,
            static_cast<float>(nvcomp_lossy_delta)));
      break;
    case BITCOMP_FP32_DATA:
      BTCHK(bitcompBatchCompressLossyScalar_fp32(
            plan,
            reinterpret_cast<const float* const*>(in_ptrs),
            out_ptrs,
            in_sizes,
            out_sizes,
            static_cast<float>(nvcomp_lossy_delta)));
      break;
    case BITCOMP_FP64_DATA:
      BTCHK(bitcompBatchCompressLossyScalar_fp64(
            plan,
            reinterpret_cast<const double* const*>(in_ptrs),
            out_ptrs,
            in_sizes,
            out_sizes,
            nvcomp_lossy_delta));
      break;
    default:
      // Should never happen since we check for this in getLossyDataType, but just in case, handle it gracefully
      throw nvcomp::NVCompException(nvcompErrorInvalidValue, "Unsupported data type for lossy bitcomp compression");
  }

  if(statuses) {
    // Note:
    // Bitcomp doesn't have an argument in its native API to handle
    // nvCOMP statuses.
    CUDA_CHECK(cudaMemsetAsync(statuses, 0, sizeof(nvcompStatus_t) * num_chunks, stream));
  }

  BTCHK(bitcompDestroyPlan(plan));
  return nvcompSuccess;
}

static nvcompStatus_t bitcompGetCompressAlignments(
    nvcompBatchedBitcompCompressOpts_t /* opts */,
    nvcompAlignmentRequirements_t* reqs)
{
  reqs->input = reqs->output = reqs->temp = 8;
  return nvcompSuccess;
}

static nvcompStatus_t bitcompGetMaxOutputChunkSize(
    size_t maxUncompressedChunkBytes,
    nvcompBatchedBitcompCompressOpts_t /* opts */,
    size_t* maxCompressedChunkBytes)
{
  *maxCompressedChunkBytes = bitcompMaxBuflen(maxUncompressedChunkBytes);
  return nvcompSuccess;
}

static nvcompStatus_t bitcompGetTempSizeAsync(
    size_t /* numChunks */,
    size_t /* maxUncompressedChunkBytes */,
    nvcompBatchedBitcompCompressOpts_t /* opts */,
    size_t* tempBytes,
    size_t /* maxTotalUncompressedBytes */)
{
  // For lossy compression, no temporary memory needed
  *tempBytes = 0;
  return nvcompSuccess;
}

static bool handleCommandLineArgument(
    const std::string& arg,
    const char* const* additionalArgs,
    size_t& additionalArgsUsed)
{
  if (arg == "--algorithm" || arg == "-a") {
    int algorithm_type = atoi(*additionalArgs);
    additionalArgsUsed = 1;
    if (algorithm_type < 0 || algorithm_type > 1) {
      std::cerr << "ERROR: Bitcomp algorithm must be 0 or 1, but it is "
                << algorithm_type << std::endl;
      return false;
    }
    nvcompBatchedBitcompCompressOpts.algorithm = algorithm_type;
    return true;
  }

  if(arg == "--fp") {
    const char* fp = *additionalArgs;
    additionalArgsUsed = 1;
    int bits = atoi(fp);
    if(bits == 16 || bits == 32 || bits == 64) {
      nvcomp_lossy_fp_bits = bits;
    } else {
      std::cerr << "ERROR: --fp must be 16, 32 or 64" << std::endl;
      return false;
    }
    return true;
  }

  if(arg == "--delta") {
    const char* d = *additionalArgs;
    additionalArgsUsed = 1;
    nvcomp_lossy_delta = atof(d);
    return true;
  }
    return false;
}

static bool isBitcompInputValid(const std::vector<std::vector<char>>& data,
                                bool compressed_inputs)
{
  // Convert floating point bits to byte size (16->2, 32->4, 64->8)
  size_t typeSize = nvcomp_lossy_fp_bits / 8;

  if(!compressed_inputs) {
    for (const auto& chunk : data) {
      if ((chunk.size() % typeSize) != 0) {
        std::cerr << "ERROR: Input data must have a length and chunk size that "
                    "are a multiple of "
                  << typeSize << ", the size of the specified data type." \
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
      bitcompGetTempSizeAsync,
      bitcompGetMaxOutputChunkSize,
      bitcompBatchLossyCompressAsync,
      bitcompGetCompressAlignments,
      nvcompBatchedBitcompDecompressGetTempSizeAsync,
      nvcompBatchedBitcompDecompressGetTempSizeSync,
      nvcompBatchedBitcompDecompressAsync,
      nvcompBatchedBitcompGetDecompressSizeAsync,
      nvcompBatchedBitcompDecompressGetRequiredAlignments,
      isBitcompInputValid,
      nvcompBatchedBitcompCompressOpts,
      nvcompBatchedBitcompDecompressOpts,
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
      output_decompressed_filename,
      nvcomp_lossy_delta,
      nvcomp_lossy_fp_bits);
}
