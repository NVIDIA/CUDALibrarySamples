/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "benchmark_template_chunked.cuh"
#include "cascaded_cli_options.hpp"
#include "nvcomp.h"

NVBENCH_REGISTER_CRITERION(total_time_criterion);

static nvcompBatchedCascadedCompressOpts_t nvcompBatchedCascadedCompressOpts = nvcompBatchedCascadedCompressDefaultOpts;
static nvcompBatchedCascadedDecompressOpts_t nvcompBatchedCascadedDecompressOpts = [] {
  auto opts = nvcompBatchedCascadedDecompressDefaultOpts;
  opts.common_opts = nvcompBatchedCascadedCompressDefaultOpts.common_opts;
  return opts;
}();

const std::vector<parameter_type> custom_params = {
  {"t",
   "type",
   "Cascaded data type. Must be one of (u)char, (u)short, (u)int, or (u)longlong.",
   "uint",
   [](const char *arg) -> bool {
     bool valid;
     nvcompBatchedCascadedCompressOpts.common_opts.data_type = string_to_data_type(arg, valid);
     nvcompBatchedCascadedDecompressOpts.common_opts.data_type =
       nvcompBatchedCascadedCompressOpts.common_opts.data_type;
     return valid;
   }},
  {"cm",
   "mode",
   "Compression mode: asymmetric or symmetric.",
   "asymmetric",
   [](const char *arg) -> bool {
     const bool valid = apply_cascaded_mode_arg(arg, nvcompBatchedCascadedCompressOpts);
     if (valid)
     {
       nvcompBatchedCascadedDecompressOpts.common_opts.mode = nvcompBatchedCascadedCompressOpts.common_opts.mode;
     }
     return valid;
   }},
  {"cl",
   "level",
   "Asymmetric compression level (0-255); currently ignored by symmetric mode.",
   "3",
   [](const char *arg) -> bool { return apply_cascaded_level_arg(arg, nvcompBatchedCascadedCompressOpts); }},
  {"cf",
   "fine_grained_encodings",
   "Fine-grained encodings: none or comma-separated delta,rle,for.",
   "delta,rle",
   [](const char *arg) -> bool {
     return apply_cascaded_fine_grained_encodings_arg(arg, nvcompBatchedCascadedCompressOpts);
   }}
};

static bool isCascadedInputValid(
  const std::vector<std::vector<char>> &data,
  bool compressed_inputs,
  [[maybe_unused]] const nvcompBatchedCascadedCompressOpts_t compress_opts,
  [[maybe_unused]] const nvcompBatchedCascadedDecompressOpts_t decompress_opts
)
{
  // Find the type size, to check that all chunk sizes are a multiple of it.
  size_t typeSize = 1;
  auto type = nvcompBatchedCascadedCompressOpts.common_opts.data_type;
  switch (type)
  {
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

  if (!compressed_inputs)
  {
    for (const auto &chunk : data)
    {
      if (!nvcomp::ensure_length_type_aligned(chunk.size(), typeSize))
      {
        return false;
      }
    }
  }
  return true;
}

static void run_benchmark_compress(nvbench::state &state)
{
  run_compression(
    nvcompBatchedCascadedCompressGetTempSize,
    nvcompBatchedCascadedCompressGetTempSizeSync,
    nvcompBatchedCascadedCompressGetMaxOutputChunkSize,
    nvcompBatchedCascadedCompressAsync,
    nvcompBatchedCascadedCompressGetRequiredAlignments,
    isCascadedInputValid,
    nvcompBatchedCascadedCompressOpts,
    nvcompBatchedCascadedDecompressOpts,
    state
  );
}

void run_benchmark_decompress(nvbench::state &state)
{
  run_decompression(
    nvcompBatchedCascadedDecompressGetTempSize,
    nvcompBatchedCascadedDecompressGetTempSizeSync,
    nvcompBatchedCascadedDecompressAsync,
    nvcompBatchedCascadedGetDecompressSizeAsync,
    nvcompBatchedCascadedDecompressGetRequiredAlignments,
    nvcompBatchedCascadedDecompressOpts,
    state
  );
}

NVBENCH_BENCH(run_benchmark_compress)
  .set_name("Cascaded Chunked Compression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
NVBENCH_BENCH(run_benchmark_decompress)
  .set_name("Cascaded Chunked Decompression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
