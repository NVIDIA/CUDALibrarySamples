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

#include <cuda_fp16.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "benchmark_common.h"
#include "benchmark_lossy_common.h"
#include "nvcomp/native/bitcomp.h"

#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif // _MSC_VER

#include "nvbench/nvbench.cuh"

#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif // _MSC_VER

#include "custom_criterion.hpp"

NVBENCH_REGISTER_CRITERION(total_time_criterion);

using nvcomp::benchmark_assert;
using nvcomp::getLossyDataType;
using nvcomp::multi_file;
using nvcomp::verifyLossyCompression;

// Uncompressed input, populated by main() from the requested file.
static std::vector<char> input;
// Compressed output, produced by the compression benchmark and reused by the
// decompression benchmark (NVBench runs the two in registration order).
static std::vector<char> compressed;
static bitcompDataType_t bitcomp_type = BITCOMP_UNSIGNED_8BIT;
static int bitcomp_algorithm = 0;
static bool use_lossy = false;
static int lossy_fp_bits = 16;
static double lossy_delta = 1.0;

static bool set_bitcomp_type(const std::string &name)
{
  static const std::unordered_map<std::string, bitcompDataType_t> types = {
    {"uchar", BITCOMP_UNSIGNED_8BIT},
    {"char", BITCOMP_SIGNED_8BIT},
    {"ushort", BITCOMP_UNSIGNED_16BIT},
    {"short", BITCOMP_SIGNED_16BIT},
    {"uint", BITCOMP_UNSIGNED_32BIT},
    {"int", BITCOMP_SIGNED_32BIT},
    {"ulonglong", BITCOMP_UNSIGNED_64BIT},
    {"longlong", BITCOMP_SIGNED_64BIT},
  };
  const auto it = types.find(name);
  if (it == types.end())
  {
    std::cerr << "ERROR: Invalid type '" << name << "'." << std::endl;
    return false;
  }
  bitcomp_type = it->second;
  return true;
}

// Benchmarks the native bitcomp host (CPU) API, which compresses the input as a
// single contiguous buffer.
template <bool DO_COMPRESSION>
static void run_benchmark(nvbench::state &state)
{
  if (input.empty())
  {
    state.skip("Given input file is empty.");
    return;
  }

  const bitcompMode_t mode = use_lossy ? BITCOMP_LOSSY_FP_TO_SIGNED : BITCOMP_LOSSLESS;
  const bitcompDataType_t type = use_lossy ? getLossyDataType(lossy_fp_bits) : bitcomp_type;
  const bitcompAlgorithm_t algo = (bitcomp_algorithm == 0) ? BITCOMP_DEFAULT_ALGO : BITCOMP_SPARSE_ALGO;

  bitcompHandle_t plan;
  BTCHK(bitcompCreatePlan(&plan, input.size(), type, mode, algo));

  auto compress = [&] {
    if (use_lossy)
    {
      switch (lossy_fp_bits)
      {
        case 16:
          BTCHK(bitcompHostCompressLossy_fp16(
            plan,
            reinterpret_cast<const half *>(input.data()),
            compressed.data(),
            static_cast<half>(lossy_delta)
          ));
          break;
        case 32:
          BTCHK(bitcompHostCompressLossy_fp32(
            plan,
            reinterpret_cast<const float *>(input.data()),
            compressed.data(),
            static_cast<float>(lossy_delta)
          ));
          break;
        case 64:
          BTCHK(bitcompHostCompressLossy_fp64(
            plan,
            reinterpret_cast<const double *>(input.data()),
            compressed.data(),
            lossy_delta
          ));
          break;
      }
    }
    else
    {
      BTCHK(bitcompHostCompressLossless(plan, input.data(), compressed.data()));
    }
  };

  state.add_element_count(input.size(), "Bytes");

  if constexpr (DO_COMPRESSION)
  {
    compressed.resize(bitcompMaxBuflen(input.size()));
    state.exec(nvbench::exec_tag::no_gpu, [&](nvbench::launch &) { compress(); });

    size_t compressed_size = 0;
    BTCHK(bitcompGetCompressedSize(compressed.data(), &compressed_size));
    auto &summary = state.add_summary("Compression Ratio");
    summary.set_string("name", "Compression Ratio");
    summary.set_float64("value", static_cast<double>(input.size()) / compressed_size);
  }
  else
  {
    // The compression benchmark (registered first) fills `compressed`. If this
    // benchmark is run on its own, produce the compressed data here first.
    if (compressed.empty())
    {
      compressed.resize(bitcompMaxBuflen(input.size()));
      compress();
    }

    std::vector<char> decompressed(input.size());
    BTCHK(bitcompHostUncompress(plan, compressed.data(), decompressed.data()));

    // Verify correctness once, outside the timed region.
    if (use_lossy)
    {
      const nvcompType_t nvcomp_type = lossy_fp_bits == 16   ? NVCOMP_TYPE_FLOAT16
                                       : lossy_fp_bits == 32 ? NVCOMP_TYPE_FLOAT32
                                                             : NVCOMP_TYPE_FLOAT64;
      verifyLossyCompression(input.data(), decompressed.data(), input.size(), lossy_delta, nvcomp_type);
    }
    else
    {
      benchmark_assert(
        std::memcmp(input.data(), decompressed.data(), input.size()) == 0,
        "The decompressed data did not match the input"
      );
    }

    state.exec(nvbench::exec_tag::no_gpu, [&](nvbench::launch &) {
      BTCHK(bitcompHostUncompress(plan, compressed.data(), decompressed.data()));
    });
  }

  BTCHK(bitcompDestroyPlan(plan));
}

static void run_benchmark_compress(nvbench::state &state) { run_benchmark<true>(state); }
static void run_benchmark_decompress(nvbench::state &state) { run_benchmark<false>(state); }
NVBENCH_BENCH(run_benchmark_compress)
  .set_name("Bitcomp CPU Compression")
  .set_is_cpu_only(true)
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
NVBENCH_BENCH(run_benchmark_decompress)
  .set_name("Bitcomp CPU Decompression")
  .set_is_cpu_only(true)
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);

int main(int argc, char **argv)
{
  std::string filename;
  size_t duplicate_count = 1;

  // Consume the format-specific options and forward everything else (NVBench's
  // own flags) to NVBENCH_MAIN_BODY.
  std::vector<char *> nvbench_argv{argv[0]};
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    auto value = [&]() -> const char * {
      if (i + 1 >= argc)
      {
        std::cerr << "ERROR: Missing value for '" << arg << "'." << std::endl;
        std::exit(1);
      }
      return argv[++i];
    };

    if (arg == "-f")
    {
      filename = value();
    }
    else if (arg == "-x")
    {
      duplicate_count = std::stoull(value());
    }
    else if (arg == "-t")
    {
      if (!set_bitcomp_type(value()))
      {
        return 1;
      }
    }
    else if (arg == "--alg")
    {
      bitcomp_algorithm = std::stoi(value());
      if (bitcomp_algorithm < 0 || bitcomp_algorithm > 1)
      {
        std::cerr << "ERROR: Bitcomp algorithm must be 0 or 1." << std::endl;
        return 1;
      }
    }
    else if (arg == "--lossy")
    {
      use_lossy = true;
    }
    else if (arg == "--fp")
    {
      lossy_fp_bits = std::stoi(value());
      if (lossy_fp_bits != 16 && lossy_fp_bits != 32 && lossy_fp_bits != 64)
      {
        std::cerr << "ERROR: --fp must be 16, 32 or 64." << std::endl;
        return 1;
      }
    }
    else if (arg == "--delta")
    {
      lossy_delta = std::stod(value());
    }
    else
    {
      nvbench_argv.push_back(argv[i]);
    }
  }

  if (filename.empty())
  {
    std::cerr << "Usage: " << argv[0] << " -f <filename> [-t type] [--alg 0|1] [-x dup] "
              << "[--lossy --fp 16|32|64 --delta d] [NVBench options]" << std::endl;
    return 1;
  }

  // Replicate the file content to reach the requested data size (1x = as-is).
  input = multi_file({filename}, false, 0, 1, duplicate_count)[0];

  int nvbench_argc = static_cast<int>(nvbench_argv.size());
  NVBENCH_MAIN_BODY(nvbench_argc, nvbench_argv.data());
}
