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


// Benchmark performance from the binary data file fname
#include <vector>
#include <string.h>

#include "benchmark_common.h"
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"
#include "benchmark_hlif.hpp"

using namespace nvcomp;

void run_benchmark_from_file(char* fname, nvcompManagerBase& batch_manager, int verbose_memory, cudaStream_t stream, const int benchmark_exec_count)
{
  using T = uint8_t;

  size_t input_elts = 0;
  std::vector<T> data;
  data = load_dataset_from_binary<T>(fname, &input_elts);
  run_benchmark(data, batch_manager, verbose_memory, stream, benchmark_exec_count);
}

static void print_usage()
{
  printf("Usage: benchmark_hlif [format_type] [OPTIONS]\n");
  printf("  %-35s One of <snappy / bitcomp / ans / cascaded / gdeflate / deflate / lz4 / zstd> (required).\n", "[ format_type ]");
  printf("  %-35s Binary dataset filename (required).\n", "-f, --filename");
  printf("  %-35s Chunk size (default 64 kB).\n", "-c, --chunk-size");
  printf("  %-35s GPU device number (default 0)\n", "-g, --gpu");
  printf("  %-35s Number of times to execute the benchmark (for averaging) (default 1)\n", "-n, --num-iters");
  printf("  %-35s Data type (default 'char', options are 'char', 'short', 'int', 'longlong', 'float16')\n", "-t, --type");
  printf(
      "  %-35s Output GPU memory allocation sizes (default off)\n",
      "-m, --memory");
  exit(1);
}

int main(int argc, char* argv[])
{
  char* fname = NULL;
  int gpu_num = 0;
  int verbose_memory = 0;
  int num_iters = 1;

  // Cascaded compression options
  nvcompBatchedCascadedCompressOpts_t cascaded_compress_opts = nvcompBatchedCascadedCompressDefaultOpts;

  // Shared opts
  int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  std::string comp_format;

  bool explicit_type = false;
  bool explicit_chunk_size = false;

  // Parse command-line arguments
  char** argv_end = argv + argc;
  argv += 1;

  if(argc < 4) {
    print_usage();
    return 1;
  }

  // First the format
  comp_format = std::string{*argv++};
  if (comp_format == "lz4") {
  } else if (comp_format == "snappy") {
  } else if (comp_format == "bitcomp") {
  } else if (comp_format == "ans") {
  } else if (comp_format == "cascaded") {
  } else if (comp_format == "gdeflate") {
  } else if (comp_format == "deflate") {
  } else if (comp_format == "zstd") {
  } else {
    printf("invalid format\n");
    print_usage();
    return 1;
  }

  while (argv != argv_end) {
    char* arg = *argv++;
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-?") == 0) {
      print_usage();
      return 1;
    }
    if (strcmp(arg, "--memory") == 0 || strcmp(arg, "-m") == 0) {
      verbose_memory = 1;
      continue;
    }


    // all arguments below require at least a second value in argv
    if (argv >= argv_end) {
      print_usage();
      return 1;
    }

    char* optarg = *argv++;
    if (strcmp(arg, "--filename") == 0 || strcmp(arg, "-f") == 0) {
      fname = optarg;
      continue;
    }

    if (strcmp(arg, "--gpu") == 0 || strcmp(arg, "-g") == 0) {
      gpu_num = atoi(optarg);
      continue;
    }

    if (strcmp(arg, "--num-iters") == 0 || strcmp(arg, "-n") == 0) {
      num_iters = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--chunk-size") == 0 || strcmp(arg, "-c") == 0) {
      chunk_size = atoi(optarg);
      explicit_chunk_size = true;
      continue;
    }

    if (strcmp(arg, "--type") == 0 || strcmp(arg, "-t") == 0) {
      explicit_type = true;
      if (strcmp(optarg, "char") == 0) {
        data_type = NVCOMP_TYPE_CHAR;
      } else if (strcmp(optarg, "short") == 0) {
        data_type = NVCOMP_TYPE_SHORT;
      } else if (strcmp(optarg, "int") == 0) {
        data_type = NVCOMP_TYPE_INT;
      } else if (strcmp(optarg, "longlong") == 0) {
        data_type = NVCOMP_TYPE_LONGLONG;
      } else if (strcmp(optarg, "float16") == 0) {
        data_type = NVCOMP_TYPE_FLOAT16;
      } else {
        print_usage();
        return 1;
      }
      continue;
    }

    if (strcmp(arg, "--num_rles") == 0 || strcmp(arg, "-r") == 0) {
      cascaded_compress_opts.num_RLEs = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--num_deltas") == 0 || strcmp(arg, "-d") == 0) {
      cascaded_compress_opts.num_deltas = atoi(optarg);
      continue;
    }
    if (strcmp(arg, "--num_bps") == 0 || strcmp(arg, "-b") == 0) {
      cascaded_compress_opts.use_bp = (atoi(optarg) != 0);
      continue;
    }

    print_usage();
    return 1;
  }

  if (fname == NULL) {
    print_usage();
    return 1;
  }

  CUDA_CHECK(cudaSetDevice(gpu_num));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  {
    std::shared_ptr<nvcompManagerBase> manager;
    if (comp_format == "lz4") {
      manager = std::make_shared<LZ4Manager>(chunk_size, nvcompBatchedLZ4CompressOpts_t{data_type, {0}}, nvcompBatchedLZ4DecompressDefaultOpts, stream, NoComputeNoVerify);
    } else if (comp_format == "snappy") {
      manager = std::make_shared<SnappyManager>(chunk_size, nvcompBatchedSnappyCompressDefaultOpts, nvcompBatchedSnappyDecompressDefaultOpts, stream, NoComputeNoVerify);
    } else if (comp_format == "bitcomp") {
      manager = std::make_shared<BitcompManager>(chunk_size, nvcompBatchedBitcompCompressOpts_t{0 /* algo--fixed for now */, data_type, {0}}, nvcompBatchedBitcompDecompressDefaultOpts, stream, NoComputeNoVerify);
    } else if (comp_format == "ans") {
      manager = std::make_shared<ANSManager>(chunk_size, nvcompBatchedANSCompressOpts_t{nvcomp_rANS, data_type, {0}}, nvcompBatchedANSDecompressDefaultOpts, stream, NoComputeNoVerify);
    } else if (comp_format == "cascaded") {
      if (explicit_type) {
        cascaded_compress_opts.type = data_type;
      }

      if (explicit_chunk_size) {
        cascaded_compress_opts.internal_chunk_bytes = chunk_size;
      }
      manager = std::make_shared<CascadedManager>(chunk_size, cascaded_compress_opts, nvcompBatchedCascadedDecompressDefaultOpts, stream, NoComputeNoVerify);
    } else if (comp_format == "gdeflate") {
      manager = std::make_shared<GdeflateManager>(chunk_size, nvcompBatchedGdeflateCompressOpts_t{0 /* algo--fixed for now */, {0}}, nvcompBatchedGdeflateDecompressDefaultOpts, stream, NoComputeNoVerify);
    } else if (comp_format == "deflate") {
      manager = std::make_shared<DeflateManager>(chunk_size, nvcompBatchedDeflateCompressDefaultOpts, nvcompBatchedDeflateDecompressDefaultOpts, stream, NoComputeNoVerify);
    } else if (comp_format == "zstd") {
      // Get file size
      manager = std::make_shared<ZstdManager>(static_cast<size_t>(chunk_size), nvcompBatchedZstdCompressDefaultOpts, nvcompBatchedZstdDecompressDefaultOpts, stream, NoComputeNoVerify);
    } else {
      print_usage();
      return 1;
    }

    run_benchmark_from_file(fname, *manager, verbose_memory, stream, num_iters);
    // Scope destroys manager before stream is destroyed, as required.
  }

  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}