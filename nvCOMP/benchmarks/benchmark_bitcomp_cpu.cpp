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
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "nvcomp/native/bitcomp.h"
#include "benchmark_common.h"

using nvcomp::multi_file;
using nvcomp::verifyLossyCompression;
using nvcomp::benchmark_assert;

int main(int argc, char** argv) {
  std::string filename;
  size_t warmup_count = 1;
  size_t iteration_count = 1;
  size_t duplicate_count = 1;
  bool csv_output = false;
  bitcompDataType_t bitcomp_type = BITCOMP_UNSIGNED_8BIT;
  int algorithm = 0;
  bool use_lossy = false;
  double lossy_delta = 1.0;
  int lossy_fp_bits = 16;

  // Simple argument parsing
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-f" && i + 1 < argc) {
      filename = argv[++i];
    } else if (arg == "-w" && i + 1 < argc) {
      warmup_count = std::stoull(argv[++i]);
    } else if (arg == "-i" && i + 1 < argc) {
      iteration_count = std::stoull(argv[++i]);
    } else if (arg == "-x" && i + 1 < argc) {
      duplicate_count = std::stoull(argv[++i]);
    } else if (arg == "-c" && i + 1 < argc) {
      csv_output = (std::string(argv[++i]) == "true");
    } else if (arg == "-t" && i + 1 < argc) {
      std::string type = argv[++i];
      if (type == "uchar") bitcomp_type = BITCOMP_UNSIGNED_8BIT;
      else if (type == "char") bitcomp_type = BITCOMP_SIGNED_8BIT;
      else if (type == "ushort") bitcomp_type = BITCOMP_UNSIGNED_16BIT;
      else if (type == "short") bitcomp_type = BITCOMP_SIGNED_16BIT;
      else if (type == "uint") bitcomp_type = BITCOMP_UNSIGNED_32BIT;
      else if (type == "int") bitcomp_type = BITCOMP_SIGNED_32BIT;
      else if (type == "ulonglong") bitcomp_type = BITCOMP_UNSIGNED_64BIT;
      else if (type == "longlong") bitcomp_type = BITCOMP_SIGNED_64BIT;
    } else if (arg == "-a" && i + 1 < argc) {
      algorithm = std::stoi(argv[++i]);
      if (algorithm < 0 || algorithm > 1) {
        std::cerr << "ERROR: Invalid algorithm (must be 0 or 1)" << std::endl;
        return 1;
      }
    } else if (arg == "--lossy") {
      use_lossy = true;
    } else if (arg == "--fp" && i + 1 < argc) {
      lossy_fp_bits = std::stoi(argv[++i]);
    } else if (arg == "--delta" && i + 1 < argc) {
      lossy_delta = std::stod(argv[++i]);
    }
  }

  if (filename.empty()) {
    std::cerr << "Usage: " << argv[0] << " -f <filename> [options]" << std::endl;
    return 1;
  }

  // Set compression mode and override bitcomp_type for lossy compression
  bitcompMode_t bitcomp_mode;

  if (use_lossy) {
    bitcomp_mode = BITCOMP_LOSSY_FP_TO_SIGNED;
    if (lossy_fp_bits == 16) bitcomp_type = BITCOMP_FP16_DATA;
    else if (lossy_fp_bits == 32) bitcomp_type = BITCOMP_FP32_DATA;
    else if (lossy_fp_bits == 64) bitcomp_type = BITCOMP_FP64_DATA;
    else {
      std::cerr << "ERROR: Invalid fp bits for lossy: " << lossy_fp_bits << std::endl;
      return 1;
    }
  } else {
    bitcomp_mode = BITCOMP_LOSSLESS;
    // bitcomp_type is already set from command line parsing
  }

  bitcompAlgorithm_t bitcomp_algo = (algorithm == 0) ? BITCOMP_DEFAULT_ALGO : BITCOMP_SPARSE_ALGO;

  // Read input file as single chunk
  std::vector<std::vector<char>> inputs;
  inputs = multi_file({filename}, false, 0, 1, duplicate_count);

  if (inputs.empty() || inputs[0].empty()) {
    std::cerr << "Failed to read input file or file is empty" << std::endl;
    return 1;
  }

  std::cout << "Processing entire file as single chunk (size: " << inputs[0].size() << " bytes)" << std::endl;

  // Single file processing - no batch
  const void* h_input_ptr = inputs[0].data();
  size_t input_size = inputs[0].size();

  // Allocate buffers
  size_t max_comp_size = bitcompMaxBuflen(input_size);
  std::vector<char> h_compressed_data(max_comp_size);
  char* h_compressed_ptr = h_compressed_data.data();

  std::vector<char> h_decompressed_data(input_size);
  char* h_decompressed_ptr = h_decompressed_data.data();

  // Create plan
  bitcompHandle_t plan;
  BTCHK(bitcompCreatePlan(&plan, input_size, bitcomp_type, bitcomp_mode, bitcomp_algo));

  // Lambda for compression operations (used in both warmup and benchmark)
  size_t compressed_size = 0;
  auto compress_file = [&](bool get_size = false) {
    if (use_lossy) {
      // Lossy compression
      if (lossy_fp_bits == 16) {
        BTCHK(bitcompHostCompressLossy_fp16(plan, reinterpret_cast<const half*>(h_input_ptr), h_compressed_ptr, static_cast<half>(lossy_delta)));
      } else if (lossy_fp_bits == 32) {
        BTCHK(bitcompHostCompressLossy_fp32(plan, reinterpret_cast<const float*>(h_input_ptr), h_compressed_ptr, static_cast<float>(lossy_delta)));
      } else if (lossy_fp_bits == 64) {
        BTCHK(bitcompHostCompressLossy_fp64(plan, reinterpret_cast<const double*>(h_input_ptr), h_compressed_ptr, lossy_delta));
      }
    } else {
      BTCHK(bitcompHostCompressLossless(plan, h_input_ptr, h_compressed_ptr));
    }

    if (get_size) {
      BTCHK(bitcompGetCompressedSize(h_compressed_ptr, &compressed_size));
    }
  };

  // Warmup
  for (size_t w = 0; w < warmup_count; ++w) {
    compress_file(true);  // Use same logic as benchmark
    BTCHK(bitcompHostUncompress(plan, h_compressed_ptr, h_decompressed_ptr));
  }

  // Benchmark compression
  auto start_time = std::chrono::high_resolution_clock::now();

  for (size_t iter = 0; iter < iteration_count; ++iter) {
    compress_file(true);  // Get compressed size during benchmark
  }

  auto compress_end_time = std::chrono::high_resolution_clock::now();

  // Benchmark decompression
  for (size_t iter = 0; iter < iteration_count; ++iter) {
    BTCHK(bitcompHostUncompress(plan, h_compressed_ptr, h_decompressed_ptr));
  }

  auto decompress_end_time = std::chrono::high_resolution_clock::now();

  // Verify data (only on last iteration)
  if (use_lossy) {
    try {
      verifyLossyCompression(h_input_ptr, h_decompressed_ptr, input_size, lossy_delta, bitcomp_type);
    } catch (const std::exception& e) {
      std::cerr << "Lossy verification failed: " << e.what() << std::endl;
      return 1;
    }
  } else {
    // For lossless, do exact byte comparison
    try {
      benchmark_assert(std::memcmp(h_input_ptr, h_decompressed_ptr, input_size) == 0,
                       "The decompressed data did not match the input");
    } catch (const std::exception& e) {
      std::cerr << "Verification failed: " << e.what() << std::endl;
      return 1;
    }
  }

  // Calculate metrics
  size_t total_input_bytes = input_size;
  size_t total_compressed_bytes = compressed_size;

  auto compress_duration = std::chrono::duration_cast<std::chrono::microseconds>(compress_end_time - start_time);
  auto decompress_duration = std::chrono::duration_cast<std::chrono::microseconds>(decompress_end_time - compress_end_time);

  double comp_time_s = compress_duration.count() * 1e-6 / iteration_count;
  double decomp_time_s = decompress_duration.count() * 1e-6 / iteration_count;

  double compression_ratio = (double)total_input_bytes / total_compressed_bytes;
  double compression_throughput = total_input_bytes / (1e9 * comp_time_s);
  double decompression_throughput = total_input_bytes / (1e9 * decomp_time_s);

  if (!csv_output) {
    std::cout << "----------" << std::endl;
    std::cout << "files: 1" << std::endl;
    std::cout << "uncompressed (B): " << total_input_bytes << std::endl;
    std::cout << "comp_size: " << total_compressed_bytes
              << ", compressed ratio: " << std::fixed << std::setprecision(4) << compression_ratio << std::endl;
    std::cout << "compression throughput (GB/s): " << compression_throughput << std::endl;
    std::cout << "decompression throughput (GB/s): " << decompression_throughput << std::endl;
  } else {
    // Header
    std::cout << "Files" << ","
              << "Duplicate data" << ","
              << "Size in MiB" << ","
              << "Chunks" << ","
              << "Avg chunk size in KiB" << ","
              << "Max chunk size in KiB" << ","
              << "Uncompressed size in bytes" << ","
              << "Compressed size in bytes" << ","
              << "Compression ratio" << ","
              << "Compression throughput (uncompressed) in GB/s" << ","
              << "Decompression throughput (uncompressed) in GB/s" << std::endl;

    // Values
    std::cout << "1,"
              << duplicate_count << ","
              << (total_input_bytes / (1024 * 1024)) << ","
              << "1,"
              << (total_input_bytes / 1024) << ","
              << (total_input_bytes / 1024) << ","
              << total_input_bytes << ","
              << total_compressed_bytes << ","
              << std::fixed << std::setprecision(2) << compression_ratio << ","
              << compression_throughput << ","
              << decompression_throughput << std::endl;
  }

  // Cleanup
  BTCHK(bitcompDestroyPlan(plan));

  return 0;
}
