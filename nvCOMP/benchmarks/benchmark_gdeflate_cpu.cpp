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
#include <numeric>

#include "nvcomp/native/gdeflate_cpu.h"
#include "benchmark_common.h"

using nvcomp::multi_file;
using nvcomp::benchmark_assert;

int main(int argc, char** argv) {
  std::string filename;
  size_t warmup_count = 1;
  size_t iteration_count = 1;
  size_t duplicate_count = 1;
  bool csv_output = false;
  int compression_level = 5;
  size_t chunk_size = 65536;

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
    } else if (arg == "-l" && i + 1 < argc) {
      compression_level = std::stoi(argv[++i]);
      if(compression_level < 0 || compression_level > 12) {
        std::cerr << "Gdeflate CPU compression level must be between 0 and 12 (both inclusive)";
        return 1;
      }
    } else if (arg == "-p" && i + 1 < argc) {
      chunk_size = std::stoull(argv[++i]);
      if(chunk_size > gdeflate::nvcompGdeflateCPUCompressionMaxAllowedChunkSize) {
        std::cerr << "Gdeflate CPU doens't support chunk sizes larger than "
                  << gdeflate::nvcompGdeflateCPUCompressionMaxAllowedChunkSize
                  << std::endl;
        return 1;
      }
    }
  }

  if (filename.empty()) {
    std::cerr << "Usage: " << argv[0] << " -f <filename> [options]" << std::endl;
    return 1;
  }

  // Read input file as single chunk
  std::vector<std::vector<char>> inputs;
  inputs = multi_file({filename}, true, chunk_size, 1, duplicate_count);

  if (inputs.empty() || inputs[0].empty()) {
    std::cerr << "Failed to read input file or file is empty" << std::endl;
    return 1;
  }

  const size_t batch_size = inputs.size();
  size_t total_input_bytes = 0;
  size_t max_input_chunk_size = 0;
  std::vector<size_t> input_sizes;
  input_sizes.reserve(batch_size);

  for (const std::vector<char>& chunk : inputs) {
    auto input_chunk_size = chunk.size();
    total_input_bytes += input_chunk_size;
    max_input_chunk_size = std::max(input_chunk_size, max_input_chunk_size);
    input_sizes.emplace_back(input_chunk_size);
  }

  // Allocate space for compressed data
  size_t max_compressed_chunk_size;
  gdeflate::compressCPUGetMaxOutputChunkSize(max_input_chunk_size, &max_compressed_chunk_size);
  std::vector<std::vector<char>> compressed_data(batch_size, std::vector<char>(max_compressed_chunk_size));
  std::vector<const char*> in_ptrs(batch_size);
  std::vector<char*> compressed_ptrs(batch_size);
  std::vector<size_t> compressed_sizes(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    in_ptrs[i] = inputs[i].data();
    compressed_ptrs[i] = compressed_data[i].data();
  }

  // Allocate space for decompression
  std::vector<size_t> reported_decompressed_sizes(batch_size);
  std::vector<std::vector<char>> decompressed_data(batch_size);
  std::vector<char*> decompressed_ptrs(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    decompressed_data[i].resize(input_sizes[i]);
    decompressed_ptrs[i] = decompressed_data[i].data();
  }

  // Warmup
  for (size_t w = 0; w < warmup_count; ++w) {
    gdeflate::compressCPU(
        reinterpret_cast<const void* const*>(in_ptrs.data()),
        input_sizes.data(),
        max_input_chunk_size,
        batch_size,
        reinterpret_cast<void* const*>(compressed_ptrs.data()),
        compressed_sizes.data(),
        compression_level);
  }

  // Benchmark compression
  auto start_time = std::chrono::high_resolution_clock::now();

  for (size_t iter = 0; iter < iteration_count; ++iter) {
    gdeflate::compressCPU(
        reinterpret_cast<const void* const*>(in_ptrs.data()),
        input_sizes.data(),
        max_input_chunk_size,
        batch_size,
        reinterpret_cast<void* const*>(compressed_ptrs.data()),
        compressed_sizes.data(),
        compression_level);
  }

  auto compress_end_time = std::chrono::high_resolution_clock::now();

  // Benchmark decompression
  for (size_t iter = 0; iter < iteration_count; ++iter) {
    gdeflate::decompressCPU(
      reinterpret_cast<const void* const*>(compressed_ptrs.data()),
      compressed_sizes.data(),
      batch_size,
      reinterpret_cast<void* const*>(decompressed_ptrs.data()),
      input_sizes.data(),
      reported_decompressed_sizes.data()
    );
  }

  auto decompress_end_time = std::chrono::high_resolution_clock::now();

  // Do exact byte comparison on the output
  try {
    for (size_t i = 0; i < batch_size; ++i) {
      benchmark_assert(input_sizes[i] == reported_decompressed_sizes[i],
                       "The reported decompressed size does not match with the input size in chunk i=" + std::to_string(i));
      benchmark_assert(std::memcmp(in_ptrs[i], decompressed_ptrs[i], input_sizes[i]) == 0,
                       "The decompressed data did not match the input in chunk i=" + std::to_string(i));
    }
  } catch (const std::exception& e) {
    std::cerr << "Verification failed: " << e.what() << std::endl;
    return 1;
  }

  // Calculate metrics
  size_t total_compressed_bytes = std::accumulate(compressed_sizes.begin(), compressed_sizes.end(), size_t(0));

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
              << std::to_string(inputs.size()) << ","
              << (total_input_bytes / inputs.size() / 1024) << ","
              << (max_input_chunk_size / 1024) << ","
              << total_input_bytes << ","
              << total_compressed_bytes << ","
              << std::fixed << std::setprecision(2) << compression_ratio << ","
              << compression_throughput << ","
              << decompression_throughput << std::endl;
  }

  return 0;
}
