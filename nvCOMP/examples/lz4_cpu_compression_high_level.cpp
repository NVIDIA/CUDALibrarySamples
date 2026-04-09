/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * Example: compress on CPU with lz4_cpu.hpp (LZ4CPUManager) and decompress on
 * GPU with lz4.hpp (create_manager from compressed buffer, then configure_
 * decompression / decompress). Demonstrates C++ high-level API usage.
 *
 * Build: see examples/CMakeLists.txt (lz4_cpu_compression_high_level).
 * Run: ./lz4_cpu_compression_high_level -f <file> [ -f <file2> ... ]
 */

#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "util.h"

#include "nvcomp/nvcompManagerFactory.hpp"
#include "nvcomp/lz4_cpu.hpp"

using namespace nvcomp;

static void run_example(const std::vector<std::vector<char>>& data)
{
  assert(!data.empty());

  size_t total_bytes = 0;
  for (const std::vector<char>& part : data) {
    total_bytes += part.size();
  }

  constexpr size_t CHUNK_SIZE = 1 << 16;
  constexpr int COMPRESSION_LEVEL = 10;

  size_t uncomp_size = total_bytes;

  std::vector<uint8_t> host_input(uncomp_size, 0);
  size_t offset = 0;
  for (const std::vector<char>& part : data) {
    std::memcpy(host_input.data() + offset, part.data(), part.size());
    offset += part.size();
  }

  std::cout << "----------" << std::endl;
  std::cout << "files: " << data.size() << std::endl;
  std::cout << "uncompressed (B): " << uncomp_size << std::endl;

  LZ4CPUManager cpu_manager(CHUNK_SIZE, COMPRESSION_LEVEL);

  CompressionConfig comp_config = cpu_manager.configure_compression(uncomp_size);

  std::vector<uint8_t> host_comp(comp_config.max_compressed_buffer_size);

  cpu_manager.compress(host_input.data(), host_comp.data(), comp_config);

  size_t comp_size = cpu_manager.get_compressed_output_size(host_comp.data());

  std::cout << "comp_size: " << comp_size
            << ", ratio: " << std::fixed << std::setprecision(2)
            << (double)uncomp_size / comp_size << std::endl;

  cudaStream_t stream = cudaStreamDefault;

  uint8_t* d_comp = nullptr;
  CUDA_CHECK(cudaMallocSafe(&d_comp, comp_size));
  CUDA_CHECK(cudaMemcpyAsync(d_comp, host_comp.data(), comp_size, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::shared_ptr<nvcompManagerBase> decomp_manager =
      create_manager(d_comp, stream, NoComputeNoVerify, NVCOMP_DECOMPRESS_BACKEND_DEFAULT, false);

  DecompressionConfig decomp_config = decomp_manager->configure_decompression(d_comp);
  if (decomp_config.decomp_data_size != uncomp_size) {
    CUDA_CHECK(cudaFree(d_comp));
    throw std::runtime_error("configure_decompression size mismatch: got "
        + std::to_string(decomp_config.decomp_data_size) + " expected " + std::to_string(uncomp_size));
  }

  uint8_t* d_decomp = nullptr;
  CUDA_CHECK(cudaMallocSafe(&d_decomp, decomp_config.decomp_data_size));
  decomp_manager->decompress(d_decomp, d_comp, decomp_config);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  nvcompStatus_t decomp_status = *decomp_config.get_status();
  if (decomp_status != nvcompSuccess) {
    CUDA_CHECK(cudaFree(d_comp));
    CUDA_CHECK(cudaFree(d_decomp));
    throw std::runtime_error("Decompression failed: " + std::to_string(static_cast<int>(decomp_status)));
  }

  std::vector<uint8_t> host_decomp(uncomp_size);
  CUDA_CHECK(cudaMemcpy(host_decomp.data(), d_decomp, uncomp_size, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_comp));
  CUDA_CHECK(cudaFree(d_decomp));

  if (std::memcmp(host_decomp.data(), host_input.data(), uncomp_size) != 0) {
    throw std::runtime_error("Decompressed data does not match input");
  }

  std::cout << "Decompression validated :)" << std::endl;
}

int main(int argc, char* argv[])
{
  std::vector<std::string> file_names;

  int i = 1;
  while (i < argc) {
    const char* arg = argv[i++];
    if (strcmp(arg, "-f") == 0) {
      while (i < argc && argv[i][0] != '-') {
        file_names.emplace_back(argv[i++]);
      }
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  if (file_names.empty()) {
    std::cerr << "Usage: " << (argc ? argv[0] : "lz4_cpu_compression_high_level")
              << " -f <file> [ -f <file2> ... ]" << std::endl;
    return 1;
  }

  try {
    std::vector<std::vector<char>> data = multi_file(file_names);
    run_example(data);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
