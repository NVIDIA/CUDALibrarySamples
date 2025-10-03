/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "snappy.h"
#include "nvcomp/snappy.h"
#include "BatchData.h"


static void run_example(const std::vector<std::vector<char>>& data,
                        size_t warmup_iteration_count, size_t total_iteration_count)
{
  assert(!data.empty());
  if(warmup_iteration_count >= total_iteration_count) {
    throw std::runtime_error("ERROR: the total iteration count must be greater than the warmup iteration count");
  }

  size_t total_bytes =
    std::accumulate(data.begin(), data.end(), size_t(0), [](const size_t& a, const std::vector<char>& part) {
        return a + part.size();
  });

  std::cout << "----------" << std::endl;
  std::cout << "files: " << data.size() << std::endl;
  std::cout << "uncompressed (B): " << total_bytes << std::endl;

  constexpr size_t chunk_size = 1 << 16;
  static_assert(chunk_size <= snappy::kBlockSize, "Chunk size must be less than the constant specified in the Snappy library");

  // Build up input batch on CPU
  BatchDataCPU input_data_cpu(data, chunk_size);
  const size_t chunk_count = input_data_cpu.size();
  std::cout << "chunks: " << chunk_count << std::endl;

  // compression

  // Allocate and prepare output/compressed batch
  BatchDataCPU compressed_data_cpu(
      snappy::MaxCompressedLength(chunk_size), chunk_count);

  // loop over chunks on the CPU, compressing each one
  for (size_t i = 0; i < chunk_count; ++i) {
    snappy::RawCompress(static_cast<const char*>(input_data_cpu.ptrs()[i]),
                   input_data_cpu.sizes()[i],
                   static_cast<char*>(compressed_data_cpu.ptrs()[i]),
                   &compressed_data_cpu.sizes()[i]);
    if (compressed_data_cpu.sizes()[i] == 0) {
      throw std::runtime_error(
          "Snappy CPU failed to compress chunk " + std::to_string(i) + ".");
    }
  }

  // compute compression ratio
  size_t comp_bytes = std::accumulate(compressed_data_cpu.sizes(), compressed_data_cpu.sizes() + chunk_count, size_t(0));

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;

  // Decompression options
  nvcompBatchedSnappyDecompressOpts_t decompress_opts = nvcompBatchedSnappyDecompressDefaultOpts;

  // Query decompression alignment requirements
  nvcompAlignmentRequirements_t decompression_alignment_reqs;
  nvcompStatus_t status = nvcompBatchedSnappyDecompressGetRequiredAlignments(
    decompress_opts,
    &decompression_alignment_reqs);
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: nvcompBatchedSnappyDecompressGetRequiredAlignments() not successful");
  }

  // Copy compressed data to GPU
  BatchData compressed_data(compressed_data_cpu, true, decompression_alignment_reqs.input);

  // Allocate and build up decompression batch on GPU
  BatchData decomp_data(input_data_cpu, false, decompression_alignment_reqs.output);

  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // CUDA events to measure decompression time
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  // Snappy GPU decompression
  size_t decomp_temp_bytes;
  status = nvcompBatchedSnappyDecompressGetTempSizeAsync(
      chunk_count,
      chunk_size,
      decompress_opts,
      &decomp_temp_bytes,
      chunk_count * chunk_size);
  if (status != nvcompSuccess) {
    throw std::runtime_error("nvcompBatchedSnappyDecompressGetTempSizeAsync() failed.");
  }

  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

  size_t* d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, chunk_count * sizeof(size_t)));

  nvcompStatus_t* d_status_ptrs;
  CUDA_CHECK(cudaMalloc(&d_status_ptrs, chunk_count * sizeof(nvcompStatus_t)));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto perform_decompression = [&]() {
    if (nvcompBatchedSnappyDecompressAsync(
          compressed_data.ptrs(),
          compressed_data.sizes(),
          decomp_data.sizes(),
          d_decomp_sizes,
          chunk_count,
          d_decomp_temp,
          decomp_temp_bytes,
          decomp_data.ptrs(),
          decompress_opts,
          d_status_ptrs,
          stream) != nvcompSuccess) {
      throw std::runtime_error("ERROR: nvcompBatchedSnappyDecompressAsync() not successful");
    }
  };

  // Run warm-up decompression
  for (size_t iter = 0; iter < warmup_iteration_count; ++iter) {
    perform_decompression();
  }

  // Re-run decompression to get throughput
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (size_t iter = warmup_iteration_count; iter < total_iteration_count; ++iter) {
    perform_decompression();
  }
  CUDA_CHECK(cudaEventRecord(end, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Validate decompressed data against input
  if (!(input_data_cpu == decomp_data)) {
    throw std::runtime_error("Failed to validate decompressed data");
  } else {
    std::cout << "decompression validated :)" << std::endl;
  }

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
  ms /= total_iteration_count - warmup_iteration_count;

  double decompression_throughput = ((double)total_bytes / ms) * 1e-6;
  std::cout << "decompression throughput (GB/s): " << decompression_throughput
            << std::endl;

  CUDA_CHECK(cudaFree(d_decomp_temp));
  CUDA_CHECK(cudaFree(d_decomp_sizes));
  CUDA_CHECK(cudaFree(d_status_ptrs));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char* argv[])
{
  std::vector<std::string> file_names;

  size_t warmup_iteration_count = 2;
  size_t total_iteration_count = 5;

  do {
    if (argc < 3) {
      break;
    }

    int i = 1;
    while (i < argc) {
      const char* current_argv = argv[i++];
      if (strcmp(current_argv, "-f") == 0) {
        // parse until next `-` argument
        while (i < argc && argv[i][0] != '-') {
          file_names.emplace_back(argv[i++]);
        }
      } else {
        std::cerr << "Unknown argument: " << current_argv << std::endl;
        return 1;
      }
    }
  } while (0);

  if (file_names.empty()) {
    std::cerr << "Must specify at least one file via '-f <file>'." << std::endl;
    return 1;
  }

  auto data = multi_file(file_names);

  run_example(data, warmup_iteration_count, total_iteration_count);

  return 0;
}