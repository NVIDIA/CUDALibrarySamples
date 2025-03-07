/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#include <nvcomp/native/gdeflate_cpu.h>
#include <nvcomp/gdeflate.h>
#include "BatchData.h"


static void run_example(const std::vector<std::vector<char>>& data,
                        size_t warmup_iteration_count, size_t total_iteration_count)
{
  assert(!data.empty());
  if(warmup_iteration_count >= total_iteration_count) {
    throw std::runtime_error("ERROR: the total iteration count must be greater than the warmup iteration count");
  }

  size_t total_bytes = 0;
  for (const std::vector<char>& part : data) {
    total_bytes += part.size();
  }

  std::cout << "----------" << std::endl;
  std::cout << "files: " << data.size() << std::endl;
  std::cout << "uncompressed (B): " << total_bytes << std::endl;

  const size_t chunk_size = 1 << 16;

  // Build up input batch on CPU
  BatchDataCPU input_data_cpu(data, chunk_size);
  const size_t chunk_count = input_data_cpu.size();
  std::cout << "chunks: " << chunk_count << std::endl;

  // compression

  // Get max output size per chunk
  nvcompStatus_t status;
  size_t max_out_bytes;
  status = nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
      chunk_size, nvcompBatchedGdeflateDefaultOpts, &max_out_bytes);
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: nvcompBatchedGdeflateCompressGetMaxOutputChunkSize() not successful");
  }

  // Allocate and prepare output/compressed batch
  BatchDataCPU compressed_data_cpu(max_out_bytes, chunk_count);

  // Compress on the CPU using gdeflate CPU batched API
  gdeflate::compressCPU(
      input_data_cpu.ptrs(),
      input_data_cpu.sizes(),
      chunk_size,
      chunk_count,
      compressed_data_cpu.ptrs(),
      compressed_data_cpu.sizes());

  // compute compression ratio
  size_t* compressed_sizes_host = compressed_data_cpu.sizes();
  size_t comp_bytes = 0;
  for (size_t i = 0; i < chunk_count; ++i)
    comp_bytes += compressed_sizes_host[i];

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;

  // Copy compressed data to GPU
  BatchData compressed_data(compressed_data_cpu, true, nvcompBatchedGdeflateDecompressRequiredAlignments.input);

  // Allocate and build up decompression batch on GPU
  BatchData decomp_data(input_data_cpu, false, nvcompBatchedGdeflateDecompressRequiredAlignments.output);

  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // CUDA events to measure decompression time
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  // gdeflate GPU decompression
  size_t decomp_temp_bytes;
  status = nvcompBatchedGdeflateDecompressGetTempSize(
      chunk_count, chunk_size, &decomp_temp_bytes);
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: nvcompBatchedGdeflateDecompressGetTempSize() not successful");
  }

  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

  size_t* d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, chunk_count * sizeof(size_t)));

  nvcompStatus_t* d_statuses;
  CUDA_CHECK(cudaMalloc(&d_statuses, chunk_count * sizeof(nvcompStatus_t)));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto perform_decompression = [&]() {
    if (nvcompBatchedGdeflateDecompressAsync(
          compressed_data.ptrs(),
          compressed_data.sizes(),
          decomp_data.sizes(),
          d_decomp_sizes,
          chunk_count,
          d_decomp_temp,
          decomp_temp_bytes,
          decomp_data.ptrs(),
          d_statuses,
          stream) != nvcompSuccess) {
      throw std::runtime_error("ERROR: nvcompBatchedGdeflateDecompressAsync() not successful");
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
  CUDA_CHECK(cudaFree(d_statuses));

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
