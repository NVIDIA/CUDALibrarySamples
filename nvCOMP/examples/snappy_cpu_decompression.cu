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


BatchDataCPU GetBatchDataCPU(const BatchData& batch_data, bool copy_data)
{
  BatchDataCPU batch_data_cpu(
      batch_data.ptrs(),
      batch_data.sizes(),
      batch_data.data(),
      batch_data.size(),
      copy_data);
  return batch_data_cpu;
}

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
  static_assert(chunk_size <= nvcompSnappyCompressionMaxAllowedChunkSize, "Chunk size must be less than the constant specified in the nvCOMP library");

  auto nvcompBatchedSnappyOpts = nvcompBatchedSnappyCompressDefaultOpts;

  // Query compression alignment requirements
  nvcompAlignmentRequirements_t compression_alignment_reqs;
  nvcompStatus_t status = nvcompBatchedSnappyCompressGetRequiredAlignments(
    nvcompBatchedSnappyOpts,
    &compression_alignment_reqs);
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: nvcompBatchedSnappyCompressGetRequiredAlignments() not successful");
  }

  // Build up GPU data
  BatchData input_data(data, chunk_size, compression_alignment_reqs.input);
  const size_t chunk_count = input_data.size();

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = nvcompBatchedSnappyCompressGetTempSizeAsync(
      chunk_count,
      chunk_size,
      nvcompBatchedSnappyOpts,
      &comp_temp_bytes,
      chunk_count * chunk_size);
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: nvcompBatchedSnappyCompressGetTempSizeAsync() not successful");
  }

  void* d_comp_temp;
  CUDA_CHECK(cudaMallocSafe(&d_comp_temp, comp_temp_bytes));

  size_t max_out_bytes;
  status = nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
      chunk_size, nvcompBatchedSnappyOpts, &max_out_bytes);
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: nvcompBatchedSnappyCompressGetMaxOutputChunkSize() not successful");
  }

  BatchData compressed_data(max_out_bytes, chunk_count, compression_alignment_reqs.output);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  auto perform_compression = [&]() {
    if (nvcompBatchedSnappyCompressAsync(
          input_data.ptrs(),
          input_data.sizes(),
          chunk_size,
          chunk_count,
          d_comp_temp,
          comp_temp_bytes,
          compressed_data.ptrs(),
          compressed_data.sizes(),
          nvcompBatchedSnappyOpts,
          nullptr,
          stream) != nvcompSuccess) {
      throw std::runtime_error("nvcompBatchedSnappyCompressAsync() failed.");
    }
  };

  // Run warm-up compression
  for (size_t iter = 0; iter < warmup_iteration_count; ++iter) {
    perform_compression();
  }

  // Re-run compression to get throughput
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (size_t iter = warmup_iteration_count; iter < total_iteration_count; ++iter) {
    perform_compression();
  }
  CUDA_CHECK(cudaEventRecord(end, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
  ms /= total_iteration_count - warmup_iteration_count;

  // compute compression ratio
  std::vector<size_t> compressed_sizes_host(chunk_count);
  CUDA_CHECK(cudaMemcpy(
      compressed_sizes_host.data(),
      compressed_data.sizes(),
      chunk_count * sizeof(*compressed_data.sizes()),
      cudaMemcpyDeviceToHost));

  size_t comp_bytes = std::accumulate(compressed_sizes_host.begin(), compressed_sizes_host.end(), size_t(0));

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;
  std::cout << "compression throughput (GB/s): "
            << (double)total_bytes / (1.0e6 * ms) << std::endl;

  // Allocate and prepare output/compressed batch
  BatchDataCPU compressed_data_cpu = GetBatchDataCPU(compressed_data, true);
  BatchDataCPU decompressed_data_cpu = GetBatchDataCPU(input_data, false);

  // loop over chunks on the CPU, decompressing each one
  for (size_t i = 0; i < chunk_count; ++i) {
    bool result = snappy::GetUncompressedLength(
                    static_cast<const char*>(compressed_data_cpu.ptrs()[i]),
                    compressed_data_cpu.sizes()[i],
                    &decompressed_data_cpu.sizes()[i]);
    result = result &&
                  snappy::RawUncompress(
                    static_cast<const char*>(compressed_data_cpu.ptrs()[i]),
                    compressed_data_cpu.sizes()[i],
                    static_cast<char*>(decompressed_data_cpu.ptrs()[i]));
    if (!result || decompressed_data_cpu.sizes()[i] == 0) {
      throw std::runtime_error(
          "Snappy CPU failed to decompress chunk " + std::to_string(i) + ".");
    }
  }
  // Validate decompressed data against input
  if (!(decompressed_data_cpu == input_data)) {
    throw std::runtime_error("Failed to validate CPU decompressed data");
  } else {
    std::cout << "CPU decompression validated :)" << std::endl;
  }

  CUDA_CHECK(cudaFree(d_comp_temp));

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
