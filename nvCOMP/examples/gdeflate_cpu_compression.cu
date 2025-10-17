/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#include "BatchData.h"

#include <nvcomp/native/gdeflate_cpu.h>
#include <nvcomp/gdeflate.h>

// Benchmark performance from the binary data file fname
static void run_example(const std::vector<std::vector<char>>& data)
{
  size_t total_bytes = 0;
  for (const std::vector<char>& part : data) {
    total_bytes += part.size();
  }

  std::cout << "----------" << std::endl;
  std::cout << "files: " << data.size() << std::endl;
  std::cout << "uncompressed (B): " << total_bytes << std::endl;

  const size_t chunk_size = 1 << 16;

  // build up input batch on CPU
  BatchDataCPU input_data_cpu(data, chunk_size);
  std::cout << "chunks: " << input_data_cpu.size() << std::endl;

  // compression

  // Get max output size per chunk
  nvcompStatus_t status;
  size_t max_out_bytes;
  status = nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
      chunk_size, nvcompBatchedGdeflateDefaultOpts, &max_out_bytes);
  if( status != nvcompSuccess){
    throw std::runtime_error("ERROR: nvcompBatchedGdeflateCompressGetMaxOutputChunkSize() not successful");
  }

  // Allocate and prepare output/compressed batch
  BatchDataCPU compress_data_cpu(max_out_bytes, input_data_cpu.size());

  // Compress on the CPU using gdeflate CPU batched API
  gdeflate::compressCPU(
      input_data_cpu.ptrs(),
      input_data_cpu.sizes(),
      chunk_size,
      input_data_cpu.size(),
      compress_data_cpu.ptrs(),
      compress_data_cpu.sizes());

  // compute compression ratio
  size_t* compressed_sizes_host = compress_data_cpu.sizes();
  size_t comp_bytes = 0;
  for (size_t i = 0; i < compress_data_cpu.size(); ++i)
    comp_bytes += compressed_sizes_host[i];

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;

  // Copy compressed data to GPU
  BatchData compress_data(compress_data_cpu, true);

  // Allocate and build up decompression batch on GPU
  BatchData decomp_data(input_data_cpu, false);

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
      compress_data.size(), chunk_size, &decomp_temp_bytes);
  if( status != nvcompSuccess){
    throw std::runtime_error("ERROR: nvcompBatchedGdeflateDecompressGetTempSize() not successful");
  }

  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

  size_t* d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, decomp_data.size() * sizeof(size_t)));

  nvcompStatus_t* d_statuses;
  CUDA_CHECK(cudaMalloc(&d_statuses, decomp_data.size() * sizeof(nvcompStatus_t)));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Run decompression
  status = nvcompBatchedGdeflateDecompressAsync(
      compress_data.ptrs(),
      compress_data.sizes(),
      decomp_data.sizes(),
      d_decomp_sizes,
      compress_data.size(),
      d_decomp_temp,
      decomp_temp_bytes,
      decomp_data.ptrs(),
      d_statuses,
      stream);
  if( status != nvcompSuccess){
    throw std::runtime_error("ERROR: nvcompBatchedGdeflateDecompressAsync() not successful");
  }

  // Validate decompressed data against input
  if (!(input_data_cpu == decomp_data))
    throw std::runtime_error("Failed to validate decompressed data");
  else
    std::cout << "decompression validated :)" << std::endl;

  // Re-run decompression to get throughput
  CUDA_CHECK(cudaEventRecord(start, stream));
  status = nvcompBatchedGdeflateDecompressAsync(
      compress_data.ptrs(),
      compress_data.sizes(),
      decomp_data.sizes(),
      d_decomp_sizes,
      compress_data.size(),
      d_decomp_temp,
      decomp_temp_bytes,
      decomp_data.ptrs(),
      d_statuses,
      stream);
  CUDA_CHECK(cudaEventRecord(end, stream));
  if( status != nvcompSuccess){
    throw std::runtime_error("ERROR: nvcompBatchedGdeflateDecompressAsync() not successful");
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));

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

std::vector<char> readFile(const std::string& filename)
{
  std::vector<char> buffer(4096);
  std::vector<char> host_data;

  std::ifstream fin(filename, std::ifstream::binary);
  fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  size_t num;
  do {
    num = fin.readsome(buffer.data(), buffer.size());
    host_data.insert(host_data.end(), buffer.begin(), buffer.begin() + num);
  } while (num > 0);

  return host_data;
}

std::vector<std::vector<char>>
multi_file(const std::vector<std::string>& filenames)
{
  std::vector<std::vector<char>> split_data;

  for (auto const& filename : filenames) {
    split_data.emplace_back(readFile(filename));
  }

  return split_data;
}

int main(int argc, char* argv[])
{
  std::vector<std::string> file_names(argc - 1);

  if (argc == 1) {
    std::cerr << "Must specify at least one file." << std::endl;
    return 1;
  }

  // if `-f` is specified, assume single file mode
  if (strcmp(argv[1], "-f") == 0) {
    if (argc == 2) {
      std::cerr << "Missing file name following '-f'" << std::endl;
      return 1;
    } else if (argc > 3) {
      std::cerr << "Unknown extra arguments with '-f'." << std::endl;
      return 1;
    }

    file_names = {argv[2]};
  } else {
    // multi-file mode
    for (int i = 1; i < argc; ++i) {
      file_names[i - 1] = argv[i];
    }
  }

  auto data = multi_file(file_names);

  run_example(data);

  return 0;
}
