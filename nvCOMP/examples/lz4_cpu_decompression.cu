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

#include "lz4.h"
#include "lz4hc.h"
#include "nvcomp/lz4.h"

BatchDataCPU GetBatchDataCPU(const BatchData& batch_data, bool copy_data)
{
  BatchDataCPU compress_data_cpu(
      batch_data.ptrs(),
      batch_data.sizes(),
      batch_data.data(),
      batch_data.size(),
      copy_data);
  return compress_data_cpu;
}

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

  // build up metadata
  BatchData input_data(data, chunk_size);

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  nvcompStatus_t status = nvcompBatchedLZ4CompressGetTempSize(
      input_data.size(),
      chunk_size,
      nvcompBatchedLZ4DefaultOpts,
      &comp_temp_bytes);
  if( status != nvcompSuccess){
    throw std::runtime_error("ERROR: nvcompBatchedLZ4CompressGetTempSize() not successful");
  }

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t max_out_bytes;
  status = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
      chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);
  if( status != nvcompSuccess){
    throw std::runtime_error("ERROR: nvcompBatchedLZ4CompressGetMaxOutputChunkSize() not successful");
  }

  BatchData compress_data(max_out_bytes, input_data.size());

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(start, stream));

  status = nvcompBatchedLZ4CompressAsync(
      input_data.ptrs(),
      input_data.sizes(),
      chunk_size,
      input_data.size(),
      d_comp_temp,
      comp_temp_bytes,
      compress_data.ptrs(),
      compress_data.sizes(),
      nvcompBatchedLZ4DefaultOpts,
      stream);
  if (status != nvcompSuccess) {
    throw std::runtime_error("nvcompBatchedLZ4CompressAsync() failed.");
  }
  
  CUDA_CHECK(cudaEventRecord(end, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // free compression memory
  CUDA_CHECK(cudaFree(d_comp_temp));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));

  // compute compression ratio
  std::vector<size_t> compressed_sizes_host(compress_data.size());
  CUDA_CHECK(cudaMemcpy(
      compressed_sizes_host.data(),
      compress_data.sizes(),
      compress_data.size() * sizeof(*compress_data.sizes()),
      cudaMemcpyDeviceToHost));

  size_t comp_bytes = 0;
  for (const size_t s : compressed_sizes_host) {
    comp_bytes += s;
  }

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;
  std::cout << "compression throughput (GB/s): "
            << (double)total_bytes / (1.0e6 * ms) << std::endl;

  // Allocate and prepare output/compressed batch
  BatchDataCPU compress_data_cpu = GetBatchDataCPU(compress_data, true);
  BatchDataCPU decompress_data_cpu = GetBatchDataCPU(input_data, false);

  // loop over chunks on the CPU, decompressing each one
  for (size_t i = 0; i < input_data.size(); ++i) {
    const int size = LZ4_decompress_safe(
        static_cast<const char*>(compress_data_cpu.ptrs()[i]),
        static_cast<char*>(decompress_data_cpu.ptrs()[i]),
        static_cast<int>(compress_data_cpu.sizes()[i]),
        static_cast<int>(decompress_data_cpu.sizes()[i]));
    if (size == 0) {
      throw std::runtime_error(
          "LZ4 CPU failed to decompress chunk " + std::to_string(i) + ".");
    }
  }
  // Validate decompressed data against input
  if (!(decompress_data_cpu == input_data))
    throw std::runtime_error("Failed to validate CPU decompressed data");
  else
    std::cout << "CPU decompression validated :)" << std::endl;

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
