/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "zlib.h"
#include "libdeflate.h"
#include "nvcomp/deflate.h"
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
                        int algo,
                        size_t warmup_iteration_count, size_t total_iteration_count)
{
  assert(!data.empty());
  assert(algo >= 0 && algo <= 1);
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

  constexpr size_t chunk_size = 1 << 16;
  static_assert(chunk_size <= nvcompDeflateCompressionMaxAllowedChunkSize, "Chunk size must be less than the constant specified in the nvCOMP library");

  // Compression options
  auto nvcompBatchedDeflateOpts = nvcompBatchedDeflateCompressDefaultOpts;

  // Query compression alignment requirements
  nvcompAlignmentRequirements_t compression_alignment_reqs;
  nvcompStatus_t status = nvcompBatchedDeflateCompressGetRequiredAlignments(
    nvcompBatchedDeflateOpts,
    &compression_alignment_reqs);
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: nvcompBatchedDeflateCompressGetRequiredAlignments() not successful");
  }

  // Build up GPU data
  BatchData input_data(data, chunk_size, compression_alignment_reqs.input);
  const size_t chunk_count = input_data.size();

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = nvcompBatchedDeflateCompressGetTempSizeAsync(
      chunk_count,
      chunk_size,
      nvcompBatchedDeflateOpts,
      &comp_temp_bytes,
      chunk_count * chunk_size);
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: nvcompBatchedDeflateCompressGetTempSizeAsync() not successful");
  }

  void* d_comp_temp;
  CUDA_CHECK(cudaMallocSafe(&d_comp_temp, comp_temp_bytes));

  size_t max_out_bytes;
  status = nvcompBatchedDeflateCompressGetMaxOutputChunkSize(
      chunk_size, nvcompBatchedDeflateOpts, &max_out_bytes);
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: nvcompBatchedDeflateCompressGetMaxOutputChunkSize() not successful");
  }

  BatchData compressed_data(max_out_bytes, chunk_count, compression_alignment_reqs.output);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  auto perform_compression = [&]() {
    if (nvcompBatchedDeflateCompressAsync(
          input_data.ptrs(),
          input_data.sizes(),
          chunk_size,
          chunk_count,
          d_comp_temp,
          comp_temp_bytes,
          compressed_data.ptrs(),
          compressed_data.sizes(),
          nvcompBatchedDeflateOpts,
          nullptr,
          stream) != nvcompSuccess) {
      throw std::runtime_error("nvcompBatchedDeflateCompressAsync() failed.");
    }
  };

  // Warm-up compression iterations
  for (size_t iter = 0; iter < warmup_iteration_count; ++iter) {
    perform_compression();
  }

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
  BatchDataCPU compressed_data_cpu = GetBatchDataCPU(compressed_data, true);
  BatchDataCPU decompressed_data_cpu = GetBatchDataCPU(input_data, false);

  // loop over chunks on the CPU, decompressing each one
  for (size_t i = 0; i < chunk_count; ++i) {
    if (algo == 0) {
        struct libdeflate_decompressor  *decompressor;
        decompressor = libdeflate_alloc_decompressor();
        enum libdeflate_result res = libdeflate_deflate_decompress(decompressor, compressed_data_cpu.ptrs()[i], compressed_data_cpu.sizes()[i], 
                                                   decompressed_data_cpu.ptrs()[i], decompressed_data_cpu.sizes()[i], NULL);

       if (res != LIBDEFLATE_SUCCESS) {
       throw std::runtime_error(
           "libdeflate CPU failed to decompress chunk " + std::to_string(i) + ".");
       }
    } else if (algo == 1) {
        z_stream zs1;
        zs1.zalloc = NULL;
        zs1.zfree = NULL;
        zs1.msg = NULL;
        zs1.next_in = (Bytef*)compressed_data_cpu.ptrs()[i];
        zs1.avail_in = static_cast<uInt>(compressed_data_cpu.sizes()[i]);
        zs1.next_out = (Bytef*)decompressed_data_cpu.ptrs()[i];
        zs1.avail_out = static_cast<uInt>(decompressed_data_cpu.sizes()[i]);
        // -15 to disable zlib header/footer (raw deflate)
        int ret = inflateInit2(&zs1, -15);
        if (ret != Z_OK) {
           throw std::runtime_error("inflateInit2 error " + std::to_string(ret));
        }
        if ((ret = inflate(&zs1, Z_FINISH)) != Z_STREAM_END) {
           throw std::runtime_error("zlib::inflate operation fail " + std::to_string(ret));;
            if ((ret = inflateEnd(&zs1)) != Z_OK) {
               throw std::runtime_error("Call to inflateEnd failed: " + std::to_string(ret));
            }
        }
        if ((ret = inflateEnd(&zs1)) != Z_OK) {
           throw std::runtime_error("Call to inflateEnd failed: " + std::to_string(ret));
        }
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

  int algo = -1;
  size_t warmup_iteration_count = 2;
  size_t total_iteration_count = 5;

  do {
    if (argc < 5) {
      break;
    }

    int i = 1;
    while (i < argc) {
      const char* current_argv = argv[i++];
      if (strcmp(current_argv, "-a") == 0) {
        if(i >= argc) {
          std::cerr << "Missing value for argument '-a <algorithm>'" << std::endl;
          return 1;
        }
        algo = atoi(argv[i++]);
      } else if (strcmp(current_argv, "-f") == 0) {
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

  if (argc < 5) {
    std::cerr << "Must choose an algorithm via '-a <algo>', and must specify at least one file via '-f <file>'." << std::endl;
    return 1;
  } else if (algo < 0 || algo > 1) {
    std::cerr << "Must choose an algorithm via '-a <algo>'. '<algo>' can be 0 or 1. (0 libdeflate, 1 zlib_inflate)" << std::endl;
    return 1;
  } else if (file_names.empty()) {
   std::cerr << "Must specify at least one file via '-f <file>'" << std::endl;
   return 1;
  }

  auto data = multi_file(file_names);

  run_example(data, algo, warmup_iteration_count, total_iteration_count);

  return 0;
}
