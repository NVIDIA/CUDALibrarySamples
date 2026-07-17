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

#include "BatchData.h"
#include "nvcomp/gzip.h"

#include <zlib.h>

static void
run_example(const std::vector<std::vector<char>> &data, size_t warmup_iteration_count, size_t total_iteration_count)
{
  assert(!data.empty());
  if (warmup_iteration_count >= total_iteration_count)
  {
    throw std::runtime_error("ERROR: the total iteration count must be greater than the warmup iteration count");
  }

  size_t total_bytes = 0;
  for (const std::vector<char> &part : data)
  {
    total_bytes += part.size();
  }

  std::cout << "----------" << std::endl;
  std::cout << "files: " << data.size() << std::endl;
  std::cout << "uncompressed (B): " << total_bytes << std::endl;

  const size_t chunk_size = 1 << 16;
  static_assert(
    chunk_size <= nvcompGzipCompressionMaxAllowedChunkSize,
    "Chunk size must be less than the maximum chunk size supported by the nvCOMP Gzip compressor"
  );

  // Build up input batch on CPU, then copy it to the GPU
  BatchDataCPU input_data_cpu(data, chunk_size);
  const size_t chunk_count = input_data_cpu.size();
  std::cout << "chunks: " << chunk_count << std::endl;

  // Compression options
  nvcompBatchedGzipCompressOpts_t compress_opts = nvcompBatchedGzipCompressDefaultOpts;

  // Query compression alignment requirements
  nvcompAlignmentRequirements_t compression_alignment_reqs{};
  nvcompStatus_t status = nvcompBatchedGzipCompressGetRequiredAlignments(compress_opts, &compression_alignment_reqs);
  if (status != nvcompSuccess)
  {
    throw std::runtime_error("ERROR: nvcompBatchedGzipCompressGetRequiredAlignments() not successful");
  }

  // Copy uncompressed data to the GPU
  BatchData input_data(input_data_cpu, true, compression_alignment_reqs.input);

  // Query the maximum compressed size of a chunk. Gzip framing adds overhead, so the
  // output can exceed the input size for incompressible data.
  size_t max_compressed_chunk_size;
  status = nvcompBatchedGzipCompressGetMaxOutputChunkSize(chunk_size, compress_opts, &max_compressed_chunk_size);
  if (status != nvcompSuccess)
  {
    throw std::runtime_error("ERROR: nvcompBatchedGzipCompressGetMaxOutputChunkSize() not successful");
  }

  // Allocate the compressed output batch on the GPU
  BatchData compressed_data(max_compressed_chunk_size, chunk_count, compression_alignment_reqs.output);

  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // CUDA events to measure compression time
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  // Allocate the temporary workspace required by the compressor
  size_t comp_temp_bytes;
  status = nvcompBatchedGzipCompressGetTempSizeAsync(
    chunk_count,
    chunk_size,
    compress_opts,
    &comp_temp_bytes,
    chunk_count * chunk_size
  );
  if (status != nvcompSuccess)
  {
    throw std::runtime_error("nvcompBatchedGzipCompressGetTempSizeAsync() failed.");
  }

  void *d_comp_temp;
  CUDA_CHECK(cudaMallocSafe(&d_comp_temp, comp_temp_bytes));

  nvcompStatus_t *d_status_ptrs;
  CUDA_CHECK(cudaMallocSafe(&d_status_ptrs, chunk_count * sizeof(nvcompStatus_t)));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto perform_compression = [&]() {
    if (nvcompBatchedGzipCompressAsync(
          input_data.ptrs(),
          input_data.sizes(),
          chunk_size,
          chunk_count,
          d_comp_temp,
          comp_temp_bytes,
          compressed_data.ptrs(),
          compressed_data.sizes(),
          compress_opts,
          d_status_ptrs,
          stream
        ) != nvcompSuccess)
    {
      throw std::runtime_error("ERROR: nvcompBatchedGzipCompressAsync() not successful");
    }
  };

  // Run warm-up compression
  for (size_t iter = 0; iter < warmup_iteration_count; ++iter)
  {
    perform_compression();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Validate the GPU-compressed output by decompressing it on the CPU with zlib
  // and comparing against the original input.
  BatchDataCPU compressed_data_cpu(
    compressed_data.ptrs(),
    compressed_data.sizes(),
    compressed_data.data(),
    compressed_data.size(),
    true
  );

  auto gzip_inflate = [](const uint8_t *src, size_t src_bytes, std::vector<uint8_t> &dst) -> size_t {
    z_stream zs{};
    // 15 | 16 - expect a gzip header
    if (inflateInit2(&zs, 15 | 16) != Z_OK)
    {
      throw std::runtime_error("Call to inflateInit2 failed");
    }
    zs.next_in = const_cast<Bytef *>(reinterpret_cast<const Bytef *>(src));
    zs.avail_in = static_cast<uInt>(src_bytes);
    zs.next_out = reinterpret_cast<Bytef *>(dst.data());
    zs.avail_out = static_cast<uInt>(dst.size());
    int ret = inflate(&zs, Z_FINISH);
    const size_t produced = zs.total_out;
    inflateEnd(&zs);
    if (ret != Z_STREAM_END)
    {
      throw std::runtime_error("Gzip decompression failed: " + std::to_string(ret));
    }
    return produced;
  };

  std::vector<uint8_t> decompressed_chunk;
  for (size_t i = 0; i < chunk_count; ++i)
  {
    const size_t uncompressed_chunk_size = input_data_cpu.sizes()[i];
    decompressed_chunk.resize(uncompressed_chunk_size);
    const size_t produced = gzip_inflate(
      reinterpret_cast<const uint8_t *>(compressed_data_cpu.ptrs()[i]),
      compressed_data_cpu.sizes()[i],
      decompressed_chunk
    );
    if (produced != uncompressed_chunk_size ||
        std::memcmp(decompressed_chunk.data(), input_data_cpu.ptrs()[i], uncompressed_chunk_size) != 0)
    {
      throw std::runtime_error("Failed to validate compressed data");
    }
  }
  std::cout << "compression validated :)" << std::endl;

  // Compute compression ratio from the actual compressed sizes
  size_t comp_bytes = 0;
  for (size_t i = 0; i < chunk_count; ++i)
  {
    comp_bytes += compressed_data_cpu.sizes()[i];
  }
  std::cout << "comp_size: " << comp_bytes << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;

  // Re-run compression to get throughput
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (size_t iter = warmup_iteration_count; iter < total_iteration_count; ++iter)
  {
    perform_compression();
  }
  CUDA_CHECK(cudaEventRecord(end, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
  ms /= total_iteration_count - warmup_iteration_count;

  double compression_throughput = ((double)total_bytes / ms) * 1e-6;
  std::cout << "compression throughput (GB/s): " << compression_throughput << std::endl;

  CUDA_CHECK(cudaFree(d_comp_temp));
  CUDA_CHECK(cudaFree(d_status_ptrs));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char *argv[])
{
  std::vector<std::string> file_names;

  size_t warmup_iteration_count = 2;
  size_t total_iteration_count = 5;

  do
  {
    if (argc < 3)
    {
      break;
    }

    int i = 1;
    while (i < argc)
    {
      const char *current_argv = argv[i++];
      if (strcmp(current_argv, "-f") == 0)
      {
        // parse until next `-` argument
        while (i < argc && argv[i][0] != '-')
        {
          file_names.emplace_back(argv[i++]);
        }
      }
      else
      {
        std::cerr << "Unknown argument: " << current_argv << std::endl;
        return 1;
      }
    }
  } while (0);

  if (file_names.empty())
  {
    std::cerr << "Must specify at least one file via '-f <file>'." << std::endl;
    return 1;
  }

  try
  {
    auto data = multi_file(file_names);
    run_example(data, warmup_iteration_count, total_iteration_count);
  }
  catch (const std::exception &e)
  {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
