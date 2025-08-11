/*
 * Copyright (c) 2025 NVIDIA CORPORATION AND AFFILIATES. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the NVIDIA CORPORATION nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "zstd.h"
#include "nvcomp/zstd.h"
#include "BatchData.h"

static void run_example(const std::vector<std::vector<char>>& data,
                        int compression_level,
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
  static_assert(chunk_size <= ZSTD_BLOCKSIZE_MAX, "Chunk size must be less than the constant specified in the Zstandard library");

  // Build up input batch on CPU
  BatchDataCPU input_data_cpu(data, chunk_size);
  const size_t chunk_count = input_data_cpu.size();
  std::cout << "chunks: " << chunk_count << std::endl;

  // compression

  // Allocate and prepare output/compressed batch
  BatchDataCPU compressed_data_cpu(
      ZSTD_compressBound(chunk_size), chunk_count);

  // loop over chunks on the CPU, compressing each one
  const auto min_compression_level = ZSTD_minCLevel();
  const auto max_compression_level = ZSTD_maxCLevel();
  if(compression_level < min_compression_level || compression_level > max_compression_level) {
    throw std::runtime_error("Unsupported compression level: " + std::to_string(compression_level) + ". Supported range: " + std::to_string(min_compression_level) + " - " +  std::to_string(max_compression_level));
  }
  for (size_t i = 0; i < chunk_count; ++i) {
    size_t size = ZSTD_compress(compressed_data_cpu.ptrs()[i],
                                compressed_data_cpu.sizes()[i],
                                input_data_cpu.ptrs()[i],
                                input_data_cpu.sizes()[i],
                                compression_level);
    if (ZSTD_isError(size)) {
      throw std::runtime_error(
          "Zstandard CPU failed to compress chunk " + std::to_string(i) + ". Error code: " + std::to_string(size) + ", Message: " + ZSTD_getErrorName(size));
    }
    compressed_data_cpu.sizes()[i] = size;
  }

  // compute compression ratio
  size_t comp_bytes = std::accumulate(compressed_data_cpu.sizes(), compressed_data_cpu.sizes() + chunk_count, size_t(0));

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;

  // Decompression options
  nvcompBatchedZstdDecompressOpts_t decompress_opts = nvcompBatchedZstdDecompressDefaultOpts;

  // Query decompression alignment requirements
  nvcompAlignmentRequirements_t decompression_alignment_reqs;
  nvcompStatus_t status = nvcompBatchedZstdDecompressGetRequiredAlignments(
    decompress_opts,
    &decompression_alignment_reqs);
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: nvcompBatchedZstdDecompressGetRequiredAlignments() not successful");
  }

  // Copy compressed data to GPU
  BatchData compressed_data(compressed_data_cpu, true, decompression_alignment_reqs.input);

  // Allocate and build up decompression batch on GPU
  BatchData decomp_data(input_data_cpu, false, decompression_alignment_reqs.output);

  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Allocate necessary buffers
  nvcompStatus_t* d_status_ptrs;
  CUDA_CHECK(cudaMalloc(&d_status_ptrs, chunk_count * sizeof(nvcompStatus_t)));

  size_t* d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, chunk_count * sizeof(size_t)));

  // CUDA events to measure decompression time
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  // Zstandard GPU decompression
  // Determine scratch space needed asynchronously
  size_t decomp_temp_bytes_async;
  status = nvcompBatchedZstdDecompressGetTempSizeAsync(
      chunk_count,
      chunk_size,
      decompress_opts,
      &decomp_temp_bytes_async,
      chunk_count * chunk_size);
  if (status != nvcompSuccess) {
    throw std::runtime_error("nvcompBatchedZstdDecompressGetTempSizeAsync() failed.");
  }

  // Determine scratch space needed synchronously
  size_t decomp_temp_bytes_sync;
  status = nvcompBatchedZstdDecompressGetTempSizeSync(
      compressed_data.ptrs(),
      compressed_data.sizes(),
      chunk_count,
      chunk_size,
      &decomp_temp_bytes_sync,
      chunk_size * chunk_count,
      decompress_opts,
      d_status_ptrs,
      stream);
  if (status != nvcompSuccess) {
    throw std::runtime_error("nvcompBatchedZstdDecompressGetTempSizeSync() failed.");
  }
  size_t decomp_temp_bytes = std::min(decomp_temp_bytes_sync, decomp_temp_bytes_async);

  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto perform_decompression = [&]() {
    if (nvcompBatchedZstdDecompressAsync(
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
      throw std::runtime_error("ERROR: nvcompBatchedZstdDecompressAsync() not successful");
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
  int compression_level = 6;

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
      } else if (strcmp(current_argv, "-l") == 0) {
        if(i >= argc) {
          std::cerr << "Missing value for argument '-l <compression level>'" << std::endl;
          return 1;
        }
        compression_level = atoi(argv[i++]);
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

  run_example(data, compression_level, warmup_iteration_count, total_iteration_count);

  return 0;
}
