/*
 * Copyright (c) 2020-2025 NVIDIA CORPORATION AND AFFILIATES. All rights reserved.
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

#include "zlib.h"
#include "libdeflate.h"
#include "nvcomp/deflate.h"
#include "BatchData.h"


static void run_example(const std::vector<std::vector<char>>& data,
                        int algo, int compression_level,
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

  // Allocate and prepare output/compressed batch
  BatchDataCPU compressed_data_cpu(
      chunk_size, chunk_count);

  // loop over chunks on the CPU, compressing each one
  for (size_t i = 0; i < chunk_count; ++i) {
    size_t actual_len = 0;
    if (algo == 0) { //libdeflate
      struct libdeflate_compressor *compressor;
      compressor = libdeflate_alloc_compressor(compression_level);
      if(compressor == NULL) {
        throw std::runtime_error("Unsupported compression level: " + std::to_string(compression_level));
      }
      size_t len = libdeflate_deflate_compress(compressor, input_data_cpu.ptrs()[i],
                            input_data_cpu.sizes()[i], compressed_data_cpu.ptrs()[i], compressed_data_cpu.sizes()[i]);
      if (len == 0) {
        throw std::runtime_error(
            "libdeflate_deflate_compress failed to compress chunk " + std::to_string(i) + ".");
      }
      actual_len = len;
    } else if (algo == 1) { //zlib::compress2
     uLongf len = static_cast<uLongf>(input_data_cpu.sizes()[i]);
     if(compression_level < Z_NO_COMPRESSION || compression_level > Z_BEST_COMPRESSION) {
        throw std::runtime_error("Unsupported compression level: " + std::to_string(compression_level) + ". Supported range: " + std::to_string(Z_NO_COMPRESSION) + " - " +  std::to_string(Z_BEST_COMPRESSION));
     }
     int ret = compress2((uint8_t *)compressed_data_cpu.ptrs()[i], &len, (const Bytef *) input_data_cpu.ptrs()[i], static_cast<uLong>(input_data_cpu.sizes()[i]), compression_level);
     if (ret != Z_OK) {
         throw std::runtime_error("ZLIB compress() failed " + std::to_string(ret));
     }
     if (len >= 6) {
       memmove((uint8_t*)compressed_data_cpu.ptrs()[i], (uint8_t*)compressed_data_cpu.ptrs()[i] + 2, len - 6);
       len -= 6;
     }
     actual_len = static_cast<size_t>(len);
    } else if (algo == 2) { //zlib::deflate
     z_stream zs;
     zs.zalloc = NULL; zs.zfree = NULL;
     zs.msg = NULL;
     zs.next_in  = (Bytef *)input_data_cpu.ptrs()[i];
     zs.avail_in = static_cast<uInt>(input_data_cpu.sizes()[i]);
     zs.next_out = (Bytef *)compressed_data_cpu.ptrs()[i];
     zs.avail_out = static_cast<uInt>(input_data_cpu.sizes()[i]);
     int strategy = Z_DEFAULT_STRATEGY;
     // -15 to disable zlib header/footer (raw deflate)
     if(compression_level < Z_NO_COMPRESSION || compression_level > Z_BEST_COMPRESSION) {
        throw std::runtime_error("Unsupported compression level: " + std::to_string(compression_level) + ". Supported range: " + std::to_string(Z_NO_COMPRESSION) + " - " +  std::to_string(Z_BEST_COMPRESSION));
     }
     int ret = deflateInit2(&zs, compression_level, Z_DEFLATED, -15, 8, strategy);
     if (ret!=Z_OK) {
         throw std::runtime_error("Call to deflateInit2 failed: " + std::to_string(ret));
     }
     if ((ret = deflate(&zs, Z_FINISH)) != Z_STREAM_END) {
         throw std::runtime_error("Deflate operation failed: " + std::to_string(ret));
     }
     if ((ret = deflateEnd(&zs)) != Z_OK) {
         throw std::runtime_error("Call to deflateEnd failed: " + std::to_string(ret));
     }
     actual_len = static_cast<size_t>(zs.total_out);
    } else {
      throw std::runtime_error("Unsupported algorithm: " + std::to_string(algo));
    }
   // set the actual compressed size
   compressed_data_cpu.sizes()[i] = actual_len;
  }

  // compute compression ratio
  size_t* compressed_sizes_host = compressed_data_cpu.sizes();
  size_t comp_bytes = 0;
  for (size_t i = 0; i < chunk_count; ++i)
    comp_bytes += compressed_sizes_host[i];

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;

  // Decompression options
  nvcompBatchedDeflateDecompressOpts_t decompress_opts = nvcompBatchedDeflateDecompressDefaultOpts;

  // Query decompression alignment requirements
  nvcompAlignmentRequirements_t decompression_alignment_reqs;
  nvcompStatus_t status = nvcompBatchedDeflateDecompressGetRequiredAlignments(
    decompress_opts,
    &decompression_alignment_reqs);
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: nvcompBatchedDeflateDecompressGetRequiredAlignments() not successful");
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

  // deflate GPU decompression
  size_t decomp_temp_bytes;
  status = nvcompBatchedDeflateDecompressGetTempSizeAsync(
      chunk_count,
      chunk_size,
      decompress_opts,
      &decomp_temp_bytes,
      chunk_count * chunk_size);
  if (status != nvcompSuccess) {
    throw std::runtime_error("nvcompBatchedDeflateDecompressGetTempSizeAsync() failed.");
  }

  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

  size_t* d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, chunk_count * sizeof(size_t)));

  nvcompStatus_t* d_status_ptrs;
  CUDA_CHECK(cudaMalloc(&d_status_ptrs, chunk_count * sizeof(nvcompStatus_t)));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto perform_decompression = [&]() {
    if (nvcompBatchedDeflateDecompressAsync(
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
      throw std::runtime_error("ERROR: nvcompBatchedDeflateDecompressAsync() not successful");
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

  int algo = -1;
  int compression_level = 6;
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
      } else if (strcmp(current_argv, "-l") == 0) {
        if(i >= argc) {
          std::cerr << "Missing value for argument '-l <compression level>'" << std::endl;
          return 1;
        }
        compression_level = atoi(argv[i++]);
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
  } else if (algo < 0 || algo > 2) {
    std::cerr << "Must choose an algorithm via '-a <algo>'. '<algo>' can be 0, 1, or 2. (0 libdeflate, 1 zlib_compress2, 2 zlib_deflate)" << std::endl;
    return 1;
  } else if (file_names.empty()) {
   std::cerr << "Must specify at least one file via '-f <file>'" << std::endl;
   return 1;
  }

  auto data = multi_file(file_names);

  run_example(data, algo, compression_level, warmup_iteration_count, total_iteration_count);

  return 0;
}
