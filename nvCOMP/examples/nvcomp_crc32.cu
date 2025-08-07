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

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nvcomp/crc32.h>

#include "BatchData.h"

// Forward declarations of helper functions.
static uint32_t reverse(uint32_t x);
static uint32_t cpu_crc32(const nvcompCRC32Spec_t& spec, size_t n, const void *m_);

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
  std::cout << "total size (B): " << total_bytes << std::endl;

  const size_t chunk_size = 1 << 16;

  // For consistency with compression examples, the input data set is split into
  // chunks and the (PKZIP) CRC32 value of each chunk is computed.
  BatchDataCPU input_data_cpu(data, chunk_size);
  const size_t chunk_count = input_data_cpu.size();
  std::cout << "chunks: " << chunk_count << std::endl;
  
  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // CUDA events to measure calculation time
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  // Copy the input data to the GPU.
  BatchData input_data(input_data_cpu, /*copy_data=*/true, /*alignment=*/1);

  // Create CRC32 output buffer.
  // Note: the Thrust device vector header is included in BatchData.h in a way
  // that works around a known issue with MSVC debug iterators.
  thrust::device_vector<uint32_t> crc32_values(chunk_count);

  // Heuristically determine the optimal kernel configuration.
  nvcompCRC32KernelConf_t kernel_conf{};
  if (nvcompBatchedCRC32GetHeuristicConf(
        nvcompCRC32IgnoredInputChunkBytes,
        chunk_count,
        &kernel_conf,
        chunk_size,
        stream) != nvcompSuccess) {
    throw std::runtime_error("ERROR: nvcompBatchedCRC32GetHeuristicConf() not successful");
  }
  
  nvcompBatchedCRC32Opts_t opts{nvcompCRC32, kernel_conf, {}};

  auto calc_crc32 = [&]() {
    if (nvcompBatchedCRC32Async(
          input_data.ptrs(),
          input_data.sizes(),
          chunk_count,
          crc32_values.data().get(),
          opts,
          nvcompCRC32OnlySegment,
          /*device_statuses=*/nullptr,
          stream) != nvcompSuccess) {
      throw std::runtime_error("ERROR: nvcompBatchedCRC32Async() not successful");
    }
  };

  // Run warm-up CRC32 computation
  for (size_t iter = 0; iter < warmup_iteration_count; ++iter) {
    calc_crc32();
  }

  // Re-run CRC32 computation to get throughput
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (size_t iter = warmup_iteration_count; iter < total_iteration_count; ++iter) {
    calc_crc32();
  }
  CUDA_CHECK(cudaEventRecord(end, stream));

  // Compute reference CRC32 values on the CPU.
  std::vector<uint32_t> ref_crc32_values(chunk_count);
  for (size_t i = 0; i < chunk_count; ++i) {
    ref_crc32_values[i] = cpu_crc32(nvcompCRC32, input_data_cpu.sizes()[i], input_data_cpu.ptrs()[i]);
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Validate CRC32 values computed on the GPU against reference values.
  if (crc32_values != ref_crc32_values) {
    throw std::runtime_error("Failed to validate computed CRC32 values");
  } else {
    std::cout << "CRC32 values validated :)" << std::endl;
  }

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
  ms /= total_iteration_count - warmup_iteration_count;

  double crc32_throughput = ((double)total_bytes / ms) * 1e-6;
  std::cout << "CRC32 throughput (GB/s): " << crc32_throughput
            << std::endl;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

static uint32_t reverse(uint32_t x)
{
  x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
  x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
  x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
  x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));

  return((x >> 16) | (x << 16));
}

static uint32_t cpu_crc32(const nvcompCRC32Spec_t& spec, size_t n, const void *m_)
{
  const unsigned char *m = static_cast<const unsigned char *>(m_);
  uint32_t crc = spec.init;

  while (n--) {
      crc ^= spec.ref_in ? reverse(*m++) : (*m++ << 24);
      for (int i = 0; i < 8; i++) {
          crc = (crc << 1) ^ ((crc & 0x80000000) ? spec.poly : 0);
      }
  }

  if (spec.ref_out) {
      crc = reverse(crc);
  }

  return crc ^ spec.xorout;
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
          while (i < argc) {
            file_names.emplace_back(argv[i++]);
          }
      } else {
        std::cerr << "Unknown argument: " << current_argv << std::endl;
        return 1;
      }
    }
  } while (0);

  if (file_names.empty()) {
   std::cerr << "Must specify at least one file via '-f <file>'" << std::endl;
   return 1;
  }

  auto data = multi_file(file_names);

  run_example(data, warmup_iteration_count, total_iteration_count);

  return 0;
}
