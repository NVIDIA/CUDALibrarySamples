/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <lz4.h>
#include <lz4hc.h>

#include <nvcompdx.hpp>
#include "../common/batch_data.hpp"

using namespace nvcompdx;

// This sample demonstrates the usage of the warp-level device API for
// LZ4 GPU decompression. The compression happens through the host-side
// lz4 CPU library.

// LZ4 decompression kernel, using the preconfigured decompressor
// 1 warp per chunk, but multiple chunks per thread block
template <typename decompressor_type>
__global__ void decomp_warp_kernel(
    size_t batch_size,
    const void * const * comp_chunks,
    void * const * uncomp_chunks,
    const size_t * comp_chunk_sizes,
    size_t * decomp_chunk_sizes) {
  // Note:
  // Given the (de)compressor expression has an SM<> operator,
  // it makes the fully-typed kernel only applicable on one targeted device architecture.
  // We need to signal to the compiler not to continue compiling this kernel whenever
  // the current compilation architecture is different from the one specified in
  // the SM<> operator.
  NVCOMPDX_SKIP_IF_NOT_APPLICABLE(decompressor_type);

  const unsigned int global_chunk_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const unsigned int local_chunk_id = threadIdx.x / 32;
  if(global_chunk_id >= batch_size) {
    return;
  }

  auto decompressor = decompressor_type();
  constexpr auto shmem_size_warp = decompressor.shmem_size_group();
  extern __shared__ __align__(decompressor.shmem_alignment()) uint8_t shared_decomp_scratch_buffer[];
  assert(reinterpret_cast<uintptr_t>(shared_decomp_scratch_buffer) % decompressor.shmem_alignment() == 0);

  decompressor.execute(
    comp_chunks[global_chunk_id],
    uncomp_chunks[global_chunk_id],
    comp_chunk_sizes[global_chunk_id],
    decomp_chunk_sizes + global_chunk_id,
    shared_decomp_scratch_buffer + shmem_size_warp * local_chunk_id,
    nullptr);
}

// Benchmark performance from the binary data file
template<unsigned int Arch>
static int lz4_cpu_comp_gpu_decomp(const std::vector<std::vector<char>>& data,
                                   size_t warmup_iteration_count,
                                   size_t total_iteration_count)
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

  // Compile-time (de)compression parameters
  // We are decompressing 4 chunks per thread block
  constexpr size_t num_warps_per_chunk = 1;
  constexpr size_t num_chunks_per_block = 4;
  constexpr size_t num_warps_per_block = num_warps_per_chunk * num_chunks_per_block;
  constexpr unsigned block_size = static_cast<unsigned int>(num_warps_per_block * 32);
  constexpr size_t chunk_size = 1 << 16; // [bytes]

  // Build up input batch on CPU
  BatchDataCPU input_data_cpu(data, chunk_size);
  const size_t batch_size = input_data_cpu.batch_size();
  std::cout << "chunks: " << batch_size << std::endl;

  // Allocate and prepare output/compressed batch
  BatchDataCPU compressed_data_cpu(
      LZ4_compressBound(chunk_size), batch_size);

  // Compressing on the CPU
  // Loop over chunks on the CPU, compressing each one one by one
  for (size_t i = 0; i < batch_size; ++i) {
    // Could use LZ4_compress_default or LZ4_compress_fast instead
    const int size = LZ4_compress_HC(
        static_cast<const char*>(input_data_cpu.chunk_ptrs()[i]),
        static_cast<char*>(compressed_data_cpu.chunk_ptrs()[i]),
        static_cast<int>(input_data_cpu.chunk_sizes()[i]),
        static_cast<int>(compressed_data_cpu.chunk_sizes()[i]),
        12 /* compression level */);
    if (size == 0) {
      throw std::runtime_error(
          "LZ4 CPU failed to compress chunk " + std::to_string(i) + ".");
    }

    // Set the actual compressed size
    compressed_data_cpu.chunk_sizes()[i] = static_cast<size_t>(size);
  }

  // Compute compression ratio
  size_t* compressed_sizes_host = compressed_data_cpu.chunk_sizes();
  size_t comp_bytes = std::accumulate(compressed_sizes_host, compressed_sizes_host + batch_size, size_t(0));

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;

  // Configure the GPU decompressor
  using lz4_decompressor_type =
    decltype(Algorithm<algorithm::lz4>() +
             DataType<datatype::uint8>() +
             Direction<direction::decompress>() +
             MaxUncompChunkSize<chunk_size>() +
             Warp() +
             SM<Arch>());

  // Runtime (de)compression parameters
  const auto block_count = static_cast<unsigned int>((batch_size + num_chunks_per_block - 1) / num_chunks_per_block);

  // Global memory buffer
  // Note: lz4 decompression requires no global scratch buffer
  static_assert(lz4_decompressor_type().tmp_size_group() == 0);

  // Shared memory buffer
  const auto decomp_shared_memory = lz4_decompressor_type().shmem_size_group() * num_chunks_per_block;

  // Copy compressed data to GPU
  BatchData compressed_data(compressed_data_cpu, true, lz4_decompressor_type().input_alignment());

  // Allocate and build up decompression batch on GPU
  BatchData decomp_data(input_data_cpu, false, lz4_decompressor_type().output_alignment());

  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  float ms = measure_ms(warmup_iteration_count, total_iteration_count, stream, [&](){
    decomp_warp_kernel<lz4_decompressor_type><<<block_count, block_size, decomp_shared_memory, stream>>>(
      batch_size,
      compressed_data.chunk_ptrs(),
      decomp_data.chunk_ptrs(),
      compressed_data.chunk_sizes(),
      decomp_data.chunk_sizes()
    );
    CUDA_CHECK(cudaGetLastError());
  });

  // Validate decompressed data against input
  if (decomp_data != input_data_cpu) {
    throw std::runtime_error("Failed to validate decompressed data");
  } else {
    std::cout << "decompression validated :)" << std::endl;
  }

  double decompression_throughput = ((double)total_bytes / ms) * 1e-6;
  std::cout << "decompression throughput (GB/s): " << decompression_throughput
            << std::endl;

  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}

void print_usage()
{
  std::cerr << std::endl;
  std::cerr << "Usage: lz4_cpu_compression [OPTIONS]" << std::endl;
  std::cerr << "  -f <input file(s)>" << std::endl;
}

template<unsigned int Arch>
struct Runner {
  template<typename... Args>
  static int run(Args&&... args)
  {
    return lz4_cpu_comp_gpu_decomp<Arch>(std::forward<Args>(args)...);
  }
};

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
        // Parse until next `-` argument
        while (i < argc && argv[i][0] != '-') {
          file_names.emplace_back(argv[i++]);
        }
      } else {
        std::cerr << "Unknown argument: " << current_argv << std::endl;
        print_usage();
        return 1;
      }
    }
  } while (0);

  if (file_names.empty()) {
    std::cerr << "Must specify at least one file via '-f <file>'." << std::endl;
    print_usage();
    return 1;
  }

  auto data = multi_file(file_names);

  return run_with_current_arch<Runner>(data, warmup_iteration_count, total_iteration_count);
}