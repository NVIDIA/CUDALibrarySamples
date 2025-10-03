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


#include <nvcompdx.hpp>
#include "../common/batch_data.hpp"

using namespace nvcompdx;

// This sample demonstrates the usage of the warp-level device API for
// LZ4 GPU compression and decompression. The input data type can be
// altered using the `-t` argument. Supported options are `uint8`,
// `uint16`, and `uint32`.

// LZ4 compression kernel, using the preconfigured decompressor
// 1 warp per chunk, but multiple chunks per block
template<typename compressor_type>
__global__ void comp_warp_kernel(
  size_t batch_size,
  const void * const * uncomp_chunks,
  const size_t * uncomp_chunk_sizes,
  void * const * comp_chunks,
  size_t * comp_chunk_sizes,
  uint8_t * tmp_buffer) {
  // Note:
  // Given the (de)compressor expression has an SM<> operator,
  // it makes the fully-typed kernel only applicable on one targeted device architecture.
  // We need to signal to the compiler not to continue compiling this kernel whenever
  // the current compilation architecture is different from the one specified in
  // the SM<> operator.
  NVCOMPDX_SKIP_IF_NOT_APPLICABLE(compressor_type);

  const unsigned int global_chunk_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if(global_chunk_id >= batch_size) {
    return;
  }

  auto compressor = compressor_type();
  constexpr size_t tmp_size_warp = compressor.tmp_size_group();

  // Note: Each warp calls into execute(...) processing different chunks
  compressor.execute(
    uncomp_chunks[global_chunk_id],
    comp_chunks[global_chunk_id],
    uncomp_chunk_sizes[global_chunk_id],
    comp_chunk_sizes + global_chunk_id,
    nullptr,
    tmp_buffer + tmp_size_warp * global_chunk_id);
}

// LZ4 decompression kernel, using the preconfigured decompressor
// 1 warp per chunk, but multiple chunks per block
template<typename decompressor_type>
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
  constexpr size_t shmem_size_warp = decompressor.shmem_size_group();
  extern __shared__ __align__(decompressor.shmem_alignment()) uint8_t shared_decomp_scratch_buffer[];
  assert(reinterpret_cast<uintptr_t>(shared_decomp_scratch_buffer) % decompressor.shmem_alignment() == 0);

  // Note: Each warp calls into execute(...) processing different chunks
  decompressor.execute(
    comp_chunks[global_chunk_id],
    uncomp_chunks[global_chunk_id],
    comp_chunk_sizes[global_chunk_id],
    decomp_chunk_sizes + global_chunk_id,
    shared_decomp_scratch_buffer + shmem_size_warp * local_chunk_id,
    nullptr);
}

// Benchmark performance from the binary data file
template<datatype DT, unsigned int Arch>
static int lz4_gpu_comp_gpu_decomp(const std::vector<std::vector<char>>& data,
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
    if constexpr (DT == datatype::uint16) {
      if(part.size() % 2 != 0) {
        throw std::runtime_error("ERROR: with the selected data type (uint16), all input file sizes must be divisible by 2");
      }
    } else if constexpr (DT == datatype::uint32) {
      if(part.size() % 4 != 0) {
        throw std::runtime_error("ERROR: with the selected data type (uint32), all input file sizes must be divisible by 4");
      }
    }
  }

  std::cout << "----------" << std::endl;
  std::cout << "files: " << data.size() << std::endl;
  std::cout << "uncompressed (B): " << total_bytes << std::endl;

  // (De)Compression parameters
  // We are compressing 1 chunk per thread block
  constexpr size_t num_warps_per_chunk = 1;
  constexpr size_t num_chunks_per_block = 4;
  constexpr size_t num_warps_per_block = num_warps_per_chunk * num_chunks_per_block;
  constexpr unsigned int block_size = static_cast<unsigned int>(num_warps_per_block * 32);
  constexpr size_t chunk_size = 1 << 15; // [bytes]

  // Configure the GPU compressor
  using lz4_compressor_type =
    decltype(Algorithm<algorithm::lz4>() +
             DataType<DT>() +
             Direction<direction::compress>() +
             MaxUncompChunkSize<chunk_size>() +
             Warp() +
             SM<Arch>());

  // Build up GPU data
  BatchData input_data(data, chunk_size, lz4_compressor_type().input_alignment());
  const size_t batch_size = input_data.batch_size();
  std::cout << "chunks: " << batch_size << std::endl;

  // Global scratch buffer
  size_t comp_temp_bytes = lz4_compressor_type().tmp_size_total(batch_size);
  uint8_t* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  // Shared scratch buffer
  constexpr auto comp_shared_memory = lz4_compressor_type().shmem_size_group() * num_chunks_per_block;
  // LZ4 compression requires no shared memory
  static_assert(comp_shared_memory == 0);

  // Prepare compressed data buffer
  size_t max_out_bytes = lz4_compressor_type().max_comp_chunk_size();
  BatchData compressed_data(max_out_bytes, batch_size, lz4_compressor_type().output_alignment());

  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Runtime (de)compression parameters
  const auto block_count = static_cast<unsigned int>((batch_size + num_chunks_per_block - 1) / num_chunks_per_block);

  float ms = measure_ms(warmup_iteration_count, total_iteration_count, stream, [&]() {
    comp_warp_kernel<lz4_compressor_type><<<block_count, block_size, comp_shared_memory, stream>>>(
      batch_size,
      input_data.chunk_ptrs(),
      input_data.chunk_sizes(),
      compressed_data.chunk_ptrs(),
      compressed_data.chunk_sizes(),
      d_comp_temp
    );
    CUDA_CHECK(cudaGetLastError());
  });

  // Compute compression ratio
  std::vector<size_t> compressed_sizes_host(batch_size);
  CUDA_CHECK(cudaMemcpy(
    compressed_sizes_host.data(),
    compressed_data.chunk_sizes(),
    batch_size * sizeof(size_t),
    cudaMemcpyDeviceToHost));

  size_t comp_bytes = std::accumulate(
        compressed_sizes_host.begin(),
        compressed_sizes_host.end(),
        size_t(0));

  std::cout << "comp_size: " << comp_bytes
        << ", compressed ratio: " << std::fixed << std::setprecision(2)
        << (double)total_bytes / comp_bytes << std::endl;
  std::cout << "compression throughput (GB/s): "
        << (double)total_bytes / (1.0e6 * ms) << std::endl;

  // Configure the GPU decompressor
  using lz4_decompressor_type =
    decltype(Algorithm<algorithm::lz4>() +
             DataType<DT>() +
             Direction<direction::decompress>() +
             MaxUncompChunkSize<chunk_size>() +
             Warp() +
             SM<Arch>());

  // Global scratch buffer
  // Note: LZ4 requires no global scratch buffer for decompression
  static_assert(lz4_decompressor_type().tmp_size_group() == 0);

  // Shared scratch buffer
  const auto decomp_shared_memory = lz4_decompressor_type().shmem_size_group() * num_chunks_per_block;

  // Allocate and build up decompression batch on GPU
  BatchData decomp_data(chunk_size, batch_size, lz4_decompressor_type().output_alignment());

  ms = measure_ms(warmup_iteration_count, total_iteration_count, stream, [&](){
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
  if (input_data != decomp_data) {
    throw std::runtime_error("Failed to validate decompressed data");
  } else {
    std::cout << "decompression validated :)" << std::endl;
  }

  double decompression_throughput = ((double)total_bytes / ms) * 1e-6;
  std::cout << "decompression throughput (GB/s): " << decompression_throughput
            << std::endl;

  CUDA_CHECK(cudaFree(d_comp_temp));
  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}

void print_usage()
{
  std::cerr << std::endl;
  std::cerr << "Usage: lz4_gpu_compression_decompression [OPTIONS]" << std::endl;
  std::cerr << "  -f <input file(s)>" << std::endl;
  std::cerr << "  -t {uint8, uint16, or uint32}" << std::endl;
}

template<unsigned int Arch>
struct RunnerUint8 {
  template<typename... Args>
  static int run(Args&&... args)
  {
    return lz4_gpu_comp_gpu_decomp<datatype::uint8, Arch>(std::forward<Args>(args)...);
  }
};

template<unsigned int Arch>
struct RunnerUint16 {
  template<typename... Args>
  static int run(Args&&... args)
  {
    return lz4_gpu_comp_gpu_decomp<datatype::uint16, Arch>(std::forward<Args>(args)...);
  }
};

template<unsigned int Arch>
struct RunnerUint32 {
  template<typename... Args>
  static int run(Args&&... args)
  {
    return lz4_gpu_comp_gpu_decomp<datatype::uint32, Arch>(std::forward<Args>(args)...);
  }
};

int main(int argc, char* argv[])
{
  std::vector<std::string> file_names;

  size_t warmup_iteration_count = 2;
  size_t total_iteration_count = 5;
  datatype selected_data_type = datatype::uint8;

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
      } else if(strcmp(current_argv, "-t") == 0) {
        if(i >= argc) {
            std::cerr << "Missing value for argument '-t <data type>'" << std::endl;
            print_usage();
            return 1;
        }
        const char* data_type = argv[i++];
        if(strcmp(data_type, "uint8") == 0) {
          selected_data_type = datatype::uint8;
        } else if(strcmp(data_type, "uint16") == 0) {
            selected_data_type = datatype::uint16;
        } else if(strcmp(data_type, "uint32") == 0) {
            selected_data_type = datatype::uint32;
        } else {
            std::cerr << "Unknown data type selected (" << data_type << "). Select from the supported options: uint8, uint16, or uint32." << std::endl;
            print_usage();
            return 1;
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

  switch(selected_data_type) {
    case datatype::uint8:
      return run_with_current_arch<RunnerUint8>(data, warmup_iteration_count, total_iteration_count);
    case datatype::uint16:
      return run_with_current_arch<RunnerUint16>(data, warmup_iteration_count, total_iteration_count);
    case datatype::uint32:
      return run_with_current_arch<RunnerUint32>(data, warmup_iteration_count, total_iteration_count);
    default:
      return -1;
  }
}