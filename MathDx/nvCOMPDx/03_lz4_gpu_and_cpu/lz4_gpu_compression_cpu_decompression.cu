// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <lz4.h>
#include <lz4hc.h>

#include <nvcompdx.hpp>
#include "../common/batch_data.hpp"

using namespace nvcompdx;

// This sample demonstrates the usage of the warp-level device API for
// LZ4 GPU compression. The decompression happens through the host-side
// lz4 CPU library.

// LZ4 compression kernel, using the preconfigured compressor
// 1 warp per chunk, but multiple chunks per thread block
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
  const unsigned int local_chunk_id = threadIdx.x / 32;
  if(global_chunk_id >= batch_size) {
    return;
  }

  auto compressor = compressor_type();
  constexpr size_t shmem_size_warp = compressor.shmem_size_group();
  constexpr size_t tmp_size_warp = compressor.tmp_size_group();
  extern __shared__ __align__(compressor.shmem_alignment()) uint8_t shared_comp_scratch_buffer[];
  assert(reinterpret_cast<uintptr_t>(shared_comp_scratch_buffer) % compressor.shmem_alignment() == 0);

  compressor.execute(
    uncomp_chunks[global_chunk_id],
    comp_chunks[global_chunk_id],
    uncomp_chunk_sizes[global_chunk_id],
    comp_chunk_sizes + global_chunk_id,
    shared_comp_scratch_buffer + shmem_size_warp * local_chunk_id,
    tmp_buffer + tmp_size_warp * global_chunk_id);
}

BatchDataCPU convert_batch(const BatchData& batch_data, bool copy_data)
{
  BatchDataCPU batch_data_cpu(
      batch_data.chunk_ptrs(),
      batch_data.chunk_sizes(),
      batch_data.batch_size(),
      copy_data);
  return batch_data_cpu;
}

// Benchmark performance from the binary data file
template<unsigned int Arch>
static int lz4_gpu_comp_cpu_decomp(const std::vector<std::vector<char>>& data,
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
  // We are compressing 4 chunks per thread block
  constexpr size_t num_warps_per_chunk = 1;
  constexpr size_t num_chunks_per_block = 4;
  constexpr size_t num_warps_per_block = num_warps_per_chunk * num_chunks_per_block;
  constexpr unsigned int block_size = static_cast<unsigned int>(num_warps_per_block * 32);
  constexpr size_t chunk_size = 1 << 16; // [bytes]

  // Configure the GPU compressor
  using lz4_compressor_type =
    decltype(Algorithm<algorithm::lz4>() +
             DataType<datatype::uint8>() +
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
  const auto comp_shared_memory = lz4_compressor_type().shmem_size_group() * num_chunks_per_block;

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

  // Allocate and prepare output/compressed batch
  BatchDataCPU compressed_data_cpu = convert_batch(compressed_data, true);
  BatchDataCPU decompressed_data_cpu = convert_batch(input_data, false);

  // Loop over chunks on the CPU, decompressing each one
  for (size_t i = 0; i < batch_size; ++i) {
    const int size = LZ4_decompress_safe(
        static_cast<const char*>(compressed_data_cpu.chunk_ptrs()[i]),
        static_cast<char*>(decompressed_data_cpu.chunk_ptrs()[i]),
        static_cast<int>(compressed_data_cpu.chunk_sizes()[i]),
        static_cast<int>(decompressed_data_cpu.chunk_sizes()[i]));
    if (size == 0) {
      throw std::runtime_error(
          "LZ4 CPU failed to decompress chunk " + std::to_string(i) + ".");
    }
  }
  // Validate decompressed data against input
  if (input_data != decompressed_data_cpu) {
    throw std::runtime_error("Failed to validate CPU decompressed data");
  } else {
    std::cout << "CPU decompression validated :)" << std::endl;
  }

  CUDA_CHECK(cudaFree(d_comp_temp));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

void print_usage()
{
  std::cerr << std::endl;
  std::cerr << "Usage: lz4_cpu_decompression [OPTIONS]" << std::endl;
  std::cerr << "  -f <input file(s)>" << std::endl;
}

template<unsigned int Arch>
struct Runner {
  template<typename... Args>
  static int run(Args&&... args)
  {
    return lz4_gpu_comp_cpu_decomp<Arch>(std::forward<Args>(args)...);
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
