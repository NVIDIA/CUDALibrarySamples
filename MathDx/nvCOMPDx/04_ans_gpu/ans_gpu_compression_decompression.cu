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

#include <nvcompdx.hpp>
#include "../common/batch_data.hpp"

using namespace nvcompdx;

// This sample demonstrates the usage of the block-level device API for
// ANS GPU compression and decompression. The input data type can be
// altered using the `-t` argument. Supported options are `uint8` and
// `float16`.

// ANS compression kernel, using the preconfigured decompressor
// 1 block per chunk
template<typename compressor_type>
__global__ void comp_block_kernel(
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

  const unsigned int global_chunk_id = blockIdx.x;

  auto compressor = compressor_type();
  constexpr size_t tmp_size_block = compressor.tmp_size_group();
  extern __shared__ __align__(compressor.shmem_alignment()) uint8_t shared_comp_scratch_buffer[];
  assert(reinterpret_cast<uintptr_t>(shared_comp_scratch_buffer) % compressor.shmem_alignment() == 0);

  // Note: The entire block must call into execute(...).
  compressor.execute(
    uncomp_chunks[global_chunk_id],
    comp_chunks[global_chunk_id],
    uncomp_chunk_sizes[global_chunk_id],
    comp_chunk_sizes + global_chunk_id,
    shared_comp_scratch_buffer,
    tmp_buffer + (tmp_size_block > 0 ? tmp_size_block * global_chunk_id : 0));
}

// ANS decompression kernel, using the preconfigured decompressor
// 1 block per chunk
template<typename decompressor_type>
__global__ void decomp_block_kernel(
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

  const unsigned int global_chunk_id = blockIdx.x;

  auto decompressor = decompressor_type();
  extern __shared__ __align__(decompressor.shmem_alignment()) uint8_t shared_decomp_scratch_buffer[];
  assert(reinterpret_cast<uintptr_t>(shared_decomp_scratch_buffer) % decompressor.shmem_alignment() == 0);

  // Note: The entire block must call into execute(...).
  decompressor.execute(
    comp_chunks[global_chunk_id],
    uncomp_chunks[global_chunk_id],
    comp_chunk_sizes[global_chunk_id],
    decomp_chunk_sizes + global_chunk_id,
    shared_decomp_scratch_buffer,
    nullptr);
}

// Benchmark performance from the binary data file
template<datatype DT, unsigned int Arch>
static int ans_gpu_comp_gpu_decomp(const std::vector<std::vector<char>>& data,
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
    if constexpr (DT ==  datatype::float16) {
      if(part.size() % 2 != 0) {
        throw std::runtime_error("ERROR: with the selected data type (float16), all input file sizes must be divisible by 2");
      }
    }
  }

  std::cout << "----------" << std::endl;
  std::cout << "files: " << data.size() << std::endl;
  std::cout << "uncompressed (B): " << total_bytes << std::endl;

  // (De)Compression parameters
  // We are compressing 1 chunk per thread block
  constexpr size_t num_warps_per_chunk = 8;
  constexpr size_t num_chunks_per_block = 1;
  constexpr size_t num_warps_per_block = num_warps_per_chunk * num_chunks_per_block;
  constexpr unsigned int block_size = static_cast<unsigned int>(num_warps_per_block * 32);
  constexpr size_t chunk_size = 1 << 16; // [bytes]

  // Configure the GPU compressor
  using ans_compressor_type =
    decltype(Algorithm<algorithm::ans>() +
             DataType<DT>() +
             Direction<direction::compress>() +
             MaxUncompChunkSize<chunk_size>() +
             Block() +
             BlockWarp<num_warps_per_block, true>() +
             SM<Arch>());

  // Build up GPU data
  BatchData input_data(data, chunk_size, ans_compressor_type().input_alignment());
  const size_t batch_size = input_data.batch_size();
  std::cout << "chunks: " << batch_size << std::endl;

  // Global scratch buffer
  size_t comp_temp_bytes = ans_compressor_type().tmp_size_total(batch_size);
  uint8_t* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  // Shared scratch buffer
  const auto comp_shared_memory = ans_compressor_type().shmem_size_group();

  // Prepare compressed data buffer
  size_t max_out_bytes = ans_compressor_type().max_comp_chunk_size();
  BatchData compressed_data(max_out_bytes, batch_size, ans_compressor_type().output_alignment());

  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Runtime (de)compression parameters
  const auto block_count = static_cast<unsigned int>((batch_size + num_chunks_per_block - 1) / num_chunks_per_block);

  float ms = measure_ms(warmup_iteration_count, total_iteration_count, stream, [&]() {
    comp_block_kernel<ans_compressor_type><<<block_count, block_size, comp_shared_memory, stream>>>(
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
  using ans_decompressor_type =
    decltype(Algorithm<algorithm::ans>() +
             DataType<DT>() +
             Direction<direction::decompress>() +
             MaxUncompChunkSize<chunk_size>() +
             Block() +
             BlockWarp<num_warps_per_block, true>() +
             SM<Arch>());

  // Global scratch buffer
  // Note: ANS requires no global scratch buffer for decompression
  static_assert(ans_decompressor_type().tmp_size_group() == 0);

  // Shared scratch buffer
  const auto decomp_shared_memory = ans_decompressor_type().shmem_size_group();

  // Allocate and build up decompression batch on GPU
  BatchData decomp_data(chunk_size, batch_size, ans_decompressor_type().output_alignment());

  ms = measure_ms(warmup_iteration_count, total_iteration_count, stream, [&](){
    decomp_block_kernel<ans_decompressor_type><<<block_count, block_size, decomp_shared_memory, stream>>>(
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
  std::cerr << "Usage: ans_gpu_compression_decompression [OPTIONS]" << std::endl;
  std::cerr << "  -f <input file(s)>" << std::endl;
  std::cerr << "  -t {uint8 or float16}" << std::endl;
}

template<unsigned int Arch>
struct RunnerUint8 {
  template<typename... Args>
  static int run(Args&&... args)
  {
    return ans_gpu_comp_gpu_decomp<datatype::uint8, Arch>(std::forward<Args>(args)...);
  }
};

template<unsigned int Arch>
struct RunnerFloat16 {
  template<typename... Args>
  static int run(Args&&... args)
  {
    return ans_gpu_comp_gpu_decomp<datatype::float16, Arch>(std::forward<Args>(args)...);
  }
};

int main(int argc, char* argv[])
{
  std::vector<std::string> file_names;

  size_t warmup_iteration_count = 2;
  size_t total_iteration_count = 5;
  bool is_data_uint8 = true;

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
            is_data_uint8 = true;
        } else if(strcmp(data_type, "float16") == 0) {
            is_data_uint8 = false;
        } else {
            std::cerr << "Unknown data type selected (" << data_type << "). Select from the supported options: uint8, float16." << std::endl;
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

  if(is_data_uint8) {
    return run_with_current_arch<RunnerUint8>(data, warmup_iteration_count, total_iteration_count);
  }
  return run_with_current_arch<RunnerFloat16>(data, warmup_iteration_count, total_iteration_count);
}
