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


#include <iostream>
#include <random>
#include <cub/cub.cuh>

#include <nvcompdx.hpp>
#include "../common/batch_data.hpp"

using namespace nvcompdx;

// This sample demonstrates the usage of the block-level device API for
// ANS GPU compression, followed by fused decompression and reduction.

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

// ANS decompression + reduction kernel, using the preconfigured decompressor
// 1 block per chunk
template<typename decompressor_type, size_t max_uncomp_chunk_size, size_t block_size>
__global__ void decomp_block_fused_kernel(
  size_t batch_size,
  const void * const * comp_chunks,
  const size_t * comp_chunk_sizes,
  int * reduction_result) {
  // Note:
  // Given the (de)compressor expression has an SM<> operator,
  // it makes the fully-typed kernel only applicable on one targeted device architecture.
  // We need to signal to the compiler not to continue compiling this kernel whenever
  // the current compilation architecture is different from the one specified in
  // the SM<> operator.
  NVCOMPDX_SKIP_IF_NOT_APPLICABLE(decompressor_type);

  const unsigned int global_chunk_id = blockIdx.x;

  auto decompressor = decompressor_type();
  extern __shared__ __align__(cuda::std::max(decompressor.shmem_alignment(), alignof(int))) uint8_t shared_scratch_buffer[];
  assert(reinterpret_cast<uintptr_t>(shared_scratch_buffer) % decompressor.shmem_alignment() == 0);

  __shared__ size_t shared_output_size;
  __shared__ __align__(decompressor.output_alignment()) uint8_t shared_output_buffer[max_uncomp_chunk_size];
  assert(reinterpret_cast<uintptr_t>(shared_output_buffer) % decompressor.output_alignment() == 0);

  // Perform ANS decompression within the block -----------
  // Note: The entire block must call into execute(...).
  decompressor.execute(
    comp_chunks[global_chunk_id],
    shared_output_buffer,
    comp_chunk_sizes[global_chunk_id],
    &shared_output_size,
    shared_scratch_buffer,
    nullptr);

  __syncthreads();
  assert(shared_output_size == max_uncomp_chunk_size);

  // Perform a sum reduction within the grid --------------
  // Thread-level reduction
  int thread_sum = 0;
  for(int i = threadIdx.x; i < max_uncomp_chunk_size; i += block_size) {
    thread_sum += shared_output_buffer[i];
  }
  // Block-level reduction
  // Reuse shared memory allocated for decompression
  typedef cub::BlockReduce<int, block_size> BlockReduceT;
  auto temp_storage = reinterpret_cast<typename BlockReduceT::TempStorage*>(shared_scratch_buffer);
  int block_sum = BlockReduceT(*temp_storage).Sum(thread_sum);
  // Grid-level reduction
  if (threadIdx.x == 0) {
    atomicAdd(reduction_result, block_sum);
  }
}

template<unsigned int Arch>
static int ans_gpu_comp_gpu_decomp_and_reduce(const std::vector<int>& data,
                                              const int ground_truth,
                                              size_t warmup_iteration_count,
                                              size_t total_iteration_count)
{
  assert(!data.empty());
  if(warmup_iteration_count >= total_iteration_count) {
    throw std::runtime_error("ERROR: the total iteration count must be greater than the warmup iteration count");
  }

  size_t total_bytes = data.size() * sizeof(int);

  std::cout << "----------" << std::endl;
  std::cout << "integers: " << data.size() << std::endl;
  std::cout << "uncompressed (B): " << total_bytes << std::endl;

  // Compile-time (de)compression parameters
  // We are compressing 1 chunk per thread block
  constexpr size_t num_warps_per_chunk = 4;
  constexpr size_t num_chunks_per_block = 1;
  constexpr size_t num_warps_per_block = num_warps_per_chunk * num_chunks_per_block;
  constexpr unsigned int block_size = static_cast<unsigned int>(num_warps_per_block * 32);
  constexpr size_t chunk_size = 1 << 14; // [bytes]

  // Configure the GPU compressor
  using ans_compressor_type =
    decltype(Algorithm<algorithm::ans>() +
             DataType<datatype::uint8>() +
             Direction<direction::compress>() +
             MaxUncompChunkSize<chunk_size>() +
             Block() +
             BlockWarp<num_warps_per_chunk, true>() +
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

  // Runtime (De)compression parameters
  const auto block_count = static_cast<unsigned int>((batch_size + num_chunks_per_block - 1) / num_chunks_per_block);

  float ms = measure_ms(warmup_iteration_count, total_iteration_count, stream, [&](){
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

  // compute compression ratio
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
             DataType<datatype::uint8>() +
             Direction<direction::decompress>() +
             MaxUncompChunkSize<chunk_size>() +
             Block() +
             BlockWarp<num_warps_per_chunk, true>() +
             SM<Arch>());

  // Global scratch buffer
  // Note: ANS requires no global scratch buffer for decompression
  static_assert(ans_decompressor_type().tmp_size_group() == 0);

  // Shared scratch buffer
  const auto decomp_shared_memory = ans_decompressor_type().shmem_size_group();

  int* d_reduction_result;
  CUDA_CHECK(cudaMalloc(&d_reduction_result, sizeof(int)));

  ms = measure_ms(warmup_iteration_count, total_iteration_count, stream, [&]() {
    CUDA_CHECK(cudaMemsetAsync(d_reduction_result, 0, sizeof(int), stream));
    decomp_block_fused_kernel<ans_decompressor_type, chunk_size, block_size>
        <<<block_count, block_size, decomp_shared_memory, stream>>>(
      batch_size,
      compressed_data.chunk_ptrs(),
      compressed_data.chunk_sizes(),
      d_reduction_result
    );
    CUDA_CHECK(cudaGetLastError());
  });

  // Validate the reduction result against the ground truth data
  int h_reduction_result;
  CUDA_CHECK(cudaMemcpy(&h_reduction_result, d_reduction_result, sizeof(int), cudaMemcpyDefault));

  if (h_reduction_result != ground_truth) {
    throw std::runtime_error("Failed to validate decompressed data");
  } else {
    std::cout << "decompression validated :)" << std::endl;
  }

  double decompression_throughput = ((double)total_bytes / ms) * 1e-6;
  std::cout << "decompression throughput with reduction (GB/s): " << decompression_throughput
            << std::endl;

  CUDA_CHECK(cudaFree(d_comp_temp));
  CUDA_CHECK(cudaFree(d_reduction_result));
  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}

int generate_data(std::vector<int>& data) {
    // Generate 8GB of int data
    constexpr size_t data_count = 1 << 21;
    data.reserve(data_count);

    int seed = 20;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> data_dist(0, 255);

    int sum = 0;
    for(size_t i = 0; i < data_count; ++i) {
        int value = data_dist(gen);
        data.emplace_back(value);
        sum += value;
    }
    return sum;
}

template<unsigned int Arch>
struct Runner {
  template<typename... Args>
  static int run(Args&&... args)
  {
    return ans_gpu_comp_gpu_decomp_and_reduce<Arch>(std::forward<Args>(args)...);
  }
};

int main([[maybe_unused]] int argc,
         [[maybe_unused]] char* argv[])
{
  size_t warmup_iteration_count = 2;
  size_t total_iteration_count = 5;

  // Generate data
  std::vector<int> data;
  int ground_truth = generate_data(data);

  // Run the example of comp + fused(decomp + sum reduction)
  return run_with_current_arch<Runner>(data, ground_truth, warmup_iteration_count, total_iteration_count);
}