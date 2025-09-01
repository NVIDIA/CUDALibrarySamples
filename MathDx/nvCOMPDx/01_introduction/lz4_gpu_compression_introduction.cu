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

#include <cassert>
#include <iostream>
#include <algorithm>
#include <string>
#include <functional>

#include <nvcompdx.hpp>
#include "../common/util.hpp"

using namespace nvcompdx;


// This introductory sample demonstrates the usage of the warp-level device API
// for LZ4 GPU compression. The compressed buffer afterwards is written to disk.
// Note, however, that for optimal performance, one should perform multiple chunk
// compressions simultaneously. The example here only intends to showcase usage.

// LZ4 compression kernel, using the preconfigured compressor
// 1 warp per chunk
template<typename compressor_type>
__global__ void comp_warp_kernel(
  const void * const uncomp_chunk,
  const size_t uncomp_chunk_size,
  void * comp_chunk,
  size_t * comp_chunk_size,
  uint8_t * tmp_buffer) {
  // Note:
  // Given the (de)compressor expression has an SM<> operator,
  // it makes the fully-typed kernel only applicable on one targeted device architecture.
  // We need to signal to the compiler not to continue compiling this kernel whenever
  // the current compilation architecture is different from the one specified in
  // the SM<> operator.
  NVCOMPDX_SKIP_IF_NOT_APPLICABLE(compressor_type);

  auto compressor = compressor_type();
  constexpr size_t shmem_alignment = compressor.shmem_alignment();
  extern __shared__ __align__(shmem_alignment) uint8_t shared_comp_scratch_buffer[];
  assert(reinterpret_cast<uintptr_t>(shared_comp_scratch_buffer) % compressor.shmem_alignment() == 0);

  compressor.execute(
    uncomp_chunk,
    comp_chunk,
    uncomp_chunk_size,
    comp_chunk_size,
    shared_comp_scratch_buffer,
    tmp_buffer);
}

static constexpr size_t chunk_size = 1 << 16;

// Benchmark performance from the binary data file
template<unsigned int Arch>
int lz4_gpu_comp_introduction(const std::vector<char>& data,
                              std::vector<char>& compressed,
                              size_t warmup_iteration_count,
                              size_t total_iteration_count)
{
  assert(!data.empty());

  size_t total_bytes = data.size();
  size_t batch_size = 1; // a single chunk
  std::cout << "----------" << std::endl;
  std::cout << "uncompressed (B): " << total_bytes << std::endl;
  std::cout << "chunks: " << 1 << std::endl;

  // Configure the GPU compressor
  using lz4_compressor_type =
    decltype(Algorithm<algorithm::lz4>() +
             DataType<datatype::uint8>() +
             Direction<direction::compress>() +
             MaxUncompChunkSize<chunk_size>() +
             Warp() +
             SM<Arch>());

  // Allocate buffer for the input (uncompressed) data
  // Note: with cudaMalloc() the input alignment is implicitly met
  void* d_input_data;
  CUDA_CHECK(cudaMalloc(&d_input_data, total_bytes));
  CUDA_CHECK(cudaMemcpy(d_input_data, data.data(), total_bytes, cudaMemcpyHostToDevice));

  // Allocate buffer for the input/output sizes
  size_t* d_output_size;
  CUDA_CHECK(cudaMalloc(&d_output_size, sizeof(size_t)));

  // Compress on the GPU using device API
  // Allocate temporary scratch space
  uint8_t* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, lz4_compressor_type().tmp_size_total(1)));

  // Calculate the maximum compressed size
  const size_t max_comp_chunk_size = lz4_compressor_type().max_comp_chunk_size();

  // Allocate buffer for the output (compressed) data
  // Note: with cudaMalloc() the output alignment is implicitly met
  void* d_output_data;
  CUDA_CHECK(cudaMalloc(&d_output_data, max_comp_chunk_size));

  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Compression parameters
  // We are compressing 1 chunk per thread block
  const unsigned int block_size = 32; // 1 warp
  const unsigned int block_count = static_cast<unsigned int>(batch_size);
  const auto comp_shared_memory = lz4_compressor_type().shmem_size_group();

  float ms = measure_ms(warmup_iteration_count, total_iteration_count, stream, [&]() {
    comp_warp_kernel<lz4_compressor_type><<<block_count, block_size, comp_shared_memory, stream>>>(
      d_input_data,
      total_bytes,
      d_output_data,
      d_output_size,
      d_comp_temp
    );
    CUDA_CHECK(cudaGetLastError());
  });

  // Compute compression ratio
  size_t comp_bytes;
  CUDA_CHECK(cudaMemcpy(&comp_bytes, d_output_size, sizeof(size_t), cudaMemcpyDeviceToHost));
  assert(comp_bytes <= max_comp_chunk_size);

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;
  std::cout << "compression throughput (GB/s): "
            << (double)total_bytes / (1.0e6 * ms) << std::endl;

  // Copy data back to host for write out
  compressed.resize(comp_bytes);
  CUDA_CHECK(cudaMemcpy(compressed.data(), d_output_data, comp_bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_input_data));
  CUDA_CHECK(cudaFree(d_comp_temp));
  CUDA_CHECK(cudaFree(d_output_data));
  CUDA_CHECK(cudaFree(d_output_size));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

void print_usage()
{
  std::cerr << std::endl;
  std::cerr << "Usage: lz4_gpu_compression_introduction [OPTIONS]" << std::endl;
  std::cerr << "  -f <input file>" << std::endl;
  std::cerr << "  -o <output file>" << std::endl;
}

template<unsigned int Arch>
struct Runner {
  template<typename... Args>
  static int run(Args&&... args)
  {
    return lz4_gpu_comp_introduction<Arch>(std::forward<Args>(args)...);
  }
};

int main(int argc, char* argv[])
{
  std::string file_name_in;
  std::string file_name_out;

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
        if(i < argc) {
          file_name_in = argv[i++];
        }
      } else if (strcmp(current_argv, "-o") == 0) {
        if(i < argc) {
          file_name_out = argv[i++];
        }
      } else {
        std::cerr << "Unknown argument: " << current_argv << std::endl;
        print_usage();
        return 1;
      }
    }
  } while (0);

  if (file_name_in.empty()) {
    std::cerr << "Must specify one input file via '-f <file>'." << std::endl;
    print_usage();
    return 1;
  } else if (file_name_out.empty()) {
    std::cerr << "Must specify one output file via '-o <file>'." << std::endl;
    print_usage();
    return 1;
  }

  auto input = read_file(file_name_in);

  if (input.size() > chunk_size) {
    std::cerr << "The input file is too large for this example. " << std::endl
              << "Select an input file up to " << chunk_size << " bytes." << std::endl;
    return 1;
  }

  std::vector<char> output;
  int ret = run_with_current_arch<Runner>(input,
                                          output,
                                          warmup_iteration_count,
                                          total_iteration_count);

  write_file(file_name_out, output);

  return 0;
}
