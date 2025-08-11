/*
 * Copyright (c) 2022-2025 NVIDIA CORPORATION AND AFFILIATES. All rights reserved.
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

#pragma once

// Benchmark performance from the binary data file fname
#include <vector>
#include <numeric>

// TODO: Our Windows x86 CI images for some reason do not contain the NVTX headers,
//       whereas the regular Windows CTK installations do contain them. Check why.
#if defined(__x86_64) && !defined(_MSC_VER)
#define NVTX_ENABLED
#endif // defined(__x86_64) && !defined(_MSC_VER)

#ifdef NVTX_ENABLED
#include <nvtx3/nvToolsExt.h>
#endif // NVTX_ENABLED

#include "benchmark_common.h"
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

using namespace nvcomp;

const int chunk_size = 1 << 16;

template<typename T = uint8_t>
void run_benchmark(
    const std::vector<T>& data, nvcompManagerBase& batch_manager, int verbose_memory, 
    cudaStream_t stream, const int benchmark_exec_count = 1, const bool warmup = true)
{
  size_t input_element_count = data.size();

  // Make sure dataset fits on GPU to benchmark total compression
  size_t freeMem;
  size_t totalMem;
  CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
  if (freeMem < input_element_count * sizeof(T)) {
    std::cout << "Insufficient GPU memory to perform compression." << std::endl;
    exit(1);
  }

  std::cout << "----------" << std::endl;
  std::cout << "uncompressed (B): " << data.size() * sizeof(T) << std::endl;

  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input_element_count;
  CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, data.data(), in_bytes, cudaMemcpyHostToDevice));

  auto compress_config = batch_manager.configure_compression(in_bytes);

  size_t comp_out_bytes = compress_config.max_compressed_buffer_size;
  benchmark_assert(
      comp_out_bytes > 0, "Output size must be greater than zero.");

  // Allocate temp workspace
  uint8_t* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  // Launch compression
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  if (warmup) {
#ifdef NVTX_ENABLED
    nvtxRangePush("compress_warmup");
#endif // NVTX_ENABLED
    batch_manager.compress(
        d_in_data,
        d_comp_out,
        compress_config);
    CUDA_CHECK(cudaStreamSynchronize(stream));

#ifdef NVTX_ENABLED
    nvtxRangePop();
#endif // NVTX_ENABLED
  }

  std::vector<float> compress_run_times(benchmark_exec_count);
  for (int ix_run = 0; ix_run < benchmark_exec_count; ++ix_run) {
#ifdef NVTX_ENABLED
    nvtxRangePush("compress");
#endif // NVTX_ENABLED
    CUDA_CHECK(cudaEventRecord(start, stream));
    batch_manager.compress(
        d_in_data,
        d_comp_out,
        compress_config);

    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
#ifdef NVTX_ENABLED
    nvtxRangePop();
#endif // NVTX_ENABLED
    comp_out_bytes = batch_manager.get_compressed_output_size(d_comp_out);
    float compress_ms;
    CUDA_CHECK(cudaEventElapsedTime(&compress_ms, start, end));
    compress_run_times[ix_run] = compress_ms;
  }

  // compute average run time.

  std::cout << "comp_size: " << comp_out_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)data.size() * sizeof(T) / comp_out_bytes << std::endl;
  std::cout << "compression throughput (GB/s): "
            << average_gbs(compress_run_times, data.size() * sizeof(T)) << std::endl;

  CUDA_CHECK(cudaFree(d_in_data));

  std::vector<float> decompress_run_times(benchmark_exec_count);

  auto decomp_config = batch_manager.configure_decompression(d_comp_out);
  // allocate output buffer
  const size_t decomp_bytes = decomp_config.decomp_data_size;
  uint8_t* decomp_out_ptr;
  CUDA_CHECK(cudaMalloc(&decomp_out_ptr, decomp_bytes));

  if (warmup) {
#ifdef NVTX_ENABLED
    nvtxRangePush("decomp warmup");
#endif // NVTX_ENABLED
    batch_manager.decompress(decomp_out_ptr, d_comp_out, decomp_config);
    CUDA_CHECK(cudaStreamSynchronize(stream));
#ifdef NVTX_ENABLED
    nvtxRangePop();
#endif // NVTX_ENABLED
  }

  for (int ix_run = 0; ix_run < benchmark_exec_count; ++ix_run) {
    // get output size
    if (verbose_memory) {
      std::cout << "decompression memory (input+output+temp) (B): "
                << (decomp_bytes + comp_out_bytes)
                << std::endl;
    }
#ifdef NVTX_ENABLED
    nvtxRangePush("decomp");
#endif // NVTX_ENABLED
    CUDA_CHECK(cudaEventRecord(start, stream));

    // execute decompression (asynchronous)
    batch_manager.decompress(decomp_out_ptr, d_comp_out, decomp_config);

    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
#ifdef NVTX_ENABLED
    nvtxRangePop();
#endif // NVTX_ENABLED

    float decompress_ms;
    CUDA_CHECK(cudaEventElapsedTime(&decompress_ms, start, end));
    decompress_run_times[ix_run] = decompress_ms;
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  std::cout << "decompression throughput (GB/s): "
            << average_gbs(decompress_run_times, decomp_bytes) << std::endl
            << "decompression time: "
            << std::accumulate(decompress_run_times.begin(), decompress_run_times.end(), 0.0) / benchmark_exec_count
            << " ms."
            << std::endl;

  CUDA_CHECK(cudaFree(d_comp_out));

  benchmark_assert(
      decomp_bytes == input_element_count * sizeof(T),
      "Decompressed result incorrect size.");

  std::vector<T> res(input_element_count);
  CUDA_CHECK(cudaMemcpy(
      res.data(),
      decomp_out_ptr,
      input_element_count * sizeof(T),
      cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(decomp_out_ptr));

  // check the size
#if VERBOSE > 1
  // dump output data
  std::cout << "Output" << std::endl;
  for (size_t i = 0; i < data.size(); i++)
    std::cout << reinterpret_cast<T*>(decomp_out_ptr)[i] << " ";
  std::cout << std::endl;
#endif // VERBOSE > 1
  benchmark_assert(res == data, "Decompressed data does not match input.");
}

#undef NVTX_ENABLED
