/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

// nvcc has a known issue with MSVC debug iterators, leading to a warning
// hit by thrust::device_vector construction from std::vector below, so this
// pragma disables the warning.
// More info at: https://github.com/NVIDIA/thrust/issues/1273
#ifdef __CUDACC__
#pragma nv_diag_suppress 20011
#endif

#include <thrust/device_vector.h>

#include <cuda.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "benchmark_common.h"
#include "benchmark_lossy_common.h"

#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif // _MSC_VER

#include "nvbench/nvbench.cuh"

#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif // _MSC_VER

#include "custom_criterion.hpp"

struct args_type
{
  std::vector<std::string> filenames;
  // Represents the number of bytes the input data needs to be a multiple of.
  // If it is not the case, the input data will be padded with zeros to satisfy the
  // requirement.
  size_t multiple_of;
  // Indicates the number of times the input data will be duplicated. In case
  // the input data went under some padding to satisfy `multiple_of`, the padded
  // data is duplicated.
  size_t duplicate_count;
  nvcompDecompressBackend_t decompress_backend;
  size_t chunk_size;
  bool compressed_inputs;
  bool single_output_buffer;
  std::string output_compressed_filename;
  std::string output_decompressed_filename;
};

static args_type args;
static std::vector<std::vector<char>> data;
// NOTE: populated by run_compression, consumed by run_decompression. Not cleared between
// NVBench runs - if parameter sweeps are added, this must be reset per configuration.
static std::vector<std::vector<char>> compressed_data_host;

template <typename T>
static thrust::device_vector<T> allocateThrustDeviceVectorSafe(size_t size)
{
  try
  {
    return thrust::device_vector<T>(size);
  }
  catch (const std::bad_alloc &)
  {
    size_t gpu_bytes_free, gpu_bytes_total;
    CUDA_CHECK(cudaMemGetInfo(&gpu_bytes_free, &gpu_bytes_total));
    if (gpu_bytes_free < size * sizeof(T))
    {
      std::cerr << "WARNING: Cannot fit data in GPU memory. Bytes requested: " << size * sizeof(T)
                << " > bytes available: " << gpu_bytes_free << ". Could not run benchmark." << std::endl;
    }
    std::exit(3);
  }
}

template <typename T>
static thrust::device_vector<T> allocateThrustDeviceVectorSafe(const std::vector<T> &host_vector)
{
  try
  {
    return thrust::device_vector<T>(host_vector);
  }
  catch (const std::bad_alloc &)
  {
    size_t gpu_bytes_free, gpu_bytes_total;
    CUDA_CHECK(cudaMemGetInfo(&gpu_bytes_free, &gpu_bytes_total));
    if (gpu_bytes_free < host_vector.size() * sizeof(T))
    {
      std::cerr << "WARNING: Cannot fit data in GPU memory. Bytes requested: " << host_vector.size() * sizeof(T)
                << " > bytes available: " << gpu_bytes_free << ". Could not run benchmark." << std::endl;
    }
    std::exit(3);
  }
}

// A helper for if the input data requires no validation. A generic lambda so it can be passed
// by value to run_compression and called with any codec's compress/decompress opts types.
static constexpr auto inputAlwaysValid = [](const std::vector<std::vector<char>> &, bool, const auto &, const auto &) {
  return true;
};

static nvcompType_t string_to_data_type(const char *name, bool &valid)
{
  valid = true;
  if (strcmp(name, "char") == 0)
  {
    return NVCOMP_TYPE_CHAR;
  }
  if (strcmp(name, "short") == 0)
  {
    return NVCOMP_TYPE_SHORT;
  }
  if (strcmp(name, "int") == 0)
  {
    return NVCOMP_TYPE_INT;
  }
  if (strcmp(name, "longlong") == 0)
  {
    return NVCOMP_TYPE_LONGLONG;
  }
  if (strcmp(name, "uchar") == 0)
  {
    return NVCOMP_TYPE_UCHAR;
  }
  if (strcmp(name, "ushort") == 0)
  {
    return NVCOMP_TYPE_USHORT;
  }
  if (strcmp(name, "uint") == 0)
  {
    return NVCOMP_TYPE_UINT;
  }
  if (strcmp(name, "ulonglong") == 0)
  {
    return NVCOMP_TYPE_ULONGLONG;
  }
  if (strcmp(name, "bits") == 0)
  {
    return NVCOMP_TYPE_BITS;
  }
  if (strcmp(name, "float16") == 0)
  {
    return NVCOMP_TYPE_FLOAT16;
  }
  if (strcmp(name, "float8_e4m3") == 0)
  {
    return NVCOMP_TYPE_FLOAT8_E4M3;
  }

  std::cerr << "ERROR: Unhandled type argument \"" << name << "\"" << std::endl;
  valid = false;
  return NVCOMP_TYPE_BITS;
}

using namespace nvcomp;

namespace
{

constexpr const char *const REQUIRED_PARAMETER = "_REQUIRED_";

class BatchData
{
public:
  BatchData(const std::vector<std::vector<char>> &host_data, const size_t alignment)
      : m_ptrs()
      , m_sizes()
      , m_data()
      , m_size(0)
  {
    m_size = host_data.size();

    // find max chunk size and build prefixsum
    std::vector<size_t> prefixsum(m_size + 1, 0);
    size_t chunk_size = 0;
    for (size_t i = 0; i < m_size; ++i)
    {
      if (chunk_size < host_data[i].size())
      {
        chunk_size = host_data[i].size();
      }
      // Align according to the given alignment
      prefixsum[i + 1] = nvcomp::roundUpTo(prefixsum[i] + host_data[i].size(), alignment);
    }

    m_data = allocateThrustDeviceVectorSafe<uint8_t>(prefixsum.back());

    std::vector<void *> uncompressed_ptrs(size());
    for (size_t i = 0; i < size(); ++i)
    {
      uncompressed_ptrs[i] = static_cast<void *>(data() + prefixsum[i]);
    }

    m_ptrs = allocateThrustDeviceVectorSafe(uncompressed_ptrs);

    std::vector<size_t> sizes(m_size);
    for (size_t i = 0; i < sizes.size(); ++i)
    {
      sizes[i] = host_data[i].size();
    }
    m_sizes = allocateThrustDeviceVectorSafe(sizes);

    // copy data to GPU
    for (size_t i = 0; i < host_data.size(); ++i)
    {
      CUDA_CHECK(cudaMemcpy(uncompressed_ptrs[i], host_data[i].data(), host_data[i].size(), cudaMemcpyHostToDevice));
    }
  }

  BatchData(const size_t max_output_size, const size_t batch_size, const size_t alignment)
      : m_ptrs()
      , m_sizes()
      , m_data()
      , host_ptrs(batch_size)
      , m_size(batch_size)
  {
    const size_t aligned_max_output_size = roundUpTo(max_output_size, alignment);
    m_data = allocateThrustDeviceVectorSafe<uint8_t>(aligned_max_output_size * size());

    std::vector<size_t> sizes(size(), aligned_max_output_size);
    m_sizes = allocateThrustDeviceVectorSafe(sizes);

    for (size_t i = 0; i < size(); ++i)
    {
      host_ptrs[i] = data() + aligned_max_output_size * i;
    }
    m_ptrs = allocateThrustDeviceVectorSafe(host_ptrs);
  }

  BatchData(BatchData &&other) = default;

  // disable copying
  BatchData(const BatchData &other) = delete;
  BatchData &operator=(const BatchData &other) = delete;

  void load_data(const std::vector<std::vector<char>> &host_data)
  {
    // copy the data to GPU
    for (size_t i = 0; i < host_data.size(); ++i)
    {
      CUDA_CHECK(cudaMemcpy(get_ptrs()[i], host_data[i].data(), host_data[i].size(), cudaMemcpyHostToDevice));
    }

    // copy the size to GPU
    std::vector<size_t> sizes(m_size);
    for (size_t i = 0; i < sizes.size(); ++i)
    {
      sizes[i] = host_data[i].size();
    }
    m_sizes = sizes;
  }

  void **ptrs() { return m_ptrs.data().get(); }

  thrust::device_ptr<void *> get_ptrs() { return m_ptrs.data(); }

  size_t *sizes() { return m_sizes.data().get(); }

  uint8_t *data() { return m_data.data().get(); }

  size_t total_size() const { return m_data.size(); }

  size_t size() const { return m_size; }

private:
  std::vector<void *> host_ptrs;
  thrust::device_vector<void *> m_ptrs;
  thrust::device_vector<size_t> m_sizes;
  thrust::device_vector<uint8_t> m_data;
  size_t m_size;
};

std::vector<std::vector<char>> readFileWithPageSizes(const std::string &filename)
{
  std::vector<std::vector<char>> res;

  std::ifstream fin(filename, std::ifstream::binary);

  while (!fin.eof())
  {
    uint64_t chunk_size;
    fin.read(reinterpret_cast<char *>(&chunk_size), sizeof(uint64_t));
    if (fin.eof())
    {
      break;
    }
    res.emplace_back(chunk_size);
    fin.read(reinterpret_cast<char *>(res.back().data()), chunk_size);
  }

  return res;
}

} // namespace

template <typename ExecTagT, typename KernelLauncherT>
void exec_or_waive_on_oom(nvbench::state &state, ExecTagT exec_tag, KernelLauncherT &&kernel_launcher)
{
  try
  {
    state.exec(exec_tag, std::forward<KernelLauncherT>(kernel_launcher));
  }
  catch (const std::exception &e)
  {
    if (std::string(e.what()).find(cudaGetErrorName(cudaErrorMemoryAllocation)) != std::string::npos)
    {
      std::cerr << "WARNING: Out of memory during NVBench measurement. Could not run benchmark." << std::endl;
      std::exit(3);
    }
    throw;
  }
}

template <
  typename CompGetTempAsyncT,
  typename CompGetTempSyncT,
  typename CompGetSizeT,
  typename CompAsyncT,
  typename CompAlignmentReqsT,
  typename IsInputValidT,
  typename CompressOptsT,
  typename DecompressOptsT,
  typename ExecTagT = decltype(nvbench::exec_tag::no_batch)>
void run_compression(
  CompGetTempAsyncT BatchedCompressGetTempSizeAsync,
  CompGetTempSyncT BatchedCompressGetTempSizeSync,
  CompGetSizeT BatchedCompressGetMaxOutputChunkSize,
  CompAsyncT BatchedCompressAsync,
  CompAlignmentReqsT BatchedCompressAlignmentReqs,
  IsInputValidT IsInputValid,
  const CompressOptsT compress_opts,
  const DecompressOptsT decompress_opts,
  nvbench::state &state,
  ExecTagT exec_tag = nvbench::exec_tag::no_batch
)
{
  benchmark_assert(IsInputValid(data, args.compressed_inputs, compress_opts, decompress_opts), "Invalid input data");
  if (args.compressed_inputs)
  {
    state.skip("Benchmark configured to skip compression because input data is already compressed.");
    return;
  }
  if (data.empty())
  {
    state.skip("Given input file is empty.");
    return;
  }

  const size_t batch_size = data.size();
  size_t total_bytes = 0;
  size_t max_uncompressed_chunk_size = 0;
  for (const std::vector<char> &chunk : data)
  {
    auto chunk_size = chunk.size();
    total_bytes += chunk_size;
    max_uncompressed_chunk_size = std::max(chunk_size, max_uncompressed_chunk_size);
  }

  auto &stream = state.get_cuda_stream();

  nvcompAlignmentRequirements_t compression_alignment_reqs{};
  nvcompStatus_t status = BatchedCompressAlignmentReqs(compress_opts, &compression_alignment_reqs);
  benchmark_assert(status == nvcompSuccess, "BatchedCompressAlignmentReqs() failed.");

  // Conditional container, used for round-trip compression-decompression benchmarking
  std::unique_ptr<BatchData> input_data =
    std::make_unique<BatchData>(data, std::max(size_t(16), compression_alignment_reqs.input));

  // Compression
  size_t max_compressed_chunk_size;
  status = BatchedCompressGetMaxOutputChunkSize(max_uncompressed_chunk_size, compress_opts, &max_compressed_chunk_size);
  benchmark_assert(status == nvcompSuccess, "BatchedCompressGetMaxOutputChunkSize() failed.");

  // Note: we need to respect the minimum output alignment requirement of the compressor.
  BatchData compressed_data(max_compressed_chunk_size, batch_size, compression_alignment_reqs.output);

  // Compress on the GPU using batched API
  size_t comp_temp_bytes_async = 0;
  status = BatchedCompressGetTempSizeAsync(
    batch_size,
    max_uncompressed_chunk_size,
    compress_opts,
    &comp_temp_bytes_async,
    batch_size * max_uncompressed_chunk_size
  );
  benchmark_assert(status == nvcompSuccess, "BatchedCompressGetTempSizeAsync() failed.");

  size_t comp_temp_bytes_sync = 0;
  status = BatchedCompressGetTempSizeSync(
    input_data->ptrs(),
    input_data->sizes(),
    batch_size,
    max_uncompressed_chunk_size,
    compress_opts,
    &comp_temp_bytes_sync,
    batch_size * max_uncompressed_chunk_size,
    stream
  );
  benchmark_assert(status == nvcompSuccess, "BatchedCompressGetTempSizeSync() failed.");

  const size_t comp_temp_bytes = comp_temp_bytes_sync;
  if (comp_temp_bytes_sync > comp_temp_bytes_async) {
    std::cerr << "The required sync compression temp space is greater than the async one." << std::endl;
  }

  void *d_comp_temp;
  CUDA_CHECK(cudaMallocSafe(&d_comp_temp, comp_temp_bytes));

  nvcompStatus_t *d_comp_statuses;
  CUDA_CHECK(cudaMallocSafe(&d_comp_statuses, batch_size * sizeof(nvcompStatus_t)));

  // Use the launch stream just in case, so sync this one first in case they are not the same one
  CUDA_CHECK(cudaStreamSynchronize(stream));

  state.add_element_count(total_bytes, "Bytes");
  // Set it large so that it never triggers in practice, unless a deadlock is truly happening.
  state.set_blocking_kernel_timeout(600.0);

  exec_or_waive_on_oom(state, exec_tag, [&](nvbench::launch &launch) {
    status = BatchedCompressAsync(
      input_data->ptrs(),
      input_data->sizes(),
      max_uncompressed_chunk_size,
      batch_size,
      d_comp_temp,
      comp_temp_bytes,
      compressed_data.ptrs(),
      compressed_data.sizes(),
      compress_opts,
      d_comp_statuses,
      launch.get_stream()
    );
    benchmark_assert(status == nvcompSuccess, "BatchedCompressAsync() failed.");
  });

  // verify statuses
  std::vector<nvcompStatus_t> h_comp_statuses(batch_size);
  CUDA_CHECK(
    cudaMemcpy(h_comp_statuses.data(), d_comp_statuses, sizeof(nvcompStatus_t) * batch_size, cudaMemcpyDeviceToHost)
  );
  for (size_t i = 0; i < batch_size; ++i)
  {
    benchmark_assert(
      h_comp_statuses[i] == nvcompSuccess,
      "Batch item not successfuly compressed: i=" + std::to_string(i) + ": status=" + std::to_string(h_comp_statuses[i])
    );
  }

  // free compression memory
  CUDA_CHECK(cudaFree(d_comp_temp));
  CUDA_CHECK(cudaFree(d_comp_statuses));

  // compute compression ratio
  std::vector<size_t> compressed_sizes_host(batch_size);
  CUDA_CHECK(cudaMemcpy(
    compressed_sizes_host.data(),
    compressed_data.sizes(),
    batch_size * sizeof(*compressed_data.sizes()),
    cudaMemcpyDeviceToHost
  ));

  const size_t comp_bytes = std::accumulate(compressed_sizes_host.begin(), compressed_sizes_host.end(), 0ULL);

  auto &summary = state.add_summary("Compression Ratio");
  summary.set_string("name", "Compression Ratio");
  summary.set_float64("value", static_cast<double>(total_bytes) / comp_bytes);

  // Copy the compressed data to the host for decompression
  if (compressed_data_host.empty())
  {
    std::vector<uint8_t *> comp_ptrs(batch_size);
    CUDA_CHECK(cudaMemcpy(comp_ptrs.data(), compressed_data.ptrs(), sizeof(size_t) * batch_size, cudaMemcpyDefault));
    for (size_t i = 0; i < batch_size; ++i)
    {
      compressed_data_host.emplace_back(compressed_sizes_host[i]);
      CUDA_CHECK(
        cudaMemcpy(compressed_data_host.back().data(), comp_ptrs[i], compressed_sizes_host[i], cudaMemcpyDeviceToHost)
      );
      if (!args.output_compressed_filename.empty())
      {
        std::ofstream outfile{
          args.output_compressed_filename.c_str() + std::string(".") + std::to_string(i),
          outfile.binary
        };
        outfile.write(reinterpret_cast<char *>(compressed_data_host.back().data()), compressed_sizes_host[i]);
        outfile.close();
      }
    }
  }
}

// DE-capable decompression opts (deflate/gzip/lz4/snappy) expose a
// `sort_before_hw_decompress` field; other formats do not. Used to detect at
// compile time whether a format can use the hardware Decompression Engine (DE).
template <typename, typename = void>
inline constexpr bool decompress_opts_supports_hw_v = false;

template <typename T>
inline constexpr bool
  decompress_opts_supports_hw_v<T, std::void_t<decltype(std::declval<T &>().sort_before_hw_decompress)>> = true;

static bool device_has_decompress_engine()
{
  // Despite the name, cudaDriverGetVersion() returns the latest CUDA version the
  // installed driver supports.
  int driver_latest_supported_cuda_version = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver_latest_supported_cuda_version));
  if (driver_latest_supported_cuda_version < 12080)
  {
    return false;
  }
  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));
  int decompress_algorithm_mask = 0;
  const CUresult res =
    cuDeviceGetAttribute(&decompress_algorithm_mask, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK, device_id);
  return res == CUDA_SUCCESS && decompress_algorithm_mask != 0;
}

template <
  typename DecompGetTempAsyncT,
  typename DecompGetTempSyncT,
  typename DecompAsyncT,
  typename DecompGetSizeT,
  typename DecompAlignmentReqsT,
  typename DecompressOptsT,
  typename ExecTagT = decltype(nvbench::exec_tag::no_batch)>
void run_decompression(
  DecompGetTempAsyncT BatchedDecompressGetTempSizeAsync,
  DecompGetTempSyncT BatchedDecompressGetTempSizeSync,
  DecompAsyncT BatchedDecompressAsync,
  DecompGetSizeT BatchedDecompressGetSize,
  DecompAlignmentReqsT BatchedDecompressAlignmentReqs,
  DecompressOptsT decompress_opts,
  nvbench::state &state,
  ExecTagT exec_tag = nvbench::exec_tag::no_batch,
  const double lossy_delta = 0.0,
  const int lossy_fp_bits = 0
)
{
  if (!args.compressed_inputs && compressed_data_host.empty())
  {
    state.skip("Benchmark configured to skip decompression because no compressed data is available.");
    return;
  }
  const auto &input_data = args.compressed_inputs ? data : compressed_data_host;
  const size_t batch_size = input_data.size();
  std::vector<size_t> h_input_sizes(batch_size);
  std::transform(input_data.begin(), input_data.end(), h_input_sizes.begin(), [](const std::vector<char> &chunk) {
    return chunk.size();
  });

  decompress_opts.backend = args.decompress_backend;

  auto &stream = state.get_cuda_stream();

  nvcompAlignmentRequirements_t decompression_alignment_reqs{};
  auto status = BatchedDecompressAlignmentReqs(decompress_opts, &decompression_alignment_reqs);
  benchmark_assert(status == nvcompSuccess, "BatchedDecompressAlignmentReqs() failed.");

  auto compressed_data = BatchData(input_data, std::max(size_t(16), decompression_alignment_reqs.input));

  size_t *d_decomp_buffer_sizes;
  CUDA_CHECK(cudaMallocSafe(&d_decomp_buffer_sizes, batch_size * sizeof(size_t)));

  // Determine the size of decompressed chunks
  status =
    BatchedDecompressGetSize(compressed_data.ptrs(), compressed_data.sizes(), d_decomp_buffer_sizes, batch_size, stream);
  benchmark_assert(status == nvcompSuccess, "BatchedDecompressGetSize() not successful");
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Copy to host the expected decompressed chunk sizes
  std::vector<size_t> h_decomp_sizes(batch_size);
  CUDA_CHECK(cudaMemcpy(h_decomp_sizes.data(), d_decomp_buffer_sizes, sizeof(size_t) * batch_size, cudaMemcpyDefault));
  size_t max_uncompressed_chunk_size = 0;
  for (auto &uncompressed_chunk_size : h_decomp_sizes)
  {
    max_uncompressed_chunk_size = std::max(max_uncompressed_chunk_size, uncompressed_chunk_size);
  }

  // Decompression
  nvcompStatus_t *d_decomp_statuses;
  CUDA_CHECK(cudaMallocSafe(&d_decomp_statuses, batch_size * sizeof(nvcompStatus_t)));

  size_t decomp_temp_bytes_async;
  status = BatchedDecompressGetTempSizeAsync(
    batch_size,
    max_uncompressed_chunk_size,
    decompress_opts,
    &decomp_temp_bytes_async,
    batch_size * max_uncompressed_chunk_size
  );
  benchmark_assert(status == nvcompSuccess, "BatchedDecompressGetTempSizeAsync() failed.");

  size_t decomp_temp_bytes_sync;
  status = BatchedDecompressGetTempSizeSync(
    compressed_data.ptrs(),
    compressed_data.sizes(),
    batch_size,
    max_uncompressed_chunk_size,
    &decomp_temp_bytes_sync,
    batch_size * max_uncompressed_chunk_size,
    decompress_opts,
    d_decomp_statuses,
    stream
  );
  benchmark_assert(status == nvcompSuccess, "BatchedDecompressGetTempSizeSync() failed.");

  const size_t decomp_temp_bytes = decomp_temp_bytes_sync;
  if (decomp_temp_bytes_sync > decomp_temp_bytes_async) {
    std::cerr << "The required sync decompression temp space is greater than the async one." << std::endl;
  }

  void *d_decomp_temp;
  CUDA_CHECK(cudaMallocSafe(&d_decomp_temp, decomp_temp_bytes));

  std::vector<void *> h_output_ptrs(batch_size);
  thrust::device_vector<void *> d_output_ptrs_tight;
  size_t total_uncomp_size = 0;
  for (size_t i = 0; i < batch_size; ++i)
  {
    total_uncomp_size += h_decomp_sizes[i];
  }
  thrust::device_vector<uint8_t> one_buffer;
  void **d_output_ptrs;

  if (args.single_output_buffer)
  {
    one_buffer = allocateThrustDeviceVectorSafe<uint8_t>(total_uncomp_size);
    size_t offset = 0;
    for (size_t i = 0; i < batch_size; ++i)
    {
      benchmark_assert(
        offset % decompression_alignment_reqs.output == 0,
        "Decompression output alignment requirement is not met"
      );
      h_output_ptrs[i] = static_cast<void *>(one_buffer.data().get() + offset);
      offset += h_decomp_sizes[i];
    }

    d_output_ptrs_tight = allocateThrustDeviceVectorSafe(h_output_ptrs);
    d_output_ptrs = d_output_ptrs_tight.data().get();
  }
  else
  {
    for (size_t i = 0; i < batch_size; ++i)
    {
      CUDA_CHECK(cudaMallocSafe(&h_output_ptrs[i], h_decomp_sizes[i]));
    }
    CUDA_CHECK(cudaMallocSafe(&d_output_ptrs, sizeof(*d_output_ptrs) * batch_size));
    // Note:
    // output alignment requirements are implicitly met
    CUDA_CHECK(
      cudaMemcpy(d_output_ptrs, h_output_ptrs.data(), sizeof(*d_output_ptrs) * batch_size, cudaMemcpyHostToDevice)
    );
  }
  size_t *d_decomp_sizes;
  CUDA_CHECK(cudaMallocSafe(&d_decomp_sizes, batch_size * sizeof(size_t)));

  // Use the launch stream just in case, so sync this one first in case they are not the same one
  CUDA_CHECK(cudaStreamSynchronize(stream));

  state.add_element_count(total_uncomp_size, "Bytes");
  // Set it large so that it never triggers in practice, unless a deadlock is truly happening.
  state.set_blocking_kernel_timeout(600.0);

  auto decompression_launcher = [&](nvbench::launch &launch) {
    status = BatchedDecompressAsync(
      compressed_data.ptrs(),
      compressed_data.sizes(),
      d_decomp_buffer_sizes,
      d_decomp_sizes,
      batch_size,
      d_decomp_temp,
      decomp_temp_bytes,
      d_output_ptrs,
      decompress_opts,
      d_decomp_statuses,
      launch.get_stream()
    );
    benchmark_assert(status == nvcompSuccess, "BatchedDecompressAsync() not successful");
  };

  // The hardware Decompression Engine (DE) submits work synchronously from the
  // host's perspective, so tell NVBench via exec_tag::sync to disable its
  // blocking kernel; otherwise it reports a false "Possible Deadlock Detected".
  bool use_sync = false;
  if constexpr (decompress_opts_supports_hw_v<DecompressOptsT>)
  {
    if ((decompress_opts.backend == NVCOMP_DECOMPRESS_BACKEND_DEFAULT ||
         decompress_opts.backend == NVCOMP_DECOMPRESS_BACKEND_HARDWARE) &&
        device_has_decompress_engine())
    {
      use_sync = true;
    }
  }
  if (use_sync)
  {
    exec_or_waive_on_oom(state, exec_tag | nvbench::exec_tag::sync, decompression_launcher);
  }
  else
  {
    exec_or_waive_on_oom(state, exec_tag, decompression_launcher);
  }

  CUDA_CHECK(
    cudaMemcpy(h_decomp_sizes.data(), d_decomp_sizes, sizeof(*d_decomp_sizes) * batch_size, cudaMemcpyDeviceToHost)
  );

  const size_t total_decomp_bytes = std::accumulate(h_decomp_sizes.begin(), h_decomp_sizes.end(), 0ULL);
  const size_t total_comp_bytes =
    std::accumulate(input_data.begin(), input_data.end(), 0ULL, [](size_t sum, const std::vector<char> &chunk) {
      return sum + chunk.size();
    });

  auto &summary = state.add_summary("Compression Ratio");
  summary.set_string("name", "Compression Ratio");
  summary.set_float64("value", static_cast<double>(total_decomp_bytes) / total_comp_bytes);

  // Verify success
  std::vector<nvcompStatus_t> h_decomp_statuses(batch_size);
  CUDA_CHECK(cudaMemcpy(
    h_decomp_statuses.data(),
    d_decomp_statuses,
    sizeof(*d_decomp_statuses) * batch_size,
    cudaMemcpyDeviceToHost
  ));
  for (size_t i = 0; i < batch_size; ++i)
  {
    benchmark_assert(
      h_decomp_statuses[i] == nvcompSuccess,
      "Batch item not successfuly decompressed: i=" + std::to_string(i) +
        ": status=" + std::to_string(h_decomp_statuses[i])
    );
    if (!args.compressed_inputs)
    {
      benchmark_assert(
        h_decomp_sizes[i] == data[i].size(),
        "Batch item of wrong size: i=" + std::to_string(i) + ": act_size=" + std::to_string(h_decomp_sizes[i]) +
          " exp_size=" + std::to_string(data[i].size())
      );
    }
  }

  CUDA_CHECK(cudaFree(d_decomp_temp));
  CUDA_CHECK(cudaFree(d_decomp_statuses));

  // Copy buffers to host and verify against original uncompressed data
  if (!args.compressed_inputs)
  {
    for (size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk)
    {
      const std::vector<char> &exp_chunk = data[ix_chunk];
      std::vector<uint8_t> act_data(h_decomp_sizes[ix_chunk]);
      CUDA_CHECK(cudaMemcpy(act_data.data(), h_output_ptrs[ix_chunk], h_decomp_sizes[ix_chunk], cudaMemcpyDeviceToHost));

      if (lossy_delta > 0.0)
      {
        verifyLossyCompression(
          reinterpret_cast<const uint8_t *>(exp_chunk.data()),
          act_data.data(),
          exp_chunk.size(),
          lossy_delta,
          getLossyDataType(lossy_fp_bits)
        );
      }
      else
      {
        for (size_t ix_byte = 0; ix_byte < exp_chunk.size(); ++ix_byte)
        {
          if (act_data[ix_byte] != static_cast<uint8_t>(exp_chunk[ix_byte]))
          {
            benchmark_assert(
              false,
              "Batch item decompressed output did not match input: ix_chunk=" + std::to_string(ix_chunk) +
                ": ix_byte=" + std::to_string(ix_byte) + " act=" + std::to_string(act_data[ix_byte]) +
                " exp=" + std::to_string(static_cast<uint8_t>(exp_chunk[ix_byte]))
            );
          }
        }
      }
    }
  }

  if (args.compressed_inputs && !args.output_decompressed_filename.empty())
  {
    std::vector<uint8_t> uncomp_data(total_decomp_bytes);
    size_t ix_offset = 0;
    for (size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk)
    {
      CUDA_CHECK(
        cudaMemcpy(&uncomp_data[ix_offset], h_output_ptrs[ix_chunk], h_decomp_sizes[ix_chunk], cudaMemcpyDeviceToHost)
      );
      if (!args.single_output_buffer)
      {
        std::ofstream outfile{
          args.output_decompressed_filename.c_str() + std::string(".") + std::to_string(ix_chunk),
          outfile.binary
        };
        outfile.write(reinterpret_cast<char *>(&uncomp_data[ix_offset]), h_decomp_sizes[ix_chunk]);
        outfile.close();
      }
      ix_offset += h_decomp_sizes[ix_chunk];
    }
    if (args.single_output_buffer)
    {
      std::ofstream outfile{args.output_decompressed_filename.c_str(), outfile.binary};
      outfile.write(reinterpret_cast<char *>(uncomp_data.data()), total_decomp_bytes);
      outfile.close();
    }
  }

  if (!args.single_output_buffer)
  {
    CUDA_CHECK(cudaFree(d_output_ptrs));
    for (size_t i = 0; i < batch_size; ++i)
    {
      CUDA_CHECK(cudaFree(h_output_ptrs[i]));
    }
  }
  CUDA_CHECK(cudaFree(d_decomp_buffer_sizes));
  CUDA_CHECK(cudaFree(d_decomp_sizes));
}

struct parameter_type
{
  std::string short_flag;
  std::string long_flag;
  std::string description;
  std::string default_value;
  std::function<bool(const char *)> handler;
};

// Each benchmark must define this vector with its custom parameters and
// their parsing functions. If the benchmark has no custom parameters,
// define an empty vector.
extern const std::vector<parameter_type> custom_params;

void usage(const std::string &name, const std::vector<parameter_type> &parameters)
{
  std::cout << "Usage: " << name << " [OPTIONS]" << std::endl;

  auto printArgUsage = [](const parameter_type &parameter) {
    const bool has_short_flag = !parameter.short_flag.empty();
    std::cout << "  ";
    if (has_short_flag)
    {
      const std::string short_prefix = parameter.short_flag.size() > 1 ? "--" : "-";
      std::cout << short_prefix << parameter.short_flag << ",";
    }
    std::cout << "--" << parameter.long_flag;
    std::cout << "  : " << parameter.description << std::endl;
    if (parameter.default_value.empty())
    {
      // no default value
    }
    else if (parameter.default_value == REQUIRED_PARAMETER)
    {
      std::cout << "    required" << std::endl;
    }
    else
    {
      std::cout << "    default=" << parameter.default_value << std::endl;
    }
  };
  for (const parameter_type &parameter : parameters)
  {
    printArgUsage(parameter);
  }
  if (!custom_params.empty())
  {
    std::cout << std::endl << "Format specific options:" << std::endl << std::endl;
    for (const auto &param : custom_params)
    {
      printArgUsage(param);
    }
  }
}

bool parse_args(int &argc, char **argv)
{
  bool skip_benchmark = false;
  args.multiple_of = 1;
  args.duplicate_count = 0;
  args.decompress_backend = NVCOMP_DECOMPRESS_BACKEND_DEFAULT;
  args.chunk_size = 65536;
  args.compressed_inputs = false;
  args.single_output_buffer = false;

  std::vector<parameter_type> params{
    {"?", "help", "Show options.", "", nullptr},
    {"db",
     "decompress_backend",
     "Decompression backend to use : Best available (0), HW (1) or CUDA (2). "
     "Default is best available",
     std::to_string(args.decompress_backend),
     [](const char *arg) {
       auto val = static_cast<nvcompDecompressBackend_t>(size_t(std::stol(arg)));
       if (val != NVCOMP_DECOMPRESS_BACKEND_DEFAULT && val != NVCOMP_DECOMPRESS_BACKEND_HARDWARE &&
           val != NVCOMP_DECOMPRESS_BACKEND_CUDA)
       {
         std::cerr << "ERROR: --decompress_backend must be 0 (default), 1 (HW), or 2 (CUDA)." << std::endl;
         return false;
       }
       else
       {
         args.decompress_backend = val;
         return true;
       }
     }},
    {"compressed",
     "compressed_inputs",
     "The input dataset is compressed.",
     std::to_string(args.compressed_inputs),
     [](const char *arg) {
       args.compressed_inputs = parse_bool(arg);
       return true;
     }},
    {"f",
     "input_file",
     "The list of inputs files. All files must start "
     "with a character other than '-'",
     REQUIRED_PARAMETER,
     nullptr},
    {"m",
     "multiple_of",
     "Add padding to the input data such that its "
     "length becomes a multiple of the given argument (in bytes). Only "
     "applicable to "
     "data without page sizes.",
     std::to_string(args.multiple_of),
     [](const char *arg) {
       args.multiple_of = size_t(std::stoull(arg));
       if (args.multiple_of == 0)
       {
         std::cerr << "ERROR: --multiple_of must be greater than 0." << std::endl;
         return false;
       }
       return true;
     }},
    {"x",
     "duplicate_data",
     "Clone uncompressed chunks multiple times (scale factor, 1x means no "
     "duplication).",
     std::to_string(args.duplicate_count),
     [](const char *arg) {
       args.duplicate_count = size_t(std::stoull(arg));
       return true;
     }},
    {"p",
     "chunk_size",
     "Chunk size when splitting uncompressed data.",
     std::to_string(args.chunk_size),
     [](const char *arg) {
       args.chunk_size = size_t(std::stoull(arg));
       if (args.chunk_size == 0)
       {
         std::cerr << "ERROR: --chunk_size must be greater than 0." << std::endl;
         return false;
       }
       return true;
     }},
    {"single",
     "single_output_buffer",
     "There is only one tight output buffer during decompression.",
     std::to_string(args.single_output_buffer),
     [](const char *arg) {
       args.single_output_buffer = parse_bool(arg);
       return true;
     }},
    {"oc",
     "output_compressed_file",
     "Output compressed basename",
     "",
     [](const char *arg) {
       args.output_compressed_filename = arg;
       return true;
     }},
    {"o", "output_decompressed_file", "Output decompressed filename", "", [](const char *arg) {
       args.output_decompressed_filename = arg;
       return true;
     }},
  };

  char **const argv_orig = argv;
  char **argv_end = argv + argc;
  const std::string name(argv[0]);
  argv += 1;

  // Collect args not consumed by us, to pass through to NVBench.
  std::vector<char *> remaining_argv{argv_orig[0]};

  auto all_params = params;
  all_params.insert(all_params.end(), custom_params.begin(), custom_params.end());

  // Remove handled arguments from argv, so that NVBench does not get confused.
  while (argv != argv_end)
  {
    char *const cur_arg_ptr = *argv;
    std::string arg(*(argv++));
    bool found = false;
    for (const parameter_type &param : all_params)
    {
      const std::string short_prefix = param.short_flag.size() > 1 ? "--" : "-";
      if ((!param.short_flag.empty() && arg == short_prefix + param.short_flag) || arg == "--" + param.long_flag)
      {
        found = true;

        if (param.long_flag == "help")
        {
          usage(name, params);
          // NVBench does not recognize "-?" as a help flag, so we add the long one instead
          static char help_flag[] = "--help";
          remaining_argv.push_back(help_flag);
          skip_benchmark = true;
          break;
        }

        // everything from here on out requires an extra parameter
        if (argv >= argv_end)
        {
          std::cerr << "ERROR: Missing argument for '" << arg << "'." << std::endl;
          usage(name, params);
          std::exit(1);
        }
        if (param.long_flag == "input_file")
        {
          // read all following arguments until a new flag is found
          char **next_argv_ptr = argv;
          while (next_argv_ptr < argv_end && (*next_argv_ptr)[0] != '-')
          {
            args.filenames.emplace_back(*next_argv_ptr);
            next_argv_ptr = ++argv;
          }
          break;
        }
        if (!param.handler(*(argv++)))
        {
          std::exit(1);
        }
        break;
      }
    }
    if (!found)
    {
      // Not our argument -- pass through to NVBench unchanged.
      remaining_argv.push_back(cur_arg_ptr);
    }
  }

  if (!skip_benchmark && args.filenames.empty())
  {
    std::cerr << "WARNING: No input file specified (-f). Benchmarks will not run." << std::endl;
    skip_benchmark = true;
  }

  // Compact argv in-place with only the args NVBench should see.
  for (size_t i = 0; i < remaining_argv.size(); ++i)
  {
    argv_orig[i] = remaining_argv[i];
  }
  argv_orig[remaining_argv.size()] = nullptr;
  argc = static_cast<int>(remaining_argv.size());

  return skip_benchmark;
}

int main(int argc, char **argv)
{
  bool skip_benchmark = parse_args(argc, argv);

  if (!skip_benchmark)
  {
    data = multi_file(args.filenames, !args.compressed_inputs, args.chunk_size, args.multiple_of, args.duplicate_count);
  }

  // Default to a single GPU (device 0) unless the user passed NVBench's
  // -d/--device/--devices; NVBench otherwise runs on all visible devices.
  std::vector<char *> nvbench_argv(argv, argv + argc);
  nvcomp::default_to_single_device(nvbench_argv);
  int nvbench_argc = static_cast<int>(nvbench_argv.size());

  NVBENCH_MAIN_BODY(nvbench_argc, nvbench_argv.data());
}
