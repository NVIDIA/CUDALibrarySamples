/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#ifndef NVCOMP_BENCHMARKS_BENCHMARK_TEMPLATE_CHUNKED_CUH
#define NVCOMP_BENCHMARKS_BENCHMARK_TEMPLATE_CHUNKED_CUH

// nvcc has a known issue with MSVC debug iterators, leading to a warning
// hit by thrust::device_vector construction from std::vector below, so this
// pragma disables the warning.
// More info at: https://github.com/NVIDIA/thrust/issues/1273
#ifdef __CUDACC__
#pragma nv_diag_suppress 20011
#endif

#include "benchmark_common.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thrust/device_vector.h>
#include <vector>

namespace nvcomp {

template <typename U, typename T>
constexpr __host__ __device__ U roundUpDiv(U const num, T const chunk)
{
  return (num + chunk - 1) / chunk;
}

template <typename U, typename T>
constexpr __host__ __device__ U roundDownTo(U const num, T const chunk)
{
  return (num / chunk) * chunk;
}

template <typename U, typename T>
constexpr __host__ __device__ U roundUpTo(U const num, T const chunk)
{
  return roundUpDiv(num, chunk) * chunk;
}

}

// Each benchmark must implement this, returning true if the argument
// was handled.  If the benchmark has no custom arguments, its
// implementation can just return false.
static bool handleCommandLineArgument(
    const std::string& arg,
    const char* const* additionalArgs,
    size_t& additionalArgsUsed);

// A helper function for if the input data requires no validation.
static bool inputAlwaysValid(const std::vector<std::vector<char>>& data)
{
  return true;
}

static nvcompType_t string_to_data_type(const char* name, bool& valid)
{
  valid = true;
  if (strcmp(name, "char") == 0) {
    return NVCOMP_TYPE_CHAR;
  }
  if (strcmp(name, "short") == 0) {
    return NVCOMP_TYPE_SHORT;
  }
  if (strcmp(name, "int") == 0) {
    return NVCOMP_TYPE_INT;
  }
  if (strcmp(name, "longlong") == 0) {
    return NVCOMP_TYPE_LONGLONG;
  }
  if (strcmp(name, "uchar") == 0) {
    return NVCOMP_TYPE_UCHAR;
  }
  if (strcmp(name, "ushort") == 0) {
    return NVCOMP_TYPE_USHORT;
  }
  if (strcmp(name, "uint") == 0) {
    return NVCOMP_TYPE_UINT;
  }
  if (strcmp(name, "ulonglong") == 0) {
    return NVCOMP_TYPE_ULONGLONG;
  }
  if (strcmp(name, "bits") == 0) {
    return NVCOMP_TYPE_BITS;
  }
  if (strcmp(name, "uint8") == 0) {
    return NVCOMP_TYPE_UINT8;
  }
  if (strcmp(name, "float16") == 0) {
    return NVCOMP_TYPE_FLOAT16;
  }

  std::cerr << "ERROR: Unhandled type argument \"" << name << "\""
            << std::endl;
  valid = false;
  return NVCOMP_TYPE_BITS;
}

using namespace nvcomp;

namespace
{

constexpr const char * const REQUIRED_PARAMTER = "_REQUIRED_";

static size_t compute_batch_size(
    const std::vector<std::vector<char>>& data, const size_t chunk_size)
{
  size_t batch_size = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
    batch_size += num_chunks;
  }

  return batch_size;
}

std::vector<size_t> compute_chunk_sizes(
    const std::vector<std::vector<char>>& data,
    const size_t batch_size,
    const size_t chunk_size)
{
  std::vector<size_t> sizes(batch_size, chunk_size);

  size_t offset = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
    if (data[i].size() % chunk_size != 0) {
      sizes[offset] = data[i].size() % chunk_size;
    }
    offset += num_chunks;
  }
  return sizes;
}

class BatchData
{
public:
  BatchData(
      const std::vector<std::vector<char>>& host_data) :
      m_ptrs(),
      m_sizes(),
      m_data(),
      m_size(0)
  {
    m_size = host_data.size();

    // find max chunk size and build prefixsum
    std::vector<size_t> prefixsum(m_size+1,0);
    size_t chunk_size = 0;
    for (size_t i = 0; i < m_size; ++i) {
      if (chunk_size < host_data[i].size()) {
        chunk_size = host_data[i].size();
      }
      // align to 8B boundaries for now
      // TODO: Set appropriate alignment based on reqs for each compressor
      prefixsum[i+1] = nvcomp::roundUpTo(prefixsum[i] + host_data[i].size(), 8);
    }

    m_data = nvcomp::thrust::device_vector<uint8_t>(prefixsum.back());

    std::vector<void*> uncompressed_ptrs(size());
    for (size_t i = 0; i < size(); ++i) {
      uncompressed_ptrs[i] = static_cast<void*>(data() + prefixsum[i]);
    }

    m_ptrs = nvcomp::thrust::device_vector<void*>(uncompressed_ptrs);
    std::vector<size_t> sizes(m_size);
    for (size_t i = 0; i < sizes.size(); ++i) {
      sizes[i] = host_data[i].size();
    }
    m_sizes = nvcomp::thrust::device_vector<size_t>(sizes);

    size_t batch_bytes_required = prefixsum.back() * sizeof(uint8_t);
    
    size_t gpu_bytes_free, gpu_bytes_total;
    CUDA_CHECK(cudaMemGetInfo( &gpu_bytes_free, &gpu_bytes_total ));
    if(gpu_bytes_free < batch_bytes_required){
      std::cerr << "WARNING: Cannot fit compressed file in GPU memory. Could not run benchmark." << std::endl;
      std::exit(0);
    }

    // copy data to GPU
    for (size_t i = 0; i < host_data.size(); ++i) {
      CUDA_CHECK(cudaMemcpy(
          uncompressed_ptrs[i],
          host_data[i].data(),
          host_data[i].size(),
          cudaMemcpyHostToDevice));
    }
  }

  BatchData(const size_t max_output_size, const size_t batch_size) :
      m_ptrs(),
      m_sizes(),
      m_data(),
      host_ptrs(batch_size),
      m_size(batch_size)
  {
    size_t batch_bytes_required = max_output_size * size() * sizeof(uint8_t);
    
    size_t gpu_bytes_free, gpu_bytes_total;
    CUDA_CHECK(cudaMemGetInfo( &gpu_bytes_free, &gpu_bytes_total ));
    if(gpu_bytes_free < batch_bytes_required){
      std::cerr << "WARNING: Cannot fit compressed file in GPU memory. Could not run benchmark." << std::endl;
      std::exit(0);
    }

    m_data = nvcomp::thrust::device_vector<uint8_t>(max_output_size * size());

    std::vector<size_t> sizes(size(), max_output_size);
    m_sizes = nvcomp::thrust::device_vector<size_t>(sizes);

    for (size_t i = 0; i < batch_size; ++i) {
      host_ptrs[i] = data() + max_output_size * i;
    }
    m_ptrs = nvcomp::thrust::device_vector<void*>(host_ptrs);
  }

  BatchData(BatchData&& other) = default;

  // disable copying
  BatchData(const BatchData& other) = delete;
  BatchData& operator=(const BatchData& other) = delete;

  void load_data (const std::vector<std::vector<char>>& host_data)
  {
    // copy data to GPU
    for (size_t i = 0; i < host_data.size(); ++i) {
      CUDA_CHECK(cudaMemcpy(
          get_ptrs()[i],
          host_data[i].data(),
          host_data[i].size(),
          cudaMemcpyHostToDevice));
    }
  }
  
  void** ptrs()
  {
    return m_ptrs.data().get();
  }

  nvcomp::thrust::device_ptr<void *> get_ptrs()
  {
    return m_ptrs.data();
  }

  size_t* sizes()
  {
    return m_sizes.data().get();
  }

  uint8_t* data()
  {
    return m_data.data().get();
  }

  size_t total_size() const
  {
    return m_data.size();
  }

  size_t size() const
  {
    return m_size;
  }

private:
  std::vector<void*> host_ptrs;
  nvcomp::thrust::device_vector<void*> m_ptrs;
  nvcomp::thrust::device_vector<size_t> m_sizes;
  nvcomp::thrust::device_vector<uint8_t> m_data;
  size_t m_size;
};

std::vector<char> readFile(const std::string& filename)
{
  std::ifstream fin(filename, std::ifstream::binary);
  if (!fin) {
    std::cerr << "ERROR: Unable to open \"" << filename << "\" for reading."
              << std::endl;
    throw std::runtime_error("Error opening file for reading.");
  }

  fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  fin.seekg(0, std::ios_base::end);
  auto fileSize = static_cast<std::streamoff>(fin.tellg());
  fin.seekg(0, std::ios_base::beg);

  std::vector<char> host_data(fileSize);
  fin.read(host_data.data(), fileSize);

  if (!fin) {
    std::cerr << "ERROR: Unable to read all of file \"" << filename << "\"."
              << std::endl;
    throw std::runtime_error("Error reading file.");
  }

  return host_data;
}


std::vector<std::vector<char>> readFileWithPageSizes(const std::string& filename)
{
  std::vector<std::vector<char>> res;

  std::ifstream fin(filename, std::ifstream::binary);

  while (!fin.eof()) {
    uint64_t chunk_size;
    fin.read(reinterpret_cast<char *>(&chunk_size), sizeof(uint64_t));
    if (fin.eof())
      break;
    res.emplace_back(chunk_size);
    fin.read(reinterpret_cast<char*>(res.back().data()), chunk_size);
  }

  return res;
}


std::vector<std::vector<char>>
multi_file(const std::vector<std::string>& filenames, const size_t chunk_size,
    const bool has_page_sizes, const size_t multiple_of, const size_t duplicate_count)
{
  std::vector<std::vector<char>> split_data;

  for (auto const& filename : filenames) {
    if (!has_page_sizes) {
      std::vector<char> filedata = readFile(filename);
      const size_t filedata_original_size = filedata.size();
      const size_t filedata_padding_size = (multiple_of - (filedata_original_size % multiple_of)) % multiple_of;
      const size_t filedata_padded_size = filedata_original_size + filedata_padding_size;

      const size_t num_chunks
          = (filedata_padded_size + chunk_size - 1) / chunk_size;
      size_t offset = 0;
      for (size_t c = 0; c < num_chunks; ++c) {
        const size_t size_of_this_chunk = std::min(chunk_size, filedata_padded_size-offset);
        std::vector<char> tmp(size_of_this_chunk, 0);
        if(offset < filedata_original_size) {
          std::copy(filedata.data() + offset,
                    filedata.data() + offset + min(filedata_original_size-offset, size_of_this_chunk),
                    tmp.begin());
        }
        split_data.emplace_back(std::move(tmp));

        offset += size_of_this_chunk;
        assert(offset <= filedata_padded_size);
      }
    } else {
      std::vector<std::vector<char>> filedata = readFileWithPageSizes(filename);
      split_data.insert(split_data.end(), filedata.begin(), filedata.end());
    }
  }

  if (duplicate_count > 1) {
    // Make duplicate_count copies of the contents of split_data,
    // but copy into a separate std::vector, to avoid issues with the
    // memory being reallocated while the contents are being copied.
    std::vector<std::vector<char>> duplicated;
    const size_t original_num_chunks = split_data.size();
    duplicated.reserve(original_num_chunks * duplicate_count);
    for (size_t d = 0; d < duplicate_count; ++d) {
      duplicated.insert(duplicated.end(), split_data.begin(), split_data.end());
    }
    // Now that there are duplicate_count copies of split_data in
    // duplicated, swap them, so that they're in split_data.
    duplicated.swap(split_data);
  }

  return split_data;
}
}

template<
    typename CompGetTempT,
    typename CompGetSizeT,
    typename CompAsyncT,
    typename DecompGetTempT,
    typename DecompAsyncT,
    typename DecompGetSizeT,
    typename IsInputValidT,
    typename FormatOptsT>
void
run_benchmark_template(
    CompGetTempT BatchedCompressGetTempSize,
    CompGetSizeT BatchedCompressGetMaxOutputChunkSize,
    CompAsyncT BatchedCompressAsync,
    DecompGetTempT BatchedDecompressGetTempSize,
    DecompAsyncT BatchedDecompressAsync,
    DecompGetSizeT BatchedDecompressGetSize,
    IsInputValidT IsInputValid,
    const FormatOptsT format_opts,
    const std::vector<std::vector<char>>& data,
    const bool warmup,
    const size_t count,
    const bool csv_output,
    const bool use_tabs,
    const size_t duplicate_count,
    const size_t num_files,
    const bool compressed_inputs = false,
    const bool single_output_buffer = false,
    const bool file_output = false,
    const std::string output_filename = "")
{
  benchmark_assert(IsInputValid(data), "Invalid input data");

  const std::string separator = use_tabs ? "\t" : ",";

  size_t total_bytes = 0;
  size_t chunk_size = 0;
  for (const std::vector<char>& part : data) {
    total_bytes += part.size();
    if (part.size() > chunk_size) {
      chunk_size = part.size();
    }
  }
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  
  // build up metadata
  BatchData input_data(data);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const size_t batch_size = input_data.size();

  std::vector<size_t> h_input_sizes(batch_size);
  CUDA_CHECK(cudaMemcpy(h_input_sizes.data(), input_data.sizes(),
      sizeof(size_t)*batch_size, cudaMemcpyDeviceToHost));

  size_t compressed_size = 0;
  double comp_time = 0.0;
  double decomp_time = 0.0;
  for (size_t iter = 0; iter < count; ++iter) {
    // compression
    nvcompStatus_t status;
    size_t comp_bytes = 0;
    float compress_ms = 0;

    size_t max_out_bytes;
    if(compressed_inputs){
      max_out_bytes = chunk_size;
    }else{
      status = BatchedCompressGetMaxOutputChunkSize(
        chunk_size, format_opts, &max_out_bytes);
      benchmark_assert(status == nvcompSuccess,
        "BatchedGetMaxOutputChunkSize() failed.");
    }

    size_t* d_decomp_sizes;
    CUDA_CHECK(cudaMalloc(
        &d_decomp_sizes, batch_size*sizeof(*d_decomp_sizes)));
    size_t* d_uncomp_sizes;
    CUDA_CHECK(cudaMalloc(
        &d_uncomp_sizes, batch_size*sizeof(*d_uncomp_sizes)));
    std::vector<size_t> h_ucomp_sizes(batch_size);
    
    BatchData compress_data(max_out_bytes, batch_size); 
    if(!compressed_inputs){
      // Compress on the GPU using batched API
      size_t comp_temp_bytes;
      status = BatchedCompressGetTempSize(
          batch_size, chunk_size, format_opts, &comp_temp_bytes);
      benchmark_assert(status == nvcompSuccess,
          "BatchedCompressGetTempSize() failed.");

      void* d_comp_temp;
      CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

      CUDA_CHECK(cudaEventRecord(start, stream));

      status = BatchedCompressAsync(
          input_data.ptrs(),
          input_data.sizes(),
          chunk_size,
          batch_size,
          d_comp_temp,
          comp_temp_bytes,
          compress_data.ptrs(),
          compress_data.sizes(),
          format_opts,
          stream);
      benchmark_assert(status == nvcompSuccess,
          "BatchedCompressAsync() failed.");

      CUDA_CHECK(cudaEventRecord(end, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      // free compression memory
      CUDA_CHECK(cudaFree(d_comp_temp));

      CUDA_CHECK(cudaEventElapsedTime(&compress_ms, start, end));

      // compute compression ratio
      std::vector<size_t> compressed_sizes_host(compress_data.size());
      CUDA_CHECK(cudaMemcpy(
          compressed_sizes_host.data(),
          compress_data.sizes(),
          compress_data.size() * sizeof(*compress_data.sizes()),
          cudaMemcpyDeviceToHost));
      for (size_t ix = 0 ; ix < batch_size; ++ix) {
        comp_bytes += compressed_sizes_host[ix];
      }

      h_ucomp_sizes = h_input_sizes;
      CUDA_CHECK(cudaMemcpy(d_uncomp_sizes, h_input_sizes.data(), sizeof(size_t) * batch_size, cudaMemcpyDefault));

      // Then do file output
      if (file_output) {
        std::vector<uint8_t> comp_data(comp_bytes);
        std::vector<uint8_t*> comp_ptrs(batch_size);
        cudaMemcpy(comp_ptrs.data(), compress_data.ptrs(), sizeof(size_t) * batch_size, cudaMemcpyDefault);
        size_t ix_offset = 0;
        for (int ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
          cudaMemcpy(&comp_data[ix_offset], comp_ptrs[ix_chunk], compressed_sizes_host[ix_chunk], cudaMemcpyDefault);
          ix_offset += compressed_sizes_host[ix_chunk];
        }

        std::ofstream outfile{output_filename.c_str(), outfile.binary};
        outfile.write(reinterpret_cast<char*>(comp_data.data()), ix_offset);
        outfile.close();
      }
    }else{
      compress_data.load_data(data);
      status = BatchedDecompressGetSize(
          compress_data.ptrs(),
          compress_data.sizes(),
          d_uncomp_sizes,
          batch_size,
          stream);
      benchmark_assert(
        status == nvcompSuccess,
        "BatchedDecompressGetSize() not successful");
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaMemcpy(h_ucomp_sizes.data(), d_uncomp_sizes, sizeof(size_t) * batch_size, cudaMemcpyDefault));
      comp_bytes = total_bytes;
    }
    //Decompression
    size_t decomp_temp_bytes;
    status = BatchedDecompressGetTempSize(
        compress_data.size(), chunk_size, &decomp_temp_bytes);
    benchmark_assert(status == nvcompSuccess,
        "BatchedDecompressGetTempSize() failed.");

    void* d_decomp_temp;
    CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

    nvcompStatus_t* d_decomp_statuses;
    CUDA_CHECK(cudaMalloc(
        &d_decomp_statuses, batch_size*sizeof(*d_decomp_statuses)));

    std::vector<void*> h_output_ptrs(batch_size);
    nvcomp::thrust::device_vector<void*> d_output_ptrs_tight;
    size_t total_uncomp_size = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      total_uncomp_size += h_ucomp_sizes[i];
    }
    nvcomp::thrust::device_vector<uint8_t> one_buffer( total_uncomp_size);
    void ** d_output_ptrs;

    if(single_output_buffer){
      size_t offset = 0;
      for (size_t i = 0; i < batch_size; ++i) {
        h_output_ptrs[i] = static_cast<void*>(one_buffer.data().get() + offset);
        offset += h_ucomp_sizes[i];
      }
      d_output_ptrs_tight = nvcomp::thrust::device_vector<void*>(h_output_ptrs);
      CUDA_CHECK(cudaEventRecord(start, stream));
      status = BatchedDecompressAsync(
          compress_data.ptrs(),
          compress_data.sizes(),
          d_uncomp_sizes,
          d_decomp_sizes,
          batch_size,
          d_decomp_temp,
          decomp_temp_bytes,
          d_output_ptrs_tight.data().get(),
          d_decomp_statuses,
          stream);
      benchmark_assert(
          status == nvcompSuccess,
          "BatchedDecompressAsync() not successful");

      CUDA_CHECK(cudaEventRecord(end, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

    }else{
      size_t gpu_bytes_free, gpu_bytes_total;
      size_t benchmark_out_bytes = 0;
      CUDA_CHECK(cudaMemGetInfo( &gpu_bytes_free, &gpu_bytes_total ));
      for (size_t i = 0; i < batch_size; ++i) {
        benchmark_out_bytes += h_ucomp_sizes[i];
      }

      if(gpu_bytes_free < benchmark_out_bytes){
        std::cerr << "WARNING: Not enough memory. Could not run benchmark." << std::endl;
        std::exit(0);
      }

      for (size_t i = 0; i < batch_size; ++i) {
          CUDA_CHECK(cudaMalloc(&h_output_ptrs[i], h_ucomp_sizes[i]));
      }
      CUDA_CHECK(cudaMalloc(&d_output_ptrs, sizeof(*d_output_ptrs)*batch_size));
      CUDA_CHECK(cudaMemcpy(d_output_ptrs, h_output_ptrs.data(),
          sizeof(*d_output_ptrs)*batch_size, cudaMemcpyHostToDevice));
    
      CUDA_CHECK(cudaEventRecord(start, stream));
      status = BatchedDecompressAsync(
          compress_data.ptrs(),
          compress_data.sizes(),
          d_uncomp_sizes,
          d_decomp_sizes,
          batch_size,
          d_decomp_temp,
          decomp_temp_bytes,
          d_output_ptrs,
          d_decomp_statuses,
          stream);
      benchmark_assert(
          status == nvcompSuccess,
          "BatchedDecompressAsync() not successful");

      CUDA_CHECK(cudaEventRecord(end, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    float decompress_ms;
    CUDA_CHECK(cudaEventElapsedTime(&decompress_ms, start, end));

    // verify success each time
    std::vector<size_t> h_decomp_sizes(batch_size);
    CUDA_CHECK(cudaMemcpy(h_decomp_sizes.data(), d_decomp_sizes,
      sizeof(*d_decomp_sizes)*batch_size, cudaMemcpyDeviceToHost));
    
    std::vector<nvcompStatus_t> h_decomp_statuses(batch_size);
    CUDA_CHECK(cudaMemcpy(h_decomp_statuses.data(), d_decomp_statuses,
      sizeof(*d_decomp_statuses)*batch_size, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < batch_size; ++i) {
      benchmark_assert(h_decomp_statuses[i] == nvcompSuccess, "Batch item not successfuly decompressed: i=" + std::to_string(i) + ": status=" +
      std::to_string(h_decomp_statuses[i]));
      if(!compressed_inputs){
        benchmark_assert(h_decomp_sizes[i] == h_input_sizes[i], "Batch item of wrong size: i=" + std::to_string(i) + ": act_size=" +
        std::to_string(h_decomp_sizes[i]) + " exp_size=" +
        std::to_string(h_input_sizes[i]));
      }
    }

    CUDA_CHECK(cudaFree(d_decomp_temp));
    CUDA_CHECK(cudaFree(d_decomp_statuses));

    // only verify last iteration
    if (iter + 1 == count && !compressed_inputs) {
      std::vector<void*> h_input_ptrs(batch_size);
      CUDA_CHECK(cudaMemcpy(h_input_ptrs.data(), input_data.ptrs(),
          sizeof(void*)*batch_size, cudaMemcpyDeviceToHost));
      for (size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
        std::vector<uint8_t> exp_data(h_input_sizes[ix_chunk]);
        CUDA_CHECK(cudaMemcpy(exp_data.data(), h_input_ptrs[ix_chunk],
            h_input_sizes[ix_chunk], cudaMemcpyDeviceToHost));
        std::vector<uint8_t> act_data(h_decomp_sizes[ix_chunk]);
        CUDA_CHECK(cudaMemcpy(act_data.data(), h_output_ptrs[ix_chunk],
        h_decomp_sizes[ix_chunk], cudaMemcpyDeviceToHost));
        for (size_t ix_byte = 0; ix_byte < h_input_sizes[ix_chunk]; ++ix_byte) {
          if (act_data[ix_byte] != exp_data[ix_byte]) {
            benchmark_assert(false, "Batch item decompressed output did not match input: ix_chunk="+std::to_string(ix_chunk) + ": ix_byte=" + std::to_string(ix_byte) + " act=" + std::to_string(act_data[ix_byte]) + " exp=" +
            std::to_string(exp_data[ix_byte]));
          }
        }
      }
    }

    if(compressed_inputs && file_output){
      total_bytes = 0;
      for (size_t ix = 0 ; ix < batch_size; ++ix) {
        total_bytes += h_decomp_sizes[ix];
      }
      std::vector<uint8_t> uncomp_data(total_bytes);
      size_t ix_offset = 0;
      for (int ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
        CUDA_CHECK(cudaMemcpy(&uncomp_data[ix_offset], h_output_ptrs[ix_chunk],
        h_decomp_sizes[ix_chunk], cudaMemcpyDeviceToHost));
        if(single_output_buffer){
          std::ofstream outfile{output_filename.c_str() + std::to_string(ix_chunk), outfile.binary};
          outfile.write(reinterpret_cast<char*>(&uncomp_data[ix_offset]), h_decomp_sizes[ix_chunk]);
          outfile.close();
        }
        ix_offset += h_decomp_sizes[ix_chunk];
      }
      if(!single_output_buffer){
        std::ofstream outfile{output_filename.c_str(), outfile.binary};
        outfile.write(reinterpret_cast<char*>(uncomp_data.data()), total_bytes);
        outfile.close();
      }
    }

    if(!single_output_buffer){
      CUDA_CHECK(cudaFree(d_output_ptrs));
      for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaFree(h_output_ptrs[i]));
      }
    }
    CUDA_CHECK(cudaFree(d_decomp_sizes));
    CUDA_CHECK(cudaFree(d_uncomp_sizes));

    // count everything from our iteration
    compressed_size += comp_bytes;
    comp_time += compress_ms * 1.0e-3;
    decomp_time += decompress_ms * 1.0e-3;
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  // average iterations
  compressed_size /= count;
  comp_time /= count;
  decomp_time /= count;

  if (!warmup) {
    const double comp_ratio = (double)total_bytes / compressed_size;
    const double compression_throughput_gbs = (double)total_bytes / (1.0e9 *
        comp_time);
    const double decompression_throughput_gbs = (double)total_bytes / (1.0e9 *
        decomp_time);

    if (!csv_output) {
      std::cout << "----------" << std::endl;
      std::cout << "files: " << num_files << std::endl;
      std::cout << "uncompressed (B): " << total_bytes << std::endl;
      std::cout << "comp_size: " << compressed_size 
                << ", compressed ratio: " << std::fixed << std::setprecision(4)
                << comp_ratio << std::endl;
      std::cout << "compression throughput (GB/s): " << compression_throughput_gbs << std::endl;
      std::cout << "decompression throughput (GB/s): " << decompression_throughput_gbs << std::endl;
    } else {
      // header
      std::cout << "Files";
      std::cout << separator << "Duplicate data";
      std::cout << separator << "Size in MB";
      std::cout << separator << "Pages";
      std::cout << separator << "Avg page size in KB";
      std::cout << separator << "Max page size in KB";
      std::cout << separator << "Ucompressed size in bytes";
      std::cout << separator << "Compressed size in bytes";
      std::cout << separator << "Compression ratio";
      std::cout << separator << "Compression throughput (uncompressed) in GB/s";
      std::cout << separator << "Decompression throughput (uncompressed) in GB/s";
      std::cout << std::endl;

      // values
      std::cout << num_files;
      std::cout << separator << duplicate_count;
      std::cout << separator << (total_bytes * 1e-6); // MB
      std::cout << separator << data.size();
      std::cout << separator << ((1e-3*total_bytes) / data.size()); // KB
      std::cout << separator << (1e-3*chunk_size); // KB
      std::cout << separator << total_bytes;
      std::cout << separator << compressed_size;
      std::cout << separator << std::fixed << std::setprecision(2)
                << comp_ratio;
      std::cout << separator << compression_throughput_gbs;
      std::cout << separator << decompression_throughput_gbs;
      std::cout << std::endl;
    }
  }
}

void run_benchmark(
    const std::vector<std::vector<char>>& data,
    const bool warmup,
    const size_t count,
    const bool csv_output,
    const bool tab_separator,
    const size_t duplicate_count,
    const size_t num_files,
    const bool compressed_inputs,
    const bool single_output_buffer);

struct args_type {
  int gpu;
  std::vector<std::string> filenames;
  size_t warmup_count;
  size_t iteration_count;
  // Represents the number of bytes the input data needs to be a multiple of.
  // If it is not the case, the input data will be padded with zeros to satisfy the
  // requirement.
  size_t multiple_of;
  // Indicates the number of times the input data will be duplicated. In case
  // the input data went under some padding to satisfy `multiple_of`, the padded
  // data is duplicated.
  size_t duplicate_count;
  bool csv_output;
  bool use_tabs;
  bool has_page_sizes;
  size_t chunk_size;
  bool compressed_inputs;
  bool single_output_buffer;
};

struct parameter_type {
  std::string short_flag;
  std::string long_flag;
  std::string description;
  std::string default_value;
};

bool parse_bool(const std::string& val)
{
  std::istringstream ss(val);
  std::boolalpha(ss);
  bool x;
  if (!(ss >> x)) {
    std::cerr << "ERROR: Invalid boolean: '" << val << "', only 'true' and 'false' are accepted." << std::endl;
    std::exit(1);
  }
  return x;
}

void usage(const std::string& name, const std::vector<parameter_type>& parameters)
{
  std::cout << "Usage: " << name << " [OPTIONS]" << std::endl;
  for (const parameter_type& parameter : parameters) {
    std::cout << "  -" << parameter.short_flag << ",--" << parameter.long_flag;
    std::cout << "  : " << parameter.description << std::endl;
    if (parameter.default_value.empty()) {
      // no default value
    } else if (parameter.default_value == REQUIRED_PARAMTER) {
      std::cout << "    required" << std::endl;
    } else {
      std::cout << "    default=" << parameter.default_value << std::endl;
    }
  }
}

std::string bool_to_string(const bool b) {
  if (b) {
    return "true";
  } else {
    return "false";
  }
}

args_type parse_args(int argc, char ** argv) {
  args_type args;
  args.gpu = 0;
  args.warmup_count = 1;
  args.iteration_count = 1;
  args.multiple_of = 1;
  args.duplicate_count = 0;
  args.csv_output = false;
  args.use_tabs = false;
  args.has_page_sizes = false;
  args.chunk_size = 65536;
  args.compressed_inputs = false;
  args.single_output_buffer = false;

  const std::vector<parameter_type> params{
    {"?", "help", "Show options.", ""},
    {"g", "gpu", "GPU device number", std::to_string(args.gpu)},
    {"f", "input_file", "The list of inputs files. All files must start "
        "with a character other than '-'", "_required_"},
    {"w", "warmup_count", "The number of warmup iterations to perform.",
        std::to_string(args.warmup_count)},
    {"i", "iteration_count", "The number of runs to average.",
        std::to_string(args.iteration_count)},
    {"m", "multiple_of", "Add padding to the input data such that its "
        "length becomes a multiple of the given argument (in bytes). Only applicable to "
        "data without page sizes.",
        std::to_string(args.multiple_of)},
    {"x", "duplicate_data", "Clone uncompressed chunks multiple times.",
        std::to_string(args.duplicate_count)},
    {"c", "csv_output", "Output in column/csv format.",
        bool_to_string(args.csv_output)},
    {"e", "tab_separator", "Use tabs instead of commas when "
        "'--csv_output' is specificed.",
        bool_to_string(args.use_tabs)},
    {"s", "file_with_page_sizes", "File(s) contain pages, each prefixed "
        "with int64 size.", bool_to_string(args.has_page_sizes)},
    {"p", "chunk_size", "Chunk size when splitting uncompressed data.",
        std::to_string(args.chunk_size)},
    {"compressed", "compressed_inputs", "The input dataset is compressed.",
        std::to_string(args.compressed_inputs)},
    {"single", "single_output_buffer", "There is only one tight output buffer.",
        std::to_string(args.single_output_buffer)},
  };

  char** argv_end = argv + argc;
  const std::string name(argv[0]);
  argv += 1;

  while (argv != argv_end) {
    std::string arg(*(argv++));
    bool found = false;
    for (const parameter_type& param : params) {
      if (arg == "-" + param.short_flag || arg == "--" + param.long_flag) {
        found = true;

        // found the parameter
        if (param.long_flag == "help") {
          usage(name, params);
          std::exit(0);
        }

        // everything from here on out requires an extra parameter
        if (argv >= argv_end) {
          std::cerr << "ERROR: Missing argument" << std::endl;
          usage(name, params);
          std::exit(1);
        }

        if (param.long_flag == "gpu") {
          args.gpu = std::stol(*(argv++));
          break;
        } else if (param.long_flag == "input_file") {
          // read all following arguments until a new flag is found
          char ** next_argv_ptr = argv;
          while (next_argv_ptr < argv_end && (*next_argv_ptr)[0] != '-') {
            args.filenames.emplace_back(*next_argv_ptr);
            next_argv_ptr = ++argv;
          }
          break;
        } else if (param.long_flag == "warmup_count") {
          args.warmup_count = size_t(std::stoull(*(argv++)));
          break;
        } else if (param.long_flag == "iteration_count") {
          args.iteration_count = size_t(std::stoull(*(argv++)));
          break;
        } else if (param.long_flag == "multiple_of") {
          args.multiple_of = size_t(std::stoull(*(argv++)));
          break;
        } else if (param.long_flag == "duplicate_data") {
          args.duplicate_count = size_t(std::stoull(*(argv++)));
          break;
        } else if (param.long_flag == "csv_output") {
          std::string on(*(argv++));
          args.csv_output = parse_bool(on);
          break;
        } else if (param.long_flag == "tab_separator") {
          std::string on(*(argv++));
          args.use_tabs = parse_bool(on);
          break;
        } else if (param.long_flag == "file_with_page_sizes") {
          std::string on(*(argv++));
          args.has_page_sizes = parse_bool(on);
          break;
        } else if (param.long_flag == "chunk_size") {
          args.chunk_size = size_t(std::stoull(*(argv++)));
          break;
        } else if (param.long_flag == "compressed_inputs") {
          std::string on(*(argv++));
          args.compressed_inputs = parse_bool(on);
          break;
        } else if (param.long_flag == "single_output_buffer") {
          std::string on(*(argv++));
          args.single_output_buffer = parse_bool(on);
          break;
        } else {
          std::cerr << "INTERNAL ERROR: Unhandled paramter '" << arg << "'." << std::endl;
          usage(name, params);
          std::exit(1);
        }
      }
    }
    size_t argumentsUsed = 0;
    if (!found && !handleCommandLineArgument(arg, argv, argumentsUsed)) {
      std::cerr << "ERROR: Unknown argument '" << arg << "'." << std::endl;
      usage(name, params);
      std::exit(1);
    }
    argv += argumentsUsed;
  }

  if (args.filenames.empty()) {
    std::cerr << "ERROR: Must specify at least one input file." << std::endl;
    std::exit(1);
  }

  if(args.multiple_of > 1 && args.has_page_sizes) {
    std::cerr << "ERROR: If the input files contain page sizes, the argument 'multiple_of' is not permitted." << std::endl;
    std::exit(1);
  }

  return args;
}

int main(int argc, char** argv)
{
  args_type args = parse_args(argc, argv);

  CUDA_CHECK(cudaSetDevice(args.gpu));

  auto data = multi_file(args.filenames, args.chunk_size, args.has_page_sizes, args.multiple_of,
      args.duplicate_count);

  // one warmup to allow cuda to initialize
  run_benchmark(data, true, args.warmup_count, false, false,
      args.duplicate_count, args.filenames.size(), args.compressed_inputs, args.single_output_buffer);

  // second run to report times
  run_benchmark(data, false, args.iteration_count, args.csv_output,
      args.use_tabs, args.duplicate_count, args.filenames.size(), args.compressed_inputs, args.single_output_buffer);

  return 0;
}

#endif

