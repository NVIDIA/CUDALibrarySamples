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

#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string.h>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)
#endif // CUDA_CHECK

size_t compute_batch_size(
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
    offset += num_chunks;
    if (data[i].size() % chunk_size != 0) {
      sizes[offset - 1] = data[i].size() % chunk_size;
    }
  }
  return sizes;
}

std::vector<void*> get_input_ptrs(
    const std::vector<std::vector<char>>& data,
    const size_t batch_size,
    const size_t chunk_size)
{
  std::vector<void*> input_ptrs(batch_size);
  size_t chunk = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
    for (size_t j = 0; j < num_chunks; ++j)
      input_ptrs[chunk++] = const_cast<void*>(
          static_cast<const void*>(data[i].data() + j * chunk_size));
  }
  return input_ptrs;
}

std::vector<char> read_file(const std::string& filename)
{
  std::ifstream fin(filename, std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
  if (!fin) {
    throw std::runtime_error("Unable to open file: " + filename);
  }
  fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  // Query size
  size_t size = fin.tellg();
  fin.seekg(0, std::ifstream::beg);

  // Read the file
  std::vector<char> host_data(size);
  fin.read(host_data.data(), size);

  return host_data;
}

std::vector<std::vector<char>> multi_file(const std::vector<std::string>& filenames)
{
  std::vector<std::vector<char>> split_data;

  for (auto const& filename : filenames) {
    split_data.emplace_back(read_file(filename));
  }

  return split_data;
}

template <typename U, typename T>
U roundUpTo(const U num, const T unit)
{
  return ((num + unit - 1) / unit) * unit;
}
