/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef VERBOSE
#define VERBOSE 0
#endif

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "nvcomp.hpp"
#include "nvcomp/cascaded.h"


#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0);

#define BTCHK(func)                                                            \
  do {                                                                         \
    bitcompResult_t rt = (func);                                               \
    if (rt != BITCOMP_SUCCESS) {                                               \
      std::cout << "Bitcomp API call failure \"" #func "\" with " << rt        \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0);

namespace nvcomp
{

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

// returns nano-seconds
inline uint64_t get_time(timespec start, timespec end)
{
  constexpr const uint64_t BILLION = 1000000000ULL;
  const uint64_t elapsed_time
      = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  return elapsed_time;
}

// size in bytes, returns GB/s
inline double gibs(struct timespec start, struct timespec end, size_t s)
{
  uint64_t t = get_time(start, end);
  return (double)s / t * 1e9 / 1024 / 1024 / 1024;
}

// size in bytes, returns GB/s
inline double
gbs(const std::chrono::time_point<std::chrono::steady_clock>& start,
    const std::chrono::time_point<std::chrono::steady_clock>& end,
    size_t s)
{
  return (double)s / std::chrono::nanoseconds(end - start).count();
}

inline double
gbs(const std::chrono::nanoseconds duration,
    size_t s)
{
  return (double)s / duration.count();
}

inline double
average_gbs(
    const std::vector<std::chrono::nanoseconds>& durations,
    size_t s)
{
  size_t count_sum = 0;
  for (auto duration : durations) {
    count_sum += duration.count();
  }

  size_t avg_duration = count_sum / durations.size();

  return (double)s / avg_duration;
}

inline double
average_gbs(
    const std::vector<float>& durations,
    size_t s)
{
  double duration_sum = 0;
  for (auto duration : durations) {
    duration_sum += duration;
  }

  double avg_duration = (duration_sum / durations.size()) / 1e3;

  return (double)s / 1e9 / avg_duration;
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

template <>
inline nvcompType_t TypeOf<float>()
{
  return NVCOMP_TYPE_INT;
}

inline bool startsWith(const std::string input, const std::string subStr)
{
  return input.substr(0, subStr.length()) == subStr;
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

void benchmark_assert(const bool pass, const std::string& msg)
{
  if (!pass) {
    printf("unhandled exception in benchmark, msg %s\n", msg.c_str());
    throw std::runtime_error("ERROR: " + msg);
  }
}

std::vector<uint8_t>
gen_data(const int max_byte, const size_t size, std::mt19937& rng)
{
  if (max_byte < 0
      || max_byte > static_cast<int>(std::numeric_limits<uint8_t>::max())) {
    throw std::runtime_error("Invalid byte value: " + std::to_string(max_byte));
  }

  std::uniform_int_distribution<uint16_t> dist(0, max_byte);

  std::vector<uint8_t> data;

  for (size_t i = 0; i < size; ++i) {
    data.emplace_back(static_cast<uint8_t>(dist(rng) & 0xff));
  }

  return data;
}

// Load dataset from binary file into a vector of type T.
template <typename T>
std::vector<T> load_dataset_from_binary(const char* fname, size_t* input_element_count)
{
  try {
    assert(fname != nullptr);
    std::ifstream file;

    // Make stream errors throw std::ios_base::failure.
    file.exceptions(std::ios::badbit | std::ios::failbit);

    file.open(fname, std::ios::binary);

    // find length of file
    file.seekg(0, std::ios::end);
    const auto filelen = std::streamoff(file.tellg());
    assert(filelen >= 0);
    file.seekg(0, std::ios::beg);

    const std::size_t max_input_element_count = [&]() {
      auto miec = filelen / sizeof(T);
      assert(miec < std::numeric_limits<std::size_t>::max());
      return std::size_t(miec);
    }();

    // If input_element_count is already set and is less than the number of elements in
    // the file, use it, otherwise load the whole file.
    if (*input_element_count == 0 || *input_element_count > max_input_element_count) {
      *input_element_count = max_input_element_count;
    }

    const std::streamsize num_bytes = [&]() {
      // We know the result fits in std::streamoff because it is
      // less or equal to filelen.
      auto nb = std::streamoff(*input_element_count) * std::streamoff(sizeof(T));
      assert(nb < std::numeric_limits<std::streamsize>::max());
      return std::streamsize(nb);
    }();

    // Construct buffer and read binary file into it.
    std::vector<T> buffer(*input_element_count);
    file.read(reinterpret_cast<char *>(buffer.data()), num_bytes);
    return buffer;
  }
  catch (const std::ios_base::failure& e) {
    std::cerr << "Error processing binary input file " << fname << ": " << e.what() << std::endl;
    std::exit(1);
    // Dummy return statement to quell compiler warnings.
    return {};
  }
}

template<typename T>
static cudaError_t cudaMallocSafe(T** devPtr, size_t size)
{
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(devPtr), size);
  if (err == cudaErrorMemoryAllocation)
  {
    // Attempt to get memory information
    size_t gpu_bytes_free, gpu_bytes_total;
    cudaError_t err_meminfo = cudaMemGetInfo(&gpu_bytes_free, &gpu_bytes_total);
    if(err_meminfo != cudaSuccess)
    {
      return err_meminfo;
    }

    if (gpu_bytes_free < size)
    {
      std::cerr << "WARNING: Cannot fit data in GPU memory. Bytes requested: " << size
                << " > bytes available: " << gpu_bytes_free << ". Could not run benchmark."
                << std::endl;
      std::exit(3);
    }
  }
  return err;
}

// Read binary file into a vector of char
inline std::vector<char> readFile(const std::string& filename)
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

// Multi-file processing with chunking and duplication support
inline std::vector<std::vector<char>>
multi_file(const std::vector<std::string>& filenames,
    const bool perform_chunking, const size_t chunk_size,
    const size_t multiple_of, const size_t duplicate_count)
{
  std::vector<std::vector<char>> split_data;

  for (auto const& filename : filenames) {
    std::vector<char> filedata = readFile(filename);
    const size_t filedata_original_size = filedata.size();
    const size_t filedata_padding_size = (multiple_of - (filedata_original_size % multiple_of)) % multiple_of;
    const size_t filedata_padded_size = filedata_original_size + filedata_padding_size;

    if (perform_chunking) {
      const size_t num_chunks
          = (filedata_padded_size + chunk_size - 1) / chunk_size;
      size_t offset = 0;
      for (size_t c = 0; c < num_chunks; ++c) {
        const size_t size_of_this_chunk = std::min(chunk_size, filedata_padded_size-offset);
        std::vector<char> tmp(size_of_this_chunk, 0);
        if(offset < filedata_original_size) {
          std::copy(filedata.data() + offset,
                    filedata.data() + offset + std::min(filedata_original_size-offset, size_of_this_chunk),
                    tmp.begin());
        }
        split_data.emplace_back(std::move(tmp));

        offset += size_of_this_chunk;
        assert(offset <= filedata_padded_size);
      }
    } else {
       split_data.emplace_back(filedata);
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

void verify_lossy_tolerance(
  const std::vector<void*>& h_input_ptrs,
  const std::vector<void*>& h_output_ptrs,
  const std::vector<size_t>& h_input_sizes,
  double delta,
  int fp_bits,
  bool is_data_on_device = true)
{
  const double error_tolerance = 0.5 * delta; // theoretical max error for Bitcomp quantization
  const size_t batch_size = h_input_ptrs.size();

  for (size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
    const size_t nbytes = h_input_sizes[ix_chunk];

    auto copy_to_host = [](void* dst, const void* src, size_t bytes, bool from_device) {
      if (from_device) {
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
      } else {
        std::memcpy(dst, src, bytes);
      }
    };

    if (fp_bits == 16) {
      // FP16 case
      const size_t nelems = nbytes / sizeof(uint16_t);
      std::vector<uint16_t> exp_data(nelems);
      copy_to_host(exp_data.data(), h_input_ptrs[ix_chunk], nbytes, is_data_on_device);
      std::vector<uint16_t> act_data(nelems);
      copy_to_host(act_data.data(), h_output_ptrs[ix_chunk], nbytes, is_data_on_device);

      for (size_t iel = 0; iel < nelems; ++iel) {
        float x = __half2float(*reinterpret_cast<const __half*>(&exp_data[iel]));
        float xr = __half2float(*reinterpret_cast<const __half*>(&act_data[iel]));
        if (std::isnan(x) || std::isinf(x)) continue;
        if (std::fabs(xr - x) > error_tolerance) {
          benchmark_assert(false, "Lossy tolerance check failed: ix_chunk=" + std::to_string(ix_chunk) +
            " ix_elem=" + std::to_string(iel) + " x=" + std::to_string(x) + " xr=" + std::to_string(xr) + " tol=" + std::to_string(error_tolerance));
        }
      }
    } else if (fp_bits == 32) {
      // FP32 case
      const size_t nelems = nbytes / sizeof(float);
      std::vector<float> exp_data(nelems);
      copy_to_host(exp_data.data(), h_input_ptrs[ix_chunk], nbytes, is_data_on_device);
      std::vector<float> act_data(nelems);
      copy_to_host(act_data.data(), h_output_ptrs[ix_chunk], nbytes, is_data_on_device);

      for (size_t iel = 0; iel < nelems; ++iel) {
        const float& x = exp_data[iel];
        const float& xr = act_data[iel];
        if (std::isnan(x) || std::isinf(x)) continue;
        if (std::fabs(xr - x) > error_tolerance) {
          benchmark_assert(false, "Lossy tolerance check failed: ix_chunk=" + std::to_string(ix_chunk) +
            " ix_elem=" + std::to_string(iel) + " x=" + std::to_string(x) + " xr=" + std::to_string(xr) + " tol=" + std::to_string(error_tolerance));
        }
      }
    } else if (fp_bits == 64) {
      // FP64 case
      const size_t nelems = nbytes / sizeof(double);
      std::vector<double> exp_data(nelems);
      copy_to_host(exp_data.data(), h_input_ptrs[ix_chunk], nbytes, is_data_on_device);
      std::vector<double> act_data(nelems);
      copy_to_host(act_data.data(), h_output_ptrs[ix_chunk], nbytes, is_data_on_device);

      for (size_t iel = 0; iel < nelems; ++iel) {
        const double& x = exp_data[iel];
        const double& xr = act_data[iel];
        if (std::isnan(x) || std::isinf(x)) continue;
        if (std::fabs(xr - x) > error_tolerance) {
          benchmark_assert(false, "Lossy tolerance check failed: ix_chunk=" + std::to_string(ix_chunk) +
            " ix_elem=" + std::to_string(iel) + " x=" + std::to_string(x) + " xr=" + std::to_string(xr) + " tol=" + std::to_string(error_tolerance));
        }
      }
    } else {
      benchmark_assert(false, "Unsupported floating point precision: " + std::to_string(fp_bits));
    }
  }
}

} // namespace nvcomp
