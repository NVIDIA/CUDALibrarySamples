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

#include "nvcomp.hpp"
#include "nvcomp/cascaded.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
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
static void cudaMallocSafe(T** devPtr, size_t size)
{
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(devPtr), size);
  if (err == cudaErrorMemoryAllocation) {
    size_t gpu_bytes_free, gpu_bytes_total;
    CUDA_CHECK(cudaMemGetInfo(&gpu_bytes_free, &gpu_bytes_total));
    if (gpu_bytes_free < size)
    {
      std::cerr << "WARNING: Cannot fit data in GPU memory. Bytes requested: " << size <<
        " > bytes_available: " << gpu_bytes_free << ". Could not run benchmark." << std::endl;
    }
    std::exit(3);
  }
  else if (err != cudaSuccess)
  {
    CUDA_CHECK(err);
  }
}

} // namespace nvcomp