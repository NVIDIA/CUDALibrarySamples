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

#pragma once

#ifndef VERBOSE
#define VERBOSE 0
#endif

#include "nvcomp.hpp"
#include "nvcomp/cascaded.h"

#if defined(_WIN32)
#undef max
#undef min
#endif

#include <chrono>
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
      throw;                                                                   \
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

// Load dataset from binary file into an array of type T
template <typename T>
std::vector<T> load_dataset_from_binary(char* fname, size_t* input_element_count)
{
  FILE* fileptr = fopen(fname, "rb");

  if (fileptr == NULL) {
    printf("Binary input file not found.\n");
    exit(1);
  }

  // find length of file
  fseek(fileptr, 0, SEEK_END);
  size_t filelen = ftell(fileptr);
  rewind(fileptr);

  // If input_element_count is already set, use it, otherwise load the whole file
  if (*input_element_count == 0 || filelen / sizeof(T) < *input_element_count) {
    *input_element_count = filelen / sizeof(T);
  }

  const size_t numElements = *input_element_count;

  std::vector<T> buffer(numElements);

  // Read binary file in to buffer
  const size_t numRead = fread(buffer.data(), sizeof(T), numElements, fileptr);
  if (numRead != numElements) {
    throw std::runtime_error(
        "Failed to read file: " + std::string(fname) + " read "
        + std::to_string(numRead) + "/"
        + std::to_string(*input_element_count * sizeof(T)) + " elements.");
  }

  fclose(fileptr);
  return buffer;
}

// Load dataset from binary file into an array of type T
template <typename T>
std::vector<T> load_dataset_from_txt(char* fname, size_t* input_element_count)
{

  std::vector<T> buffer;
  FILE* fileptr;

  fileptr = fopen(fname, "rb");

  if (fileptr == NULL) {
    printf("Text input file not found.\n");
    exit(1);
  }

  size_t i = 0;
  constexpr size_t MAX_LINE_LEN = 100;
  char line[MAX_LINE_LEN];
  while (fgets(line, MAX_LINE_LEN, fileptr) && i < *input_element_count) {
    //    std::stringstream row(line);
    buffer.push_back((T)std::stof(line));
    i++;
  }

  fclose(fileptr);

  return buffer;
}

} // namespace nvcomp
