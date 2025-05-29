// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include <stdexcept>

#include <thrust/device_vector.h>

#include "batch_data_cpu.hpp"

class BatchData
{
public:
  BatchData(
      const std::vector<std::vector<char>>& host_data,
      const size_t chunk_size,
      const size_t alignment) :
      m_chunk_ptrs(),
      m_chunk_sizes(),
      m_data(),
      m_batch_size(compute_batch_size(host_data, chunk_size))
  {
    const size_t aligned_chunk_size = nvcompdx::detail::roundUpTo(chunk_size, alignment);
    m_data = thrust::device_vector<uint8_t>(aligned_chunk_size * m_batch_size);

    std::vector<void*> uncompressed_ptrs(m_batch_size);
    for (size_t i = 0; i < m_batch_size; ++i) {
      uncompressed_ptrs[i] = static_cast<void*>(data() + aligned_chunk_size * i);
    }
    m_chunk_ptrs = thrust::device_vector<void*>(uncompressed_ptrs);

    std::vector<size_t> sizes = compute_chunk_sizes(host_data, m_batch_size, chunk_size);
    m_chunk_sizes = thrust::device_vector<size_t>(sizes);

    // copy data to GPU
    size_t offset = 0;
    for (size_t i = 0; i < host_data.size(); ++i) {
      const size_t num_chunks = nvcompdx::detail::roundUpDiv(host_data[i].size(), chunk_size);
      if (aligned_chunk_size == chunk_size) {
        CUDA_CHECK(cudaMemcpy(
            uncompressed_ptrs[offset],
            host_data[i].data(),
            host_data[i].size(),
            cudaMemcpyHostToDevice));
      } else {
        for (size_t j = 0; j < num_chunks; ++j) {
          CUDA_CHECK(cudaMemcpy(
              uncompressed_ptrs[offset + j],
              &host_data[i][j * chunk_size],
              sizes[offset + j],
              cudaMemcpyHostToDevice));
        }
      }

      offset += num_chunks;
    }
  }

  template<typename T>
  BatchData(
    const std::vector<T>& host_data,
    const size_t chunk_size,
    const size_t alignment) :
    m_chunk_ptrs(),
    m_chunk_sizes(),
    m_data(),
    m_batch_size(nvcompdx::detail::roundUpDiv(host_data.size() * sizeof(T), chunk_size))
  {
    if (chunk_size % sizeof(T) != 0) {
      throw std::invalid_argument("chunk_size must be a multiple of the size of the data type");
    }

    const size_t elements_per_chunk = chunk_size / sizeof(T);

    const size_t aligned_chunk_size = nvcompdx::detail::roundUpTo(chunk_size, alignment);
    m_data = thrust::device_vector<uint8_t>(aligned_chunk_size * m_batch_size);

    std::vector<void*> uncompressed_ptrs(m_batch_size);
    for (size_t i = 0; i < m_batch_size; ++i) {
      uncompressed_ptrs[i] = static_cast<void*>(data() + aligned_chunk_size * i);
    }
    m_chunk_ptrs = thrust::device_vector<void*>(uncompressed_ptrs);

    std::vector<size_t> sizes(m_batch_size, chunk_size);
    const size_t total_bytes = host_data.size() * sizeof(T);
    if(total_bytes % chunk_size != 0) {
      sizes.back() = total_bytes % chunk_size;
    }
    m_chunk_sizes = thrust::device_vector<size_t>(sizes);

    // copy data to GPU
    if (aligned_chunk_size == chunk_size) {
      CUDA_CHECK(cudaMemcpy(
        data(),
        host_data.data(),
        total_bytes,
        cudaMemcpyHostToDevice));
    } else {
      for (size_t i = 0; i < m_batch_size; ++i) {
        CUDA_CHECK(cudaMemcpy(
          uncompressed_ptrs[i],
          &host_data[i * elements_per_chunk],
          sizes[i],
          cudaMemcpyHostToDevice));
      }
    }
  }

  BatchData(const BatchDataCPU& batch_data,
            const bool copy_data,
            const size_t alignment) :
      m_chunk_ptrs(),
      m_chunk_sizes(),
      m_data(),
      m_batch_size(batch_data.batch_size())
  {
    m_chunk_sizes = thrust::device_vector<size_t>(
        batch_data.chunk_sizes(), batch_data.chunk_sizes() + m_batch_size);

    size_t data_size = 0;
    for (size_t i = 0; i < m_batch_size; ++i) {
      data_size += nvcompdx::detail::roundUpTo(batch_data.chunk_sizes()[i], alignment);
    }
    m_data = thrust::device_vector<uint8_t>(data_size);

    size_t offset = 0;
    std::vector<void*> ptrs(m_batch_size);
    for (size_t i = 0; i < m_batch_size; ++i) {
      ptrs[i] = data() + offset;
      offset += nvcompdx::detail::roundUpTo(batch_data.chunk_sizes()[i], alignment);
    }
    m_chunk_ptrs = thrust::device_vector<void*>(ptrs);

    if (copy_data) {
      const void* const* src = batch_data.chunk_ptrs();
      const size_t* bytes = batch_data.chunk_sizes();
      for (size_t i = 0; i < m_batch_size; ++i)
        CUDA_CHECK(cudaMemcpy(ptrs[i], src[i], bytes[i], cudaMemcpyHostToDevice));
    }
  }

  BatchData(const size_t max_output_size,
            const size_t batch_size,
            const size_t alignment) :
      m_chunk_ptrs(),
      m_chunk_sizes(),
      m_data(),
      m_batch_size(batch_size)
  {
    const size_t aligned_max_output_size = nvcompdx::detail::roundUpTo(max_output_size, alignment);
    m_data = thrust::device_vector<uint8_t>(aligned_max_output_size * m_batch_size);

    std::vector<size_t> sizes(m_batch_size, aligned_max_output_size);
    m_chunk_sizes = thrust::device_vector<size_t>(sizes);

    std::vector<void*> ptrs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      ptrs[i] = data() + aligned_max_output_size * i;
    }
    m_chunk_ptrs = thrust::device_vector<void*>(ptrs);
  }

  BatchData(BatchData&& other) noexcept = default;

  // disable copying
  BatchData(const BatchData& other) = delete;
  BatchData& operator=(const BatchData& other) = delete;

  uint8_t* data()
  {
    return m_data.data().get();
  }

  const uint8_t* data() const
  {
    return m_data.data().get();
  }

  void** chunk_ptrs()
  {
    return m_chunk_ptrs.data().get();
  }

  const void* const* chunk_ptrs() const
  {
    return m_chunk_ptrs.data().get();
  }

  size_t* chunk_sizes()
  {
    return m_chunk_sizes.data().get();
  }

  const size_t* chunk_sizes() const
  {
    return m_chunk_sizes.data().get();
  }

  size_t batch_size() const noexcept
  {
    return m_batch_size;
  }

  bool operator==(const BatchData& other) const
  {
    BatchDataCPU other_cpu(other.chunk_ptrs(),
                           other.chunk_sizes(),
                           other.batch_size(),
                           true);
    BatchDataCPU this_cpu(chunk_ptrs(),
                          chunk_sizes(),
                          batch_size(),
                          true);
    return other_cpu == this_cpu;
  }

  bool operator!=(const BatchData& other) const
  {
    return !(*this == other);
  }

  bool operator==(const BatchDataCPU& other) const
  {
    BatchDataCPU this_cpu(chunk_ptrs(),
                          chunk_sizes(),
                          batch_size(),
                          true);
    return this_cpu == other;
  }

  bool operator!=(const BatchDataCPU& other) const
  {
    return !(*this == other);
  }
private:
  thrust::device_vector<void*> m_chunk_ptrs;
  thrust::device_vector<size_t> m_chunk_sizes;
  thrust::device_vector<uint8_t> m_data;
  size_t m_batch_size;
};
