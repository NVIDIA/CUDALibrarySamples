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

#pragma once

#include <numeric>

#include "util.hpp"

class BatchDataCPU
{
public:
  BatchDataCPU(
      const std::vector<std::vector<char>>& host_data,
      const size_t chunk_size) :
      m_chunk_ptrs(),
      m_chunk_sizes(),
      m_data(),
      m_batch_size(compute_batch_size(host_data, chunk_size))
  {
    m_chunk_sizes = compute_chunk_sizes(host_data, m_batch_size, chunk_size);

    size_t data_size = std::accumulate(
        m_chunk_sizes.begin(), m_chunk_sizes.end(), size_t(0));
    m_data = std::vector<uint8_t>(data_size);

    size_t offset = 0;
    m_chunk_ptrs = std::vector<void*>(m_batch_size);
    for (size_t i = 0; i < m_batch_size; ++i) {
      m_chunk_ptrs[i] = data() + offset;
      offset += m_chunk_sizes[i];
    }

    std::vector<const void*> src = compute_input_ptrs(host_data, m_batch_size, chunk_size);
    for (size_t i = 0; i < m_batch_size; ++i) {
      std::memcpy(m_chunk_ptrs[i], src[i], m_chunk_sizes[i]);
    }
  }

  BatchDataCPU(const size_t max_output_size, const size_t batch_size) :
      m_chunk_ptrs(),
      m_chunk_sizes(),
      m_data(),
      m_batch_size(batch_size)
  {
    m_data = std::vector<uint8_t>(max_output_size * m_batch_size);

    m_chunk_sizes = std::vector<size_t>(m_batch_size, max_output_size);

    m_chunk_ptrs = std::vector<void*>(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      m_chunk_ptrs[i] = data() + max_output_size * i;
    }
  }

  BatchDataCPU(
      const void* const* device_chunk_ptrs,
      const size_t* device_chunk_sizes,
      size_t batch_size,
      bool copy_data = false) :
      m_chunk_ptrs(),
      m_chunk_sizes(),
      m_data(),
      m_batch_size(batch_size)
  {
    m_chunk_sizes = std::vector<size_t>(m_batch_size);
    CUDA_CHECK(cudaMemcpy(
        chunk_sizes(), device_chunk_sizes, m_batch_size * sizeof(size_t), cudaMemcpyDeviceToHost));

    size_t data_size
        = std::accumulate(chunk_sizes(), chunk_sizes() + m_batch_size, size_t(0));
    m_data = std::vector<uint8_t>(data_size);

    size_t offset = 0;
    m_chunk_ptrs = std::vector<void*>(m_batch_size);
    for (size_t i = 0; i < m_batch_size; ++i) {
      m_chunk_ptrs[i] = data() + offset;
      offset += chunk_sizes()[i];
    }

    if (copy_data) {
      std::vector<void*> host_chunk_ptrs(m_batch_size);
      CUDA_CHECK(cudaMemcpy(
          host_chunk_ptrs.data(),
          device_chunk_ptrs,
          m_batch_size * sizeof(void*),
          cudaMemcpyDeviceToHost));

      for (size_t i = 0; i < m_batch_size; ++i) {
        const uint8_t* chunk_ptr = reinterpret_cast<const uint8_t*>(host_chunk_ptrs[i]);
        CUDA_CHECK(
            cudaMemcpy(m_chunk_ptrs[i], chunk_ptr, m_chunk_sizes[i], cudaMemcpyDeviceToHost));
      }
    }
  }

  BatchDataCPU(BatchDataCPU&& other) = default;

  // disable copying
  BatchDataCPU(const BatchDataCPU& other) = delete;
  BatchDataCPU& operator=(const BatchDataCPU& other) = delete;

  uint8_t* data()
  {
    return m_data.data();
  }

  const uint8_t* data() const
  {
    return m_data.data();
  }

  void** chunk_ptrs()
  {
    return m_chunk_ptrs.data();
  }

  const void* const* chunk_ptrs() const
  {
    return m_chunk_ptrs.data();
  }

  size_t* chunk_sizes()
  {
    return m_chunk_sizes.data();
  }

  const size_t* chunk_sizes() const
  {
    return m_chunk_sizes.data();
  }

  size_t batch_size() const noexcept
  {
    return m_batch_size;
  }

  bool operator==(const BatchDataCPU& other)
  {
    if (m_batch_size != other.batch_size()) {
      return false;
    }

    for (size_t i = 0; i < m_batch_size; ++i) {
      if (m_chunk_sizes[i] != other.chunk_sizes()[i]) {
        return false;
      }

      const uint8_t* this_chunk_ptr = reinterpret_cast<const uint8_t*>(m_chunk_ptrs[i]);
      const uint8_t* other_chunk_ptr = reinterpret_cast<const uint8_t*>(other.chunk_ptrs()[i]);
      for (size_t j = 0; j < m_chunk_sizes[i]; ++j) {
        if (this_chunk_ptr[j] != other_chunk_ptr[j]) {
          return false;
        }
      }
    }
    return true;
  }

  bool operator!=(const BatchDataCPU& other)
  {
    return !(*this == other);
  }
private:
  std::vector<void*> m_chunk_ptrs;
  std::vector<size_t> m_chunk_sizes;
  std::vector<uint8_t> m_data;
  size_t m_batch_size;
};
