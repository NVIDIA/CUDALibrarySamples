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

#include "util.h"

class BatchData;

class BatchDataCPU
{
public:
  BatchDataCPU(
      const std::vector<std::vector<char>>& host_data,
      const size_t chunk_size) :
      m_ptrs(),
      m_sizes(),
      m_data(),
      m_size(0)
  {
    m_size = compute_batch_size(host_data, chunk_size);
    m_sizes = compute_chunk_sizes(host_data, m_size, chunk_size);

    size_t data_size = std::accumulate(
        m_sizes.begin(), m_sizes.end(), static_cast<size_t>(0));
    m_data = std::vector<uint8_t>(data_size);

    size_t offset = 0;
    m_ptrs = std::vector<void*>(size());
    for (size_t i = 0; i < size(); ++i) {
      m_ptrs[i] = data() + offset;
      offset += m_sizes[i];
    }

    std::vector<void*> src = get_input_ptrs(host_data, size(), chunk_size);
    for (size_t i = 0; i < size(); ++i)
      std::memcpy(m_ptrs[i], src[i], m_sizes[i]);
  }

  BatchDataCPU(const size_t max_output_size, const size_t batch_size) :
      m_ptrs(),
      m_sizes(),
      m_data(),
      m_size(batch_size)
  {
    m_data = std::vector<uint8_t>(max_output_size * size());

    m_sizes = std::vector<size_t>(size(), max_output_size);

    m_ptrs = std::vector<void*>(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      m_ptrs[i] = data() + max_output_size * i;
    }
  }

  // Copy Batchdata from GPU to CPU, or allocte output space based on GPU data.
  BatchDataCPU(
      const void* const* in_ptrs, // device pointer
      const size_t* in_sizes,     // device pointer
      const uint8_t* in_data,     // device pointer
      size_t in_size,
      bool copy_data = false) :
      m_ptrs(),
      m_sizes(),
      m_data(),
      m_size(in_size)
  {
    m_sizes = std::vector<size_t>(size());
    CUDA_CHECK(cudaMemcpy(
        sizes(), in_sizes, size() * sizeof(*in_sizes), cudaMemcpyDeviceToHost));

    size_t data_size
        = std::accumulate(sizes(), sizes() + size(), static_cast<size_t>(0));
    m_data = std::vector<uint8_t>(data_size);

    size_t offset = 0;
    m_ptrs = std::vector<void*>(size());
    for (size_t i = 0; i < size(); ++i) {
      m_ptrs[i] = data() + offset;
      offset += sizes()[i];
    }
    if (copy_data) {
      std::vector<void*> hs_ptrs(size());
      CUDA_CHECK(cudaMemcpy(
          hs_ptrs.data(),
          in_ptrs,
          size() * sizeof(void*),
          cudaMemcpyDeviceToHost));

      for (size_t i = 0; i < size(); ++i) {
        const uint8_t* rhptr = reinterpret_cast<const uint8_t*>(hs_ptrs[i]);
        CUDA_CHECK(
            cudaMemcpy(m_ptrs[i], rhptr, m_sizes[i], cudaMemcpyDeviceToHost));
      }
    }
  }

  BatchDataCPU(BatchDataCPU&& other) = default;
  BatchDataCPU(const BatchData& batch_data, bool copy_data = false);
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

  void** ptrs()
  {
    return m_ptrs.data();
  }
  const void* const* ptrs() const
  {
    return m_ptrs.data();
  }

  size_t* sizes()
  {
    return m_sizes.data();
  }
  const size_t* sizes() const
  {
    return m_sizes.data();
  }

  size_t size() const
  {
    return m_size;
  }

private:
  std::vector<void*> m_ptrs;
  std::vector<size_t> m_sizes;
  std::vector<uint8_t> m_data;
  size_t m_size;
};

inline bool operator==(const BatchDataCPU& lhs, const BatchDataCPU& rhs)
{
  size_t batch_size = lhs.size();

  if (lhs.size() != rhs.size())
    return false;

  for (size_t i = 0; i < batch_size; ++i) {
    if (lhs.sizes()[i] != rhs.sizes()[i])
      return false;

    const uint8_t* lhs_ptr = reinterpret_cast<const uint8_t*>(lhs.ptrs()[i]);
    const uint8_t* rhs_ptr = reinterpret_cast<const uint8_t*>(rhs.ptrs()[i]);
    for (size_t j = 0; j < rhs.sizes()[i]; ++j)
      if (lhs_ptr[j] != rhs_ptr[j])
        return false;
  }
  return true;
}