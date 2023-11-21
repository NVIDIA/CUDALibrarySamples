/*  
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 * 
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name(s) of the copyright holder(s) nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */  

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <cuda.h>
#include <cuda_fp16.hpp>

#include "../../einsum.h"

template<>
struct CuTensorTypeTraits<at::Half> {
  static cutensorDataType_t getDataType() {return CUTENSOR_R_16F;}
  static const cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_16F;}
  typedef float ScalarType;
};

template<>
struct CuTensorTypeTraits<at::BFloat16> {
  static cutensorDataType_t getDataType() {return CUTENSOR_R_16BF;}
  static const cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_16BF;}
  typedef float ScalarType;
};

template<>
struct CuTensorTypeTraits<c10::complex<float>> {
  static cutensorDataType_t getDataType() {return CUTENSOR_C_32F;}
  static const cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_32F;}
  typedef c10::complex<float> ScalarType;
};

template<>
struct CuTensorTypeTraits<c10::complex<double>> {
  static cutensorDataType_t getDataType() {return CUTENSOR_C_64F;}
  static const cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_64F;}
  typedef c10::complex<double> ScalarType;
};

size_t getMaxAvailableMemorySize()
{
    // query the size of reserved and allocated memory in the torch memory pool
    auto current_device = at::cuda::current_device();
    auto device_stats = at::cuda::CUDACachingAllocator::getDeviceStats(current_device);
    auto allocated_bytes = device_stats.allocated_bytes[static_cast<size_t>(at::cuda::CUDACachingAllocator::StatType::AGGREGATE)].current;
    auto reserved_bytes = device_stats.reserved_bytes[static_cast<size_t>(at::cuda::CUDACachingAllocator::StatType::AGGREGATE)].current;

    // cached in torch memory pool
    size_t cached_bytes = reserved_bytes - allocated_bytes;
    size_t maxSize = cached_bytes * 0.9; // 90% of the cached memory

    // freed in device memory
    size_t freed_bytes, total_bytes;
    cudaMemGetInfo(&freed_bytes, &total_bytes);
    maxSize = maxSize < freed_bytes ? freed_bytes : maxSize;
    return maxSize;
}

torch::Tensor einsum(
    std::string subscripts,
    torch::Tensor input_0,
    torch::Tensor input_1,
    bool conjA = false,
    bool conjB = false
) {
  at::Tensor output_tensor;
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_0.scalar_type(), "einsum", [&] {
    constexpr int kMaxNumModes_ = 64; // maximal number of modes supported by cuTENSOR
    cutensorOperator_t opA = conjA ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    cutensorOperator_t opB = conjB ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    Einsum<scalar_t, int64_t, kMaxNumModes_> myEinsum(subscripts, input_0.sizes().vec(), input_1.sizes().vec(), opA, opB);
    if (!myEinsum.isInitialized()) {
      throw std::runtime_error("cutensor: Initialization failed.");
    }

    output_tensor = torch::empty(myEinsum.getOutputShape(), input_0.options());

    uint64_t worksize_provided = ULLONG_MAX;
    auto ret1 = myEinsum.plan(GetCuTensorHandle(), worksize_provided);
    if (! ret1) throw std::runtime_error("cuTensor: plan creation failed.");
    uint64_t worksize = myEinsum.getWorksize();
    // try to allocate the workspace according to the cuTensor required size
    // if failed, query the available memory size and recreate the plan
    at::Tensor workspace;
    try {
      workspace = at::empty(worksize, at::CUDA(at::kByte));
    } catch (std::exception& e) {
      worksize_provided = getMaxAvailableMemorySize();
      ret1 = myEinsum.plan(GetCuTensorHandle(), worksize_provided);
      if (! ret1) throw std::runtime_error("cuTensor: plan recreation failed.");
      worksize = myEinsum.getWorksize();
      workspace = at::empty(worksize, at::CUDA(at::kByte));
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto ret = myEinsum.execute(GetCuTensorHandle(),
                                input_0.data_ptr<scalar_t>(),
                                input_1.data_ptr<scalar_t>(),
                                output_tensor.data_ptr<scalar_t>(),
                                workspace.data_ptr<uint8_t>(),
                                stream);

    if (! ret) throw std::runtime_error("cutensor: Launch failed.");
  });
  return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("einsum", &einsum, "Einsum");
}
