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
    auto allocated_bytes = device_stats.allocated_bytes[static_cast<size_t>(c10::CachingDeviceAllocator::StatType::AGGREGATE)].current;
    auto reserved_bytes = device_stats.reserved_bytes[static_cast<size_t>(c10::CachingDeviceAllocator::StatType::AGGREGATE)].current;

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
      std::cerr << "cutensor: Initialization failed." << std::endl;
      throw std::runtime_error("cutensor: Initialization failed.");
    }
    output_tensor = torch::empty(myEinsum.getOutputShape(), input_0.options());

    //get available GPU memory size
    uint64_t worksize_provided = getMaxAvailableMemorySize();
    //plan the einsum kernel
    auto ret1 = myEinsum.plan(GetCuTensorHandle(), worksize_provided, false);
    if (! ret1){
      std::cerr << "cutensor: plan creation failed." << std::endl;
      throw std::runtime_error("cutensor: plan creation failed.");
    }


    //get the estimated workspace size
    uint64_t worksize = myEinsum.getWorksize();

    //try to allocate the workspace according to the cuTensor required size
    at::Tensor workspace;
    try {
      workspace = at::empty(worksize, at::CUDA(at::kByte));
    } 
    catch (std::exception& e) {
        ret1 = myEinsum.plan(GetCuTensorHandle(), 1024ULL * 1024ULL * 1024ULL, false);
        if (! ret1){
          std::cerr << "cutensor: plan with less workspace failed." << std::endl;
          throw std::runtime_error("cutensor: plan creation failed.");
        }
        worksize = myEinsum.getWorksize();
        try {
          workspace = at::empty(worksize, at::CUDA(at::kByte));
        } 
        catch (std::exception& e) {
          std::cerr << "cutensor: error allocating workspace" << std::endl;
          throw std::runtime_error("cutensor: error allocating workspace");
        }
    }

    //launch the einsum kernel
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

 struct EinsumPlan {
   void* myEinsumPtr = nullptr;
   uint64_t worksize = 0;
   torch::Tensor workspace;
   torch::Tensor output_tensor;
   torch::Tensor input_0;
   torch::Tensor input_1;
 };
 
  EinsumPlan plan(
   const std::string& subscripts,
   const torch::Tensor& input_0,
   const torch::Tensor& input_1,
   bool conjA = false,
   bool conjB = false,
   bool jit_pref = false
 ) {
   EinsumPlan new_plan;
   new_plan.input_0 = input_0;
   new_plan.input_1 = input_1;
   AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_0.scalar_type(), "plan", [&] {
       constexpr int kMaxNumModes_ = 64;
       cutensorOperator_t opA = conjA ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
       cutensorOperator_t opB = conjB ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
       Einsum<scalar_t, int64_t, kMaxNumModes_>* myEinsumPtr = new Einsum<scalar_t, int64_t, kMaxNumModes_>(
           subscripts, input_0.sizes().vec(), input_1.sizes().vec(), opA, opB
       );
       new_plan.myEinsumPtr = myEinsumPtr;
       if (!myEinsumPtr->isInitialized()) {
         std::cerr << "cutensor: No Einsum pointer " << std::endl;
         throw std::runtime_error("cutensor: Einsum pointer is NULL");
       }
       uint64_t worksize_provided = getMaxAvailableMemorySize();
       bool ret1;
       
       if (jit_pref) {
         try {
           ret1 = myEinsumPtr->plan(GetCuTensorHandle(), worksize_provided, true);
           if (!ret1){
            ret1 = myEinsumPtr->plan(GetCuTensorHandle(), worksize_provided, true);
            if (!ret1) {
              std::cerr << "cutensor: JIT plan creation failed again" << std::endl;
              throw std::runtime_error("cutensor: JIT plan creation failed again");
            }
           } 
         } 
         catch (const std::exception& e) {
          ret1 = myEinsumPtr->plan(GetCuTensorHandle(), worksize_provided, false); // Always use non-JIT for recovery
          if (!ret1) {
            std::cerr << "Fallback to non-jit planing failed" << std::endl;
            throw std::runtime_error("cutensor: Fallback to non-jit planing failed");
          } 
         }
       }
       else {
         ret1 = myEinsumPtr->plan(GetCuTensorHandle(), worksize_provided, false); 
         if (!ret1) {
           std::cerr << "cutensor: NON-JIT plan creation failed 1 with error code but no exception" << std::endl;
           throw std::runtime_error("cutensor: NON-JIT plan creation failed 1 with error code but no exception");
         } 
       }
    
       new_plan.worksize = myEinsumPtr->getWorksize();
       try {
        new_plan.workspace = at::empty(new_plan.worksize, at::CUDA(at::kByte));
       } 
       catch (std::exception& e) {
        std::cerr << "cutensor: Workspace allocation failed: " << e.what() << std::endl;
        throw std::runtime_error("cutensor: Workspace allocation failed.");
       }
       
       new_plan.output_tensor = torch::empty(myEinsumPtr->getOutputShape(), input_0.options());
   });
   return new_plan;
 }
 
  torch::Tensor execute(
   const EinsumPlan& exec_plan
  ) {
   if (!exec_plan.myEinsumPtr) {
     throw std::runtime_error("Invalid EinsumPlan: myEinsumPtr is NULL");
   }

   AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, exec_plan.input_0.scalar_type(), "execute", [&] {
     constexpr int kMaxNumModes_ = 64;  
     Einsum<scalar_t, int64_t, kMaxNumModes_>* myEinsumPtr = static_cast<Einsum<scalar_t, int64_t, kMaxNumModes_>*>(exec_plan.myEinsumPtr);
     auto stream = at::cuda::getCurrentCUDAStream().stream();
     auto ret = myEinsumPtr->execute(
         GetCuTensorHandle(),
         exec_plan.input_0.data_ptr<scalar_t>(),
         exec_plan.input_1.data_ptr<scalar_t>(),
         exec_plan.output_tensor.data_ptr<scalar_t>(),
         exec_plan.workspace.data_ptr<uint8_t>(),
         stream
     );
     if (!ret){
      std::cerr << "cutensor: Einsum execution failed" << std::endl;
      throw std::runtime_error("cutensor: Launch failed.");
     } 
   });
   return exec_plan.output_tensor;
 }
 
 
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("einsum", &einsum, "einsum");
   m.def("execute", &execute, "execute");
   m.def("plan", &plan, "plan");

  py::class_<EinsumPlan>(m, "EinsumPlan")
    .def(py::init<>())
    .def_readwrite("myEinsumPtr", &EinsumPlan::myEinsumPtr)
    .def_readwrite("worksize", &EinsumPlan::worksize)
    .def_readwrite("workspace", &EinsumPlan::workspace)
    .def_readwrite("output_tensor", &EinsumPlan::output_tensor)
    .def_readwrite("input_0", &EinsumPlan::input_0)
    .def_readwrite("input_1", &EinsumPlan::input_1);

 }