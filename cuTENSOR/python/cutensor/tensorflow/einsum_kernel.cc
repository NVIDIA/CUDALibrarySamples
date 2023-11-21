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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/types.h"

#include "../../einsum.h"

template<>
struct CuTensorTypeTraits<Eigen::half> {
  static cutensorDataType_t getDataType() {return CUTENSOR_R_16F;}
  static const cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_16F;}
  typedef float ScalarType;
};

template<>
struct CuTensorTypeTraits<tensorflow::bfloat16> {
  static cutensorDataType_t getDataType() {return CUTENSOR_R_16BF;}
  static const cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_16BF;}
  typedef float ScalarType;
};

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class EinsumCuTensorOp : public OpKernel {
 public:

  std::string equation_;

  explicit EinsumCuTensorOp(OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr("equation", &equation_);
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_0_tensor = context->input(0);
    const Tensor& input_1_tensor = context->input(1);

    std::vector<int64> input_0_shape, input_1_shape;
    for (int i = 0; i < input_0_tensor.dims(); i++)
        input_0_shape.push_back(input_0_tensor.dim_size(i));
    for (int i = 0; i < input_1_tensor.dims(); i++)
        input_1_shape.push_back(input_1_tensor.dim_size(i));

    constexpr int kMaxNumModes_ = 64; // maximal number of modes supported by cuTENSOR
    Einsum<T, int64, kMaxNumModes_> myEinsum(equation_, input_0_shape, input_1_shape);
    OP_REQUIRES(context, myEinsum.isInitialized(), errors::Internal("cutensor: Initialization failed."));

    auto output_dims = myEinsum.getOutputShape();
    // Create an output tensor
    Tensor* output_tensor = NULL;
    TensorShape output_shape = TensorShape(output_dims);
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));

    size_t worksize = myEinsum.getWorksize();
    // create contraction plan according to the worksize provided
    // update the worksize if cutensor does not need that much memory
    auto ret1 = myEinsum.plan(GetCuTensorHandle(), worksize);
    OP_REQUIRES(context, ret1, errors::Internal("cuTensor: plan creation failed."));
    // get the updated worksize
    worksize = myEinsum.getWorksize();
    Tensor work_tensor;
    int64 work_tensor_size = worksize / sizeof(float);
    TensorShape work_shape = { work_tensor_size };
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, work_shape, &work_tensor));

    auto device = context->eigen_device<Device>();
    auto ret = myEinsum.execute(GetCuTensorHandle(),
                                input_0_tensor.flat<T>().data(),
                                input_1_tensor.flat<T>().data(),
                                output_tensor->flat<T>().data(),
                                work_tensor.flat<float>().data(),
                                device.stream());

    OP_REQUIRES(context, ret, errors::Internal("cutensor: Launch failed."));
  }
};

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("EinsumCuTensor").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      EinsumCuTensorOp<GPUDevice, T>);
REGISTER_GPU(double);
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
REGISTER_GPU(tensorflow::bfloat16);
#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow
