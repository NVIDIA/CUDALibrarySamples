/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    // create contraction plan according to the worksize provided
    // update the worksize if cutensor does not need that much memory
    auto ret1 = myEinsum.plan(GetCuTensorHandle(), CUTENSOR_WORKSPACE_DEFAULT, false /* JIT compilation */);
    OP_REQUIRES(context, ret1, errors::Internal("cuTensor: plan creation failed."));
    // get the updated worksize
    size_t worksize = myEinsum.getWorksize();
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