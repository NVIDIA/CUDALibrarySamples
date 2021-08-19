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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "absl/strings/str_split.h"

using namespace tensorflow;

REGISTER_OP("EinsumCuTensor")
    .Attr("T: {double, float, bfloat16, half}")
    .Attr("equation: string")
    .Input("input_0: T")
    .Input("input_1: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      using namespace ::tensorflow::shape_inference;

      std::string equation;
      TF_RETURN_IF_ERROR(c->GetAttr("equation", &equation));

      ShapeHandle input_0_shape = c->input(0);

      ShapeHandle input_1_shape = c->input(1);

      std::vector<std::string> lhs_and_rhs = absl::StrSplit(equation, "->");
      assert(lhs_and_rhs.size() == 2);
      auto modeC = lhs_and_rhs[1];

      lhs_and_rhs = absl::StrSplit(lhs_and_rhs[0], ",");
      bool useB = lhs_and_rhs.size() == 2;
      assert(lhs_and_rhs.size() <= 2);
      auto modeA = lhs_and_rhs[0];
      auto modeB = lhs_and_rhs[useB ? 1 : 0];

      std::unordered_map<char, DimensionHandle> dim_map;

      assert(modeA.size() == c->Rank(input_0_shape));
      for (int i = 0; i < modeA.size(); i++) {
        dim_map[modeA[i]] = c->Dim(input_0_shape, i);
      }

      assert((! useB) || (modeB.size() == c->Rank(input_1_shape)));
      for (int i = 0; useB && (i < modeB.size()); i++) {
        if (dim_map.find(modeB[i]) == dim_map.end()) {
          dim_map[modeB[i]] = c->Dim(input_1_shape, i);
        } else {
          DimensionHandle out;
          TF_RETURN_IF_ERROR(c->Merge(c->Dim(input_1_shape, i), dim_map[modeB[i]], &out));
        }
      }

      std::vector<DimensionHandle> output_dims;
      for (auto mode : modeC) {
        output_dims.push_back(dim_map[mode]);
      }

      ShapeHandle output_shape = c->MakeShape(output_dims);
      c->set_output(0, output_shape);

      return Status::OK();
    });
