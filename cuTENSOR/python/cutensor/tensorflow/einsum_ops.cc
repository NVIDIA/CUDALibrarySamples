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

      return OkStatus();
    });