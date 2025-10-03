#! /usr/bin/python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.autograd
import numpy as np
from .binding import einsum, plan, execute
from ..common import normalize_subscript

class EinsumFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, equation, input_0, input_1=None):
        equation, isBinary = normalize_subscript(equation)
        if isBinary and input_1 is None:
            raise RuntimeError('The subscript indicates two inputs, but only one was passed')
        if not isBinary and input_1 is not None:
            raise RuntimeError('The subscript indicates one input, but two were passed')
        if input_1 is None:
            input_1 = input_0.new_empty((1,))

        output = einsum(equation, input_0, input_1, False, False)

        if isBinary:
            ctx.save_for_backward(input_0, input_1)

        ctx.equation = equation
        ctx.isBinary = isBinary

        return output

    @staticmethod
    def backward(ctx, grad_output):
        equation = ctx.equation
        lhs, modeC = equation.split('->')
        if ctx.isBinary:
            input_0, input_1 = ctx.saved_tensors
            conjugate = False
            if torch.is_complex(input_0) or torch.is_complex(input_1):
                conjugate = True
            modeA, modeB = lhs.split(',')
            d_input_0 = einsum(modeC + ',' + modeB + '->' + modeA, grad_output,
                               input_1, False, conjugate)
            d_input_1 = einsum(modeA + ',' + modeC + '->' + modeB, input_0,
                               grad_output, conjugate, False)
            return None, d_input_0, d_input_1
        else:
            dummy = grad_output.new_empty((1,))
            d_input = einsum(modeC + '->' + lhs, grad_output, dummy, False, False)
            return None, d_input


    def plan(equation, input_0, input_1=None, jit_pref=False):
        equation, isBinary = normalize_subscript(equation)
        if isBinary and input_1 is None:
            raise RuntimeError('The subscript indicates two inputs, but only one was passed')
        if not isBinary and input_1 is not None:
            raise RuntimeError('The subscript indicates one input, but two were passed')
        if input_1 is None:
            input_1 = input_0.new_empty((1,))

        output = plan(equation, input_0, input_1, False, False, jit_pref)

        return output
    

    def execute(plan):
        try:
            result = execute(plan)
            if result is None:  # Handle NULL return
                raise RuntimeError("cuTENSOR execute returned NULL (CUTENSOR_STATUS_INVALID_VALUE)")
            return result
        except SystemError as e:
            raise RuntimeError(f"cuTENSOR execution failed {e}")


class Einsum(torch.nn.Module):

    def __init__(self, equation):
        super(Einsum, self).__init__()
        self.equation = equation
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, input_0, input_1):
        return EinsumFunction.apply(self.equation, input_0, input_1)
    
    def plan(self, input_0, input_1, jit_pref=False):
        return EinsumFunction.plan(self.equation, input_0, input_1=input_1, jit_pref=jit_pref)

    def execute(self, plan):
        return EinsumFunction.execute(plan)


def _compute_target_tensor(in0, in1, target, rest):
    result = ""
    rest = ''.join(rest) + target
    for m in in0[:-1] + in1[:-1] + in1[-1] + in0[-1]:
        if m in rest and not m in result:
            result += m
    # reorder target modes like target
    result = list(result)
    for i in range(len(result)):
        if result[i] not in target: continue
        for j in range(i):
            if result[j] not in target: continue
            if target.index(result[j]) > target.index(result[i]):
                result[i], result[j] = result[j], result[i]
    return ''.join(result)


def EinsumGeneral(equation, *tensors, **kwargs):
    tensors = list(tensors)
    equation, isBinary = normalize_subscript(equation)
    path = np.einsum_path(equation,
                          *[np.broadcast_to(np.nan, t.shape) for t in tensors],
                          **kwargs)
    path = path[0][1:]
    equation = equation.split('->')
    eqs = equation[0].split(',')
    target = equation[1]
    for step in path:
        if len(step) == 1:
            result = EinsumFunction.apply(eqs[0] + '->' + target, tensors[0])
            continue
        assert step[0] < step[1]
        in0 = tensors[step[0]]
        in1 = tensors[step[1]]
        tensors.pop(step[1])
        tensors.pop(step[0])
        tgt = _compute_target_tensor(eqs[step[0]], eqs[step[1]], target, [eq for idx, eq in enumerate(eqs) if idx not in step])
        assert tgt != ""
        eq = eqs[step[0]] + ',' + eqs[step[1]] + '->' + tgt
        eqs.pop(step[1])
        eqs.pop(step[0])
        eqs.append(tgt)
        result = EinsumFunction.apply(eq, in0, in1)
        tensors.append(result)
    return result
    
