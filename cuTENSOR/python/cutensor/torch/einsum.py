#! /usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  - Neither the name(s) of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import torch
import torch.autograd
import numpy as np
from .binding import einsum
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


class Einsum(torch.nn.Module):

    def __init__(self, equation):
        super(Einsum, self).__init__()
        self.equation = equation
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, input_0, input_1):
        return EinsumFunction.apply(self.equation, input_0, input_1)


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
