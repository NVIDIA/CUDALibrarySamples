# ! /usr/bin/python
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

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.framework.load_library import load_op_library

from ..common import normalize_subscript

import glob
import os
pattern = os.path.join(os.path.dirname(__file__), 'binding*.so')
glob_res = glob.glob(pattern)
binding_file, = glob_res
einsum_lib = tf.load_op_library(binding_file)


def einsum(equation, *inputs, **kwargs):
    name = kwargs.pop('name', None)

    if kwargs:
        raise TypeError(
            'invalid keyword arguments for this function: ' +
            ', '.join([format(key) for key in sorted(list(kwargs.keys()))]))

    with ops.name_scope(name, 'einsum', [equation, inputs]):
        inputs = list(inputs)

        input_shapes = [x.get_shape() for x in inputs]
        input_axis_labels, output_axis_labels = special_math_ops._einsum_parse_and_resolve_equation(
            equation, input_shapes)

        axis_labels = set(''.join(input_axis_labels) + output_axis_labels)

        for a in axis_labels:
            for input_labels in input_axis_labels:
                if (len(input_axis_labels) == 1 and
                        input_labels.count(a) == 2 and
                        input_labels == input_labels[::-1] and
                        '->' not in equation):
                    return math_ops.trace(inputs[0])
                if input_labels.count(a) > 1:
                    raise ValueError(
                        'Subscript not supported: an axis appears more than once: %s'
                        % input_labels)

        for a in axis_labels:

            input_count = sum(1 for s in input_axis_labels if a in s)

            if input_count > 2 and a not in output_axis_labels:
                tf.logging.warn(
                    'Falling back to exponential-space implementation of einsum()'
                    ' because index "%s" is summed over more than two inputs.',
                    a)
                return special_math_ops._exponential_space_einsum(
                    equation, *inputs)

        equation = ','.join(input_axis_labels) + '->' + output_axis_labels
        if len(inputs) == 1:
            # inputs.append(inputs[0])
            inputs.append(tf.constant([0], dtype=inputs[0].dtype))
        return einsum_lib.einsum_cu_tensor(input_0=inputs[0],
                                           input_1=inputs[1],
                                           equation=equation)


@ops.RegisterGradient("EinsumCuTensor")
def _einsum_cu_tensor_grad(op, grad):
    A = op.inputs[0]
    B = op.inputs[1]

    subscript, _ = normalize_subscript(op.get_attr("equation").decode())
    lhs, modeC = subscript.split('->')
    if ',' in lhs:
        modeA, modeB = lhs.split(',')

        grad_A = einsum_lib.einsum_cu_tensor(input_0=grad,
                                             input_1=B,
                                             equation=modeC + ',' + modeB + '->' +
                                             modeA)

        grad_B = einsum_lib.einsum_cu_tensor(input_0=A,
                                             input_1=grad,
                                             equation=modeA + ',' + modeC + '->' +
                                             modeB)

        return [grad_A, grad_B]
    else:
        grad = einsum_lib.einsum_cu_tensor(input_0=grad,
                                           input_1=B,
                                           equation=modeC + '->' + lhs)
        return [grad, B]

