# ! /usr/bin/python
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
        input_axis_labels, output_axis_labels = special_math_ops._einsum_v1_parse_and_resolve_equation(
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
                tf.compat.v1.logging.warn(
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
