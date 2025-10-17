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

from cutensor.c_extensions_utils import CustomExtension

einsum_torch = CustomExtension.Torch('cutensor.torch.binding',
                                     sources=['cutensor/torch/einsum.cc'])

einsum_tf = CustomExtension.Tensorflow(
    'cutensor.tensorflow.binding',
    sources=[
        'cutensor/tensorflow/einsum_kernel.cc',
        'cutensor/tensorflow/einsum_ops.cc',
        'cutensor/tensorflow/einsum_module.cc'
    ])
