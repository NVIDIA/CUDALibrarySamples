# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cuest.bindings as ce

from .cuest_parameters import CuestParameters


class DevicePointerView(object):
    """Minimal .pointer/.size wrapper so a raw device pointer can be passed to
    GPUMatrixUtility routines alongside GPUMatrix operands."""

    def __init__(self, pointer, size):
        self.pointer = pointer
        self.size = size


class CuestUtility(object):

    @staticmethod
    def make_compute_parameters(parameters_type, attribute_values):
        """One parameters object of the given type with the given
        (attribute, value) pairs applied."""
        compute_parameters = CuestParameters(
            parametersType=parameters_type,
            )
        for attribute, attribute_value in attribute_values:
            status = ce.cuestParametersConfigure(
                parametersType=parameters_type,
                parameters=compute_parameters.parameters,
                attribute=attribute,
                attributeValue=attribute_value,
                )
            if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                raise RuntimeError('cuestParametersConfigure failed: %s' % status)
        return compute_parameters
