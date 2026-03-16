# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

class CuestHandle(object):

    def __init__(
        self,
        ):

        self.initialized = False

        cuest_handle_parameters = CuestParameters(
            parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
            )

        # NOTE: The CuestHandle has a number of defaultable parameters that
        # production codes may wish to expose to the user. Notable among these
        # is the CUDA stream. The CuestHandle will default to the host stream
        # (0), which is blocking. However, production codes may wish to provide
        # a custom stream to the CuestHandle, which will allow for asynchronous
        # cuEST library calls. Any such calls must use CUDA device memory that
        # is synchronized with the stream.

        self.handle = ce.cuestHandle()
        
        status = ce.cuestCreate(
            parameters=cuest_handle_parameters.parameters,
            handle=self.handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest_create failed')

        self.initialized = True

    def __del__(self):

        if not self.initialized: return

        status = ce.cuestDestroy(
            handle=self.handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest_destroy failed')

    def set_math_mode(
        self,
        *,
        mode,
        ):

        if not self.initialized: return

        status = ce.cuestSetMathMode(
            handle=self.handle,
            mode=mode,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest_set_math_mode failed')

