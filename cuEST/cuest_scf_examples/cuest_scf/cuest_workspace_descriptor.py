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

import numpy as np

class CuestWorkspaceDescriptor():

    _dtype = np.dtype([
        ("hostBufferSizeInBytes", np.uint64, ),
        ("deviceBufferSizeInBytes", np.uint64, ),
        ], align=True
        )

    def __init__(
        self,
        *,
        hostBufferSizeInBytes: np.uint64 = 0,
        deviceBufferSizeInBytes: np.uint64 = 0,
        ):

        self.struct = np.array(
            (hostBufferSizeInBytes, deviceBufferSizeInBytes),
            dtype=self._dtype,
            )

    def __str__(
        self,
        ):

        hostsize = self.struct['hostBufferSizeInBytes']
        devicesize = self.struct['deviceBufferSizeInBytes']
        return f'host buffer size = {hostsize} bytes, device buffer size = {devicesize} bytes'

    @property
    def pointer(self):
        return self.struct.ctypes.data
