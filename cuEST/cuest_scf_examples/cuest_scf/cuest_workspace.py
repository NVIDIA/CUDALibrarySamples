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

import numpy as np

from .cuda_utility import CudaUtility

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor

class CuestWorkspace():

    _dtype = np.dtype([
        ("hostBuffer", np.uintp, ),
        ("hostBufferSizeInBytes", np.uint64, ),
        ("deviceBuffer", np.uintp, ),
        ("deviceBufferSizeInBytes", np.uint64, ),
        ], align=True
        )

    def __init__(
        self,
        *,
        workspaceDescriptor: CuestWorkspaceDescriptor,
        ):

        self.initialized = False

        hostBufferSizeInBytes = workspaceDescriptor.struct['hostBufferSizeInBytes']
        deviceBufferSizeInBytes = workspaceDescriptor.struct['deviceBufferSizeInBytes']

        self.struct = np.array(1, dtype=self._dtype)
        self.struct['deviceBufferSizeInBytes'] = deviceBufferSizeInBytes
        self.struct['hostBufferSizeInBytes'] = hostBufferSizeInBytes
        self.struct['deviceBuffer'] = 0
        self.struct['hostBuffer'] = 0

        try:
            if deviceBufferSizeInBytes:
                self.struct['deviceBuffer'] = CudaUtility.cuda_malloc(size_in_bytes=deviceBufferSizeInBytes)

            if hostBufferSizeInBytes:
                self.struct['hostBuffer'] = CudaUtility.cuda_malloc_host(size_in_bytes=hostBufferSizeInBytes)
        except Exception:
            self._free_buffers()
            raise

        self.initialized = True

    def _free_buffers(
        self,
        ):

        if not hasattr(self, 'struct'):
            return

        device_buffer = int(self.struct['deviceBuffer'])
        host_buffer = int(self.struct['hostBuffer'])

        if device_buffer:
            try:
                CudaUtility.cuda_free(device_buffer)
            except Exception:
                pass
            self.struct['deviceBuffer'] = 0

        if host_buffer:
            try:
                CudaUtility.cuda_free_host(host_buffer)
            except Exception:
                pass
            self.struct['hostBuffer'] = 0

    def __del__(
        self,
        ):

        self._free_buffers()

    @property
    def pointer(self):
        return self.struct.ctypes.data
