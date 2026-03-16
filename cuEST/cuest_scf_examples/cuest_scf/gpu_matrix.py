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

from .cuda_utility import CudaUtility

class GPUMatrix(object):

    @staticmethod
    def from_numpy(cpu_array):

        assert len(cpu_array.shape) == 2
        matrix = GPUMatrix(
            nrows=cpu_array.shape[0],
            ncols=cpu_array.shape[1],
            dtype=cpu_array.dtype,
            )

        CudaUtility.cuda_memcpy_htod(
            device_pointer=matrix.pointer,
            host_pointer=cpu_array.ctypes.data,
            size_in_bytes=matrix.nrows*matrix.ncols*matrix.type_size,
            )

        return matrix

    @property
    def shape(self):
        return (self.nrows, self.ncols)

    def zero(self):
        CudaUtility.cuda_zero_memory(
            device_pointer=self.pointer,
            size_in_bytes=self.size*self.type_size,
            )

    def __init__(
        self,
        *,
        nrows,
        ncols,
        dtype=np.double,
        initialize=True,
        ):

        self.type_size = np.empty(1, dtype=dtype).itemsize
        self.nrows = nrows
        self.ncols = ncols
        self.dtype = dtype
        self.size = nrows * ncols

        self.pointer = CudaUtility.cuda_malloc(
            size_in_bytes=self.size*self.type_size,
            )

        if initialize:
            self.zero()


    def __del__(self):
        if hasattr(self, 'pointer') and self.pointer:
            CudaUtility.cuda_free(pointer=self.pointer)


    def to_numpy(self):
        cpu_array = np.empty(
            self.shape,
            dtype=self.dtype,
            )

        CudaUtility.cuda_memcpy_dtoh(
            host_pointer=cpu_array.ctypes.data,
            device_pointer=self.pointer,
            size_in_bytes=self.nrows*self.ncols*self.type_size,
            )

        return cpu_array


    def clone(self):
        matrix = GPUMatrix(
            nrows=self.nrows,
            ncols=self.ncols,
            dtype=self.dtype,
            )

        CudaUtility.cuda_memcpy_dtod(
            dst_device_pointer=matrix.pointer,
            src_device_pointer=self.pointer,
            size_in_bytes=self.nrows*self.ncols*self.type_size,
            )

        return matrix

