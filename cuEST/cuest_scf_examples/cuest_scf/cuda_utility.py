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
import cuda.bindings.runtime as cuda

class CudaUtility(object):

    @staticmethod
    def cuda_malloc(size_in_bytes):
        ret, pointer = cuda.cudaMalloc(size_in_bytes)

        if ret != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to allocate CUDA array:", ret)

        return pointer

    @staticmethod
    def cuda_free(pointer):
    
        ret = cuda.cudaFree(pointer)
    
        if ret[0] != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to free CUDA array:", ret[0])

    @staticmethod
    def cuda_malloc_host(size_in_bytes):
    
        # NOTE: we are using cudaMallocHost as an expediency here. 
        # Ideally, in production codes, this would be simple malloc or similar
        # as pinned memory is not required here.

        ret, pointer = cuda.cudaMallocHost(size_in_bytes)
    
        if ret != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to allocate host array:", ret)
    
        return pointer

    @staticmethod
    def cuda_free_host(pointer):

        ret = cuda.cudaFreeHost(pointer)

        if ret[0] != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to free host array:", ret[0])

    @staticmethod
    def cuda_memcpy_dtoh(
        *,
        host_pointer,
        device_pointer,
        size_in_bytes,
        stream=None,
        ):
        """
        This function should take integer representations of pointers to host
        and device memory allocations, and copy size_in_bytes bytes from the
        device memory to the host memory.  The stream to execute this can be
        optionally provided.
        """
    
        ret = cuda.cudaMemcpyAsync(
            dst=host_pointer,
            src=device_pointer,
            count=size_in_bytes,
            kind=cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream=stream)
    
        if ret[0] != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError("Copy from device to host failed:", ret[0])
    
        cuda.cudaStreamSynchronize(stream)
    
    @staticmethod
    def cuda_memcpy_htod(
        *,
        device_pointer,
        host_pointer,
        size_in_bytes,
        stream=None,
        ):
        """
        This function should take integer representations of pointers to host
        and device memory allocations, and copy size_in_bytes bytes from the
        host memory to the device memory.  The stream to execute this can be
        optionally provided.
        """
    
        ret = cuda.cudaMemcpyAsync(
            dst=device_pointer,
            src=host_pointer,
            count=size_in_bytes,
            kind=cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream=stream)
    
        if ret[0] != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError("Copy from host to device failed:", ret[0])
    
        cuda.cudaStreamSynchronize(stream)

    @staticmethod
    def cuda_memcpy_dtod(
        *,
        dst_device_pointer,
        src_device_pointer,
        size_in_bytes,
        stream=None,
        ):
        """
        This function should take integer representations of pointers to two
        device memory allocations, and copy size_in_bytes bytes from the
        src memory to the destination memory.  The stream to execute this can be
        optionally provided.
        """
    
        ret = cuda.cudaMemcpyAsync(
            dst=dst_device_pointer,
            src=src_device_pointer,
            count=size_in_bytes,
            kind=cuda.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            stream=stream)
    
        if ret[0] != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError("Copy from device to device failed:", ret[0])
    
        cuda.cudaStreamSynchronize(stream)

    @staticmethod
    def cuda_zero_memory(
        *,
        device_pointer,
        size_in_bytes,
        stream=None,
        ):
        """
        Initialize a chunk of device memory to the zero.  The stream to execute this can be optionally provided.
        """

        ret = cuda.cudaMemsetAsync(
            devPtr=device_pointer,
            value=0,
            count=size_in_bytes,
            stream=stream,
            )

        if ret[0] != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError("CUDA memset failed:", ret[0])

        cuda.cudaStreamSynchronize(stream)
