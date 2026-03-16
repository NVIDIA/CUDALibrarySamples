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

# There are numerous way to access GPU resources from Python.  High level
# packages such as CuPy, TensorFlow and Pytorch can simplify allocations and
# transfers between host and device.  Here we demonstrate two lower level
# approaches to accessing CUDA functions to provide these utilities: 1) using
# the cuda-bindings package to access native CUDA in Pythonic way, and 2) using
# the ctypes module to directly access these functions from the CUDA runtime
# library.  A real implementation should ideally use a single unified
# mechanism, but a mixture is used here for demonstration purposes.

import ctypes
try:
    import numpy as np
except ImportError:
    raise RuntimeError("numpy could not be imported. It is available via\n\tpip install numpy")

try:
    import cuda.bindings.runtime as cuda
except ImportError:
    raise RuntimeError("cuda.bindings could not be imported. It is available via\n\tpip install cuda-bindings")


# These will be used by the ctypes-powered utility functions below.
try:
    _cublas = ctypes.cdll.LoadLibrary('libcublas.so')
except OSError as err:
    raise RuntimeError("Could not load cuBLAS library.") from err

try:
    _cusolver = ctypes.cdll.LoadLibrary('libcusolver.so')
except OSError as err:
    raise RuntimeError("Could not load cuSolver library.") from err



def make_stream():
    """
    This function should return a new cuestStream_t (or Python
    equivalent) representing a CUDA stream.
    """

    ret, stream = cuda.cudaStreamCreate()

    if ret != cuda.cudaError_t.cudaSuccess:
        raise RuntimeError("Unable to make CUDA stream:", ret)

    return stream


def cuda_malloc(
    *,
    size_in_bytes,
    ):
    """
    This function should allocate device memory, and return an integer
    representation of the address of the resulting block.
    """
    ret, pointer = cuda.cudaMalloc(size_in_bytes)

    if ret != cuda.cudaError_t.cudaSuccess:
        raise RuntimeError("Unable to allocate CUDA array:", ret)

    return pointer


def cuda_memcpy_dtoh(
    *,
    host_pointer,
    device_pointer,
    size_in_bytes,
    stream=0,
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


def cuda_memcpy_htod(
    *,
    device_pointer,
    host_pointer,
    size_in_bytes,
    stream=0,
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


def cuda_free(
    array
    ):
    """
    This function should free memory allocated on the device, passed in
    as an integer representation of its device pointer
    """

    ret = cuda.cudaFree(array)

    if ret[0] != cuda.cudaError_t.cudaSuccess:
        raise RuntimeError("Unable to free CUDA array:", ret[0])


def make_cublas_handle():
    """
    Helper function demonstrating how to use ctypes to access the C
    cuBLAS API to make a new cuBLAS handle.
    """
    handle = ctypes.c_void_p()

    CUBLAS_STATUS_SUCCESS = 0

    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    status = _cublas.cublasCreate_v2(ctypes.byref(handle))
    if status != CUBLAS_STATUS_SUCCESS:
        raise RuntimeError(f"cublasCreate_v2 failed with status {status}")

    return handle


def free_cublas_handle(
    *,
    handle
    ):

    """
    Helper function to free cuBLAS resource handle.
    """

    CUBLAS_STATUS_SUCCESS = 0

    _cublas.cublasDestroy_v2.restype = ctypes.c_int
    _cublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]

    status = _cublas.cublasDestroy_v2(handle)
    if status != CUBLAS_STATUS_SUCCESS:
        raise RuntimeError(f"cublasDestroy_v2 failed with status {status}")


def make_cusolver_handle():
    """
    Make a cuSolver handle via the CUDA C API.
    """
    handle = ctypes.c_void_p()

    CUSOLVER_STATUS_SUCCESS = 0

    _cusolver.cusolverDnCreate.restype = ctypes.c_int
    _cusolver.cusolverDnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    status = _cusolver.cusolverDnCreate(ctypes.byref(handle))
    if status != CUSOLVER_STATUS_SUCCESS:
        raise RuntimeError(f"cusolverDnCreate failed with status {status}")

    return handle


def free_cusolver_handle(
    handle
    ):
    """
    Free resources associated with a cuSolver handle.
    """

    CUSOLVER_STATUS_SUCCESS = 0

    _cusolver.cusolverDnDestroy.restype = ctypes.c_int
    _cusolver.cusolverDnDestroy.argtypes = [ctypes.c_void_p]

    status = _cusolver.cusolverDnDestroy(handle)
    if status != CUSOLVER_STATUS_SUCCESS:
        raise RuntimeError(f"cusolverDnDestroy failed with status {status}")


def set_cublas_stream(
    *,
    handle,
    stream,
    ):
    """
    Set the active stream for a given cuBLAS handle
    """

    CUBLAS_STATUS_SUCCESS = 0

    _cublas.cublasSetStream_v2.restype = ctypes.c_int
    _cublas.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    status = _cublas.cublasSetStream_v2(handle, stream)
    if status != CUBLAS_STATUS_SUCCESS:
        raise RuntimeError(f"cublasSetStream failed with status {status}")


def set_cusolver_stream(
    *,
    handle,
    stream,
    ):
    """
    Set the stream for a given cuSolver handle.
    """

    CUSOLVER_STATUS_SUCCESS = 0

    _cusolver.cusolverDnSetStream.restype = ctypes.c_int
    _cusolver.cusolverDnSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    status = _cusolver.cusolverDnSetStream(handle, stream)
    if status != CUSOLVER_STATUS_SUCCESS:
        raise RuntimeError(f"cusolverDnSetStream failed with status {status}")


class WorkspaceDescriptor:
    """
    This helper function needs to implement the following C structure,
    and define a pointer method that returns a pointer to it.

    typedef struct {
        size_t hostBufferSizeInBytes;      ///< Required size of host workspace buffer in bytes
        size_t deviceBufferSizeInBytes;    ///< Required size of device workspace buffer in bytes
    } cuestWorkspaceDescriptor_t;

    This implementation uses Numpy, as the ctypes.data member provides a
    convenient way to access a pointer to the object
    """

    def __init__(
        self,
        *,
        host_buffer_size_in_bytes = 0,
        device_buffer_size_in_bytes = 0,
        ):

        _workspace_descriptor_dtype = np.dtype([
            ("hostBufferSizeInBytes", np.uint64, ),
            ("deviceBufferSizeInBytes", np.uint64, ),
            ], align=True
            )

        self.struct = np.empty(1, dtype=_workspace_descriptor_dtype)
        self.struct['deviceBufferSizeInBytes'] = device_buffer_size_in_bytes
        self.struct['hostBufferSizeInBytes'] = host_buffer_size_in_bytes

    def __str__(
        self,
        ):

        host_size = self.struct['hostBufferSizeInBytes'].item()
        device_size = self.struct['deviceBufferSizeInBytes'].item()
        return f'host buffer size = {host_size} bytes, device buffer size = {device_size} bytes'

    @property
    def pointer(self):
        return self.struct.ctypes.data

class Workspace:
    """
    This helper function needs to implement the following C structure,
    and define a pointer method that returns a pointer to it.

    typedef struct {
        uintptr_t hostBuffer;              ///< Opaque pointer to host-side workspace buffer
        size_t hostBufferSizeInBytes;      ///< Size of host workspace in bytes
        uintptr_t deviceBuffer;            ///< Opaque pointer to device-side (GPU) workspace buffer
        size_t deviceBufferSizeInBytes;    ///< Size of device workspace in bytes
    } cuestWorkspace_t;

    This implementation uses Numpy, as the ctypes.data member provides a
    convenient way to access a pointer to the object
    """
    def __init__(
        self,
        *,
        workspaceDescriptor,
        ):
        _workspace_dtype = np.dtype([
            ("hostBuffer", np.uintp, ),
            ("hostBufferSizeInBytes", np.uint64, ),
            ("deviceBuffer", np.uintp, ),
            ("deviceBufferSizeInBytes", np.uint64, ),
            ], align=True
            )

        host_buffer_size_in_bytes = workspaceDescriptor.struct['hostBufferSizeInBytes'].item()
        device_buffer_size_in_bytes = workspaceDescriptor.struct['deviceBufferSizeInBytes'].item()

        self.struct = np.empty(1, dtype=_workspace_dtype)
        self.struct['deviceBufferSizeInBytes'] = device_buffer_size_in_bytes
        self.struct['hostBufferSizeInBytes'] = host_buffer_size_in_bytes

        if device_buffer_size_in_bytes:
            ret, pointer = cuda.cudaMalloc(device_buffer_size_in_bytes)
            if ret != cuda.cudaError_t.cudaSuccess:
                raise RuntimeError("Failed to allocate workspace", ret)
            self.struct['deviceBuffer'] = pointer
        else:
            self.struct['deviceBuffer'] = 0

        if host_buffer_size_in_bytes:
            self.cpu_memory = np.zeros(host_buffer_size_in_bytes, dtype=np.int8)
            self.struct['hostBuffer'] = self.cpu_memory.ctypes.data
        else:
            self.struct['hostBuffer'] = 0
            self.cpu_memory = None

    def __del__(
        self,
        ):

        if self.struct['deviceBuffer']:
            ret = cuda.cudaFree(self.struct['deviceBuffer'].item())
            if ret[0] != cuda.cudaError_t.cudaSuccess:
                raise RuntimeError("Unable to deallocate workspace:", ret[0])

    @property
    def pointer(self):
        return self.struct.ctypes.data

