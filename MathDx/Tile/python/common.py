# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host numpy and cuda.core helpers shared by the Python examples.

The kernels use CUDA Tile and the `nvmath.tile` modules.
Host NumPy arrays replace the earlier CuPy arrays. The examples copy them through `cuda.core` buffers.
A `cuda.core` stream supplies its integer handle to `cuda.tile.launch`, as required by the current Tile runtime.
"""

from __future__ import annotations

import ctypes

import cuda.tile as ct
import numpy as np
from cuda.core import Device, EventOptions, LegacyPinnedMemoryResource


class DeviceArray:
    """Shaped view of a `cuda.core` buffer, exposing the CUDA Array Interface."""

    def __init__(self, buffer, shape, dtype):
        if np.isscalar(shape):
            shape = (shape,)
        self.buffer = buffer
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = np.dtype(dtype)

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": self.shape,
            "typestr": self.dtype.str,
            "data": (int(self.buffer.handle), False),
            "version": 3,
            "strides": None,
        }

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)
        if int(np.prod(shape, dtype=np.int64)) != int(
            np.prod(self.shape, dtype=np.int64)
        ):
            raise ValueError(f"cannot reshape {self.shape} to {shape}")
        return DeviceArray(self.buffer, shape, self.dtype)


class Runtime:
    """One device, one stream, and the copies the examples share."""

    def __init__(self):
        self.device = Device(0)
        self.device.set_current()
        self.stream = self.device.create_stream()
        # Page-locked staging through cuMemAllocHost: the pool-backed
        # PinnedMemoryResource is not supported on Windows.
        self._pinned = LegacyPinnedMemoryResource()

    @property
    def tile_stream(self):
        return int(self.stream.handle)

    def _copy_host_to_buffer(self, host, buffer):
        host = np.ascontiguousarray(host)
        staging = self._pinned.allocate(host.nbytes, stream=self.stream)
        ctypes.memmove(int(staging.handle), host.ctypes.data, host.nbytes)
        buffer.copy_from(staging, stream=self.stream)
        # Freeing legacy pinned memory is not stream-ordered.
        self.stream.sync()
        staging.close()

    def _copy_buffer_to_host(self, buffer, host):
        host = np.ascontiguousarray(host)
        staging = self._pinned.allocate(host.nbytes, stream=self.stream)
        buffer.copy_to(staging, stream=self.stream)
        self.stream.sync()
        ctypes.memmove(host.ctypes.data, int(staging.handle), host.nbytes)
        staging.close()
        return host

    def to_device(self, host) -> DeviceArray:
        host = np.ascontiguousarray(host)
        buffer = self.device.allocate(host.nbytes, stream=self.stream)
        self._copy_host_to_buffer(host, buffer)
        return DeviceArray(buffer, host.shape, host.dtype)

    def zeros(self, shape, dtype) -> DeviceArray:
        dtype = np.dtype(dtype)
        if np.isscalar(shape):
            shape = (shape,)
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        buffer = self.device.allocate(nbytes, stream=self.stream)
        buffer.fill(0, stream=self.stream)
        return DeviceArray(buffer, shape, dtype)

    def full(self, shape, value, dtype) -> DeviceArray:
        return self.to_device(np.full(shape, value, dtype=dtype))

    def to_host(self, array: DeviceArray) -> np.ndarray:
        return self._copy_buffer_to_host(
            array.buffer, np.empty(array.shape, dtype=array.dtype)
        )

    def copy_from_host(self, array: DeviceArray, host):
        host = np.ascontiguousarray(host, dtype=array.dtype)
        if host.shape != array.shape:
            host = host.reshape(array.shape)
        self._copy_host_to_buffer(host, array.buffer)

    def launch(self, kernel, args, grid=(1, 1, 1)):
        ct.launch(self.tile_stream, grid, kernel, args)

    def sync(self):
        self.stream.sync()

    def measure(self, launch, warmups=1, runs=3):
        """Milliseconds per launch, averaged after a warm-up."""
        options = EventOptions(timing_enabled=True)
        for _ in range(warmups):
            launch()
        self.device.sync()
        start = self.device.create_event(options=options)
        stop = self.device.create_event(options=options)
        self.stream.record(start)
        for _ in range(runs):
            launch()
        self.stream.record(stop)
        stop.sync()
        return (stop - start) / runs


def relative_error(expected, actual) -> float:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    return float(np.linalg.norm(expected - actual) / np.linalg.norm(expected))
