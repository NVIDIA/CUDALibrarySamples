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

import ctypes

from cuda.bindings import driver
from cuda.bindings import nvrtc

class MP2KernelHelper:
    """
    Standalone float64 CUDA helper using cuda-bindings to access nvrtc and driver routines.

    API style:
    - Device pointers are passed as Python int.
    - Shape/length information is passed explicitly.
    - Kernels are JIT-compiled in __init__ and ready to use afterward.

    Methods
    -------
    make_pair_amplitudes(i, j, nocc, nvir, eps_ptr, pair_mo_ints_ptr, pair_amplitudes_ptr, stream=0, synchronize=False)
        Compute MP2 amplitudes for a given i, j (occupied) integral pair:
            amplitudes[a, b] = (i a | j b) / (eps[i] + eps[j] - eps[a] - eps[b])
    """

    _SRC = r'''
    extern "C" __global__
    void make_pair_amplitudes(
        long long i,
        long long j,
        long long nocc,
        long long nvir,
        const double* pair_mo_ints,
        const double* eps,
        double* pair_amplitudes
    ) {
        long long rel_a = (long long) blockIdx.x * blockDim.x + threadIdx.x;
        long long rel_b = (long long) blockIdx.y * blockDim.y + threadIdx.y;

        if (rel_a < nvir && rel_b < nvir) {
            double denominator = eps[i] + eps[j] - eps[rel_a + nocc] - eps[rel_b + nocc];
            double numerator = pair_mo_ints[rel_a * nvir + rel_b];
            pair_amplitudes[rel_a * nvir + rel_b] = numerator * (fabs(denominator) < 1e-20 ? 1e20 : 1.0 / denominator);
        }
    }
    '''

    def __init__(self, device_id=0, threads_per_block=256, gpu_arch=None):
        self.device_id = int(device_id)
        self.threads_per_block = int(threads_per_block)
        self.gpu_arch = gpu_arch

        self._dev = None
        self._ctx = None
        self._module = None

        # Function signatures get stored here
        self._func_make_pair_amplitudes = None

        self._init_cuda()

        if self.gpu_arch is None:
            self.gpu_arch = self._detect_gpu_arch()
        elif not isinstance(self.gpu_arch, bytes):
            self.gpu_arch = str(self.gpu_arch).encode()

        self._compile_and_load()

    @staticmethod
    def _check_driver(result):
        err = result[0]
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver error: {err}")
        if len(result) == 1:
            return None
        if len(result) == 2:
            return result[1]
        return result[1:]

    @staticmethod
    def _check_nvrtc(result, prog=None):
        err = result[0]
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            log = ""
            if prog is not None:
                log = MP2KernelHelper._get_nvrtc_log(prog)
            raise RuntimeError(f"NVRTC error: {err}\n{log}")
        if len(result) == 1:
            return None
        if len(result) == 2:
            return result[1]
        return result[1:]

    @staticmethod
    def _get_nvrtc_log(prog):
        _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log_size = int(log_size)
        if log_size <= 0:
            return ""
        buf = bytearray(log_size)
        nvrtc.nvrtcGetProgramLog(prog, buf)
        return bytes(buf).rstrip(b"\x00").decode(errors="replace")

    @staticmethod
    def _get_nvrtc_ptx(prog):
        _, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
        ptx_size = int(ptx_size)
        if ptx_size <= 0:
            raise RuntimeError("NVRTC returned empty PTX size")
        buf = bytearray(ptx_size)
        err = nvrtc.nvrtcGetPTX(prog, buf)
        if err[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC error: {err[0]}")
        return bytes(buf)

    def _init_cuda(self):
        self._check_driver(driver.cuInit(0))
        self._dev = self._check_driver(driver.cuDeviceGet(self.device_id))
        self._ctx = self._check_driver(driver.cuDevicePrimaryCtxRetain(self._dev))
        self._check_driver(driver.cuCtxSetCurrent(self._ctx))

    def _detect_gpu_arch(self):
        major = self._check_driver(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                self._dev,
            )
        )
        minor = self._check_driver(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                self._dev,
            )
        )
        return f"compute_{major}{minor}".encode()

    def _compile_and_load(self):
        prog = self._check_nvrtc(
            nvrtc.nvrtcCreateProgram(
                self._SRC.encode(),
                b"kernel_helper.cu",
                0,
                None,
                None,
            )
        )

        opts = [
            b"--std=c++11",
            b"--gpu-architecture=" + self.gpu_arch,
        ]
        self._check_nvrtc(
            nvrtc.nvrtcCompileProgram(prog, len(opts), opts),
            prog=prog,
        )

        ptx = self._get_nvrtc_ptx(prog)

        self._module = self._check_driver(driver.cuModuleLoadData(ptx))

        # Get the functions we JIT-ed
        self._func_make_pair_amplitudes = self._check_driver(
            driver.cuModuleGetFunction(self._module, b"make_pair_amplitudes")
        )

        nvrtc.nvrtcDestroyProgram(prog)

    def close(self):
        if self._module is not None:
            driver.cuModuleUnload(self._module)
            self._module = None

        if self._ctx is not None:
            driver.cuCtxSetCurrent(0)
            driver.cuDevicePrimaryCtxRelease(self._dev)
            self._ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _as_device_ptr(ptr):
        ptr = int(ptr)
        if ptr <= 0:
            raise ValueError("device pointer must be a positive integer")
        return ctypes.c_uint64(ptr)

    @staticmethod
    def _as_count(n):
        n = int(n)
        if n < 0:
            raise ValueError("count must be non-negative")
        return ctypes.c_int64(n)

    @staticmethod
    def _as_stream(stream):
        return int(stream)

    @staticmethod
    def _kernel_params(*values):
        holders = list(values)
        params = (ctypes.c_void_p * len(holders))(
            *[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in holders]
        )
        return holders, params

    # Python wrappers to functions calling the custom kernels
    def make_pair_amplitudes(self, *, i, j, nocc, nvir, eps_ptr, pair_mo_ints_ptr, pair_amplitudes_ptr, stream=0, synchronize=False):
        """
        Launch:
            pair_amplitudes[a, b] = (i a | j b) / (eps[i] + eps[j] - eps[a] - eps[b])

        Arrays are assumed to be flattened row-major device buffers:
        - eps_ptr: shape (nmo,)
        - pair_mo_ints_ptr: shape (nvir, nvir)
        - pair_amplitudes_ptr: shape (nvir, nvir)
        """
        i_arg = self._as_count(i)
        j_arg = self._as_count(j)
        nocc_arg = self._as_count(nocc)
        nvir_arg = self._as_count(nvir)
        eps_arg = self._as_device_ptr(eps_ptr)
        pair_mo_ints_arg = self._as_device_ptr(pair_mo_ints_ptr)
        pair_amplitudes_arg = self._as_device_ptr(pair_amplitudes_ptr)

        _, params = self._kernel_params(
            i_arg, j_arg, nocc_arg, nvir_arg, pair_mo_ints_arg, eps_arg, pair_amplitudes_arg
        )

        # 2D block: per-dimension thread count must satisfy block_xy^2 <= 1024 (CUDA max threads/block).
        block_xy = min(32, self.threads_per_block)
        blocks_x = (nvir_arg.value + block_xy - 1) // block_xy
        blocks_y = (nvir_arg.value + block_xy - 1) // block_xy

        self._check_driver(
            driver.cuLaunchKernel(
                self._func_make_pair_amplitudes,
                blocks_x, blocks_y, 1,
                block_xy, block_xy, 1,
                0,
                self._as_stream(stream),
                params,
                0,
            )
        )
        if synchronize:
            self._check_driver(driver.cuCtxSynchronize())

