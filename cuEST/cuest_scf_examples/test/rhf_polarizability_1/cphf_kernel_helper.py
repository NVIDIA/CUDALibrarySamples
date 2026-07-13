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

class CPHFKernelHelper:
    """
    Standalone float64 CUDA helper using cuda-bindings to access nvrtc and driver routines.

    API style:
    - Device pointers are passed as Python int.
    - Shape/length information is passed explicitly.
    - Kernels are JIT-compiled in __init__ and ready to use afterward.

    Methods
    -------
    ria_update(nrhs, nocc, nvir, ria_ptr, tia_ptr, eps_occ_ptr, eps_vir_ptr, stream=0, synchronize=False)
        Compute Fock contributions to MO hessian - vector product:
            ria[n, i, a] += (eps_vir[a] - eps_occ[i]) * tia[n, i, a]
    apply_preconditioner(nrhs, nocc, nvir, zia_ptr, ria_ptr, eps_occ_ptr, eps_vir_ptr, stream=0, synchronize=False)
        Apply orbital energy differences as a preconditioner to the CPHF residual
            zia[n, i, a] = ria[n, i, a] / (eps_occ[i] - eps_vir[a])
    """

    _SRC = r'''
    extern "C" __global__
    void ria_update(
        long long nrhs,
        long long nocc,
        long long nvir,
        double* ria,
        const double* tia,
        const double* eps_occ,
        const double* eps_vir
    ) {
        long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
        long long ntot = nrhs * nocc * nvir;

        if (idx < ntot) {
            long long rhs = idx / (nocc * nvir);
            long long ia = idx - rhs * nocc * nvir;
            long long i = ia / nvir;
            long long a = ia - i * nvir;
            ria[idx] += (eps_vir[a] - eps_occ[i]) * tia[idx];
        }
    }

    extern "C" __global__
    void apply_preconditioner(
        long long nrhs,
        long long nocc,
        long long nvir,
        double* zia,
        const double* ria,
        const double* eps_occ,
        const double* eps_vir
    ) {
        long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
        long long ntot = nrhs * nocc * nvir;

        if (idx < ntot) {
            long long rhs = idx / (nocc * nvir);
            long long ia = idx - rhs * nocc * nvir;
            long long i = ia / nvir;
            long long a = ia - i * nvir;
            zia[idx] = ria[idx] / (eps_occ[i] - eps_vir[a]);
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
        self._func_ria_update = None
        self._func_apply_preconditioner = None

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
                log = CPHFKernelHelper._get_nvrtc_log(prog)
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
        self._func_ria_update = self._check_driver(
            driver.cuModuleGetFunction(self._module, b"ria_update")
        )
        self._func_apply_preconditioner = self._check_driver(
            driver.cuModuleGetFunction(self._module, b"apply_preconditioner")
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
    def ria_update(self, *, nrhs, nocc, nvir, ria_ptr, tia_ptr, eps_occ_ptr, eps_vir_ptr, stream=0, synchronize=False):
        """
        Launch:
            ria[n, i, a] += (eps_vir[a] - eps_occ[i]) * tia[n, i, a]

        Arrays are assumed to be flattened row-major device buffers:
        - ria_ptr: shape (nocc, nvir)
        - tia_ptr: shape (nocc, nvir)
        - eps_occ_ptr: shape (nocc,)
        - eps_vir_ptr: shape (nvir,)
        """
        nrhs_arg = self._as_count(nrhs)
        nocc_arg = self._as_count(nocc)
        nvir_arg = self._as_count(nvir)
        ria_arg = self._as_device_ptr(ria_ptr)
        tia_arg = self._as_device_ptr(tia_ptr)
        eps_occ_arg = self._as_device_ptr(eps_occ_ptr)
        eps_vir_arg = self._as_device_ptr(eps_vir_ptr)

        _, params = self._kernel_params(
            nrhs_arg, nocc_arg, nvir_arg, ria_arg, tia_arg, eps_occ_arg, eps_vir_arg
        )

        ntot = nrhs_arg.value * nocc_arg.value * nvir_arg.value
        blocks = (ntot + self.threads_per_block - 1) // self.threads_per_block

        self._check_driver(
            driver.cuLaunchKernel(
                self._func_ria_update,
                blocks, 1, 1,
                self.threads_per_block, 1, 1,
                0,
                self._as_stream(stream),
                params,
                0,
            )
        )
        if synchronize:
            self._check_driver(driver.cuCtxSynchronize())

    def apply_preconditioner(self, *, nrhs, nocc, nvir, zia_ptr, ria_ptr, eps_occ_ptr, eps_vir_ptr, stream=0, synchronize=False):
        """
        Launch:
            zia[n, i, a] = ria[n, i, a] / (eps_occ[i] - eps_vir[a])

        Arrays are assumed to be flattened row-major device buffers:
        - zia_ptr: shape (nrhs, nocc, nvir)
        - ria_ptr: shape (nrhs, nocc, nvir)
        - eps_occ_ptr: shape (nocc,)
        - eps_vir_ptr: shape (nvir,)
        """
        nrhs_arg = self._as_count(nrhs)
        nocc_arg = self._as_count(nocc)
        nvir_arg = self._as_count(nvir)
        zia_arg = self._as_device_ptr(zia_ptr)
        ria_arg = self._as_device_ptr(ria_ptr)
        eps_occ_arg = self._as_device_ptr(eps_occ_ptr)
        eps_vir_arg = self._as_device_ptr(eps_vir_ptr)

        _, params = self._kernel_params(
            nrhs_arg, nocc_arg, nvir_arg, zia_arg, ria_arg, eps_occ_arg, eps_vir_arg
        )

        ntot = nrhs_arg.value * nocc_arg.value * nvir_arg.value
        blocks = (ntot + self.threads_per_block - 1) // self.threads_per_block

        self._check_driver(
            driver.cuLaunchKernel(
                self._func_apply_preconditioner,
                blocks, 1, 1,
                self.threads_per_block, 1, 1,
                0,
                self._as_stream(stream),
                params,
                0,
            )
        )
        if synchronize:
            self._check_driver(driver.cuCtxSynchronize())


