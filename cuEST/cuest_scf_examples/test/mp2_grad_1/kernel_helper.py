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
    - Device pointers are passed as Python int, so a caller holding a GPUMatrix may offset
      into it by hand (pointer + element_offset * 8) to address one block of a larger buffer.
    - Shape/length information is passed explicitly.
    - Kernels are JIT-compiled in __init__ and ready to use afterward.

    Methods
    -------
    make_pair_amplitudes(i, j, nocc, nvir, eps_ptr, pair_mo_ints_ptr, pair_amplitudes_ptr, ...)
        Compute MP2 amplitudes for a given i, j (occupied) integral pair:
            amplitudes[a, b] = (i a | j b) / (eps[i] + eps[j] - eps[a] - eps[b])
    make_pair_tau(nvir, pair_amplitudes_ptr, pair_tau_ptr, ...)
        Contravariant combination of one amplitude pair:
            tau[a, b] = 2 t[a, b] - t[b, a]
    transpose_blocks(nblocks, nrows, ncols, in_ptr, out_ptr, ...)
        Transpose each of nblocks contiguous row-major (nrows, ncols) blocks in place of the
        matching (ncols, nrows) output block.
    apply_denominator(nvir, nocc, invert, alpha, beta, eps_occ_ptr, eps_vir_ptr, x_ptr,
                      y_ptr, ...)
        Weight a virtual-occupied quantity by the orbital energy difference, either as a
        multiplication (the diagonal of the orbital Hessian) or a division (its inverse,
        used to precondition the z-vector equations).
    scale_rows(nrows, ncols, scales_ptr, matrix_ptr, ...)
        Scale row r of a row-major matrix by scales[r].
    scatter_block(nrows, ncols, ld_dst, alpha, src_ptr, dst_ptr, ...)
        Write alpha * src into a (nrows, ncols) sub-block of a wider row-major matrix.
    set_diagonal(count, ld, value, matrix_ptr, ...)
        Set the leading `count` diagonal entries of a row-major matrix to `value`.
    make_W(nmo, nocc, eps_ptr, dm1_ptr, W_ptr, ...)
        Energy-weight the relaxed one-particle density: the occupied-virtual blocks carry
        the occupied orbital energy, every other block the average of the two.
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

    extern "C" __global__
    void make_pair_tau(
        long long nvir,
        const double* pair_amplitudes,
        double* pair_tau
    ) {
        long long rel_a = (long long) blockIdx.x * blockDim.x + threadIdx.x;
        long long rel_b = (long long) blockIdx.y * blockDim.y + threadIdx.y;

        if (rel_a < nvir && rel_b < nvir) {
            pair_tau[rel_a * nvir + rel_b] =
                2.0 * pair_amplitudes[rel_a * nvir + rel_b] - pair_amplitudes[rel_b * nvir + rel_a];
        }
    }

    extern "C" __global__
    void transpose_blocks(
        long long nblocks,
        long long nrows,
        long long ncols,
        const double* in,
        double* out
    ) {
        long long row = (long long) blockIdx.x * blockDim.x + threadIdx.x;
        long long col = (long long) blockIdx.y * blockDim.y + threadIdx.y;
        long long block = (long long) blockIdx.z;

        if (row < nrows && col < ncols && block < nblocks) {
            out[block * nrows * ncols + col * nrows + row] =
                in[block * nrows * ncols + row * ncols + col];
        }
    }

    extern "C" __global__
    void apply_denominator(
        long long nvir,
        long long nocc,
        long long invert,
        double alpha,
        double beta,
        const double* eps_occ,
        const double* eps_vir,
        const double* x,
        double* y
    ) {
        long long a = (long long) blockIdx.x * blockDim.x + threadIdx.x;
        long long i = (long long) blockIdx.y * blockDim.y + threadIdx.y;

        if (a < nvir && i < nocc) {
            long long idx = a * nocc + i;
            double denominator = eps_vir[a] - eps_occ[i];
            double weight = invert ? 1.0 / denominator : denominator;
            y[idx] = beta * y[idx] + alpha * x[idx] * weight;
        }
    }

    extern "C" __global__
    void scale_rows(
        long long nrows,
        long long ncols,
        const double* scales,
        double* matrix
    ) {
        long long row = (long long) blockIdx.x * blockDim.x + threadIdx.x;
        long long col = (long long) blockIdx.y * blockDim.y + threadIdx.y;

        if (row < nrows && col < ncols) {
            matrix[row * ncols + col] *= scales[row];
        }
    }

    extern "C" __global__
    void scatter_block(
        long long nrows,
        long long ncols,
        long long ld_dst,
        double alpha,
        const double* src,
        double* dst
    ) {
        long long row = (long long) blockIdx.x * blockDim.x + threadIdx.x;
        long long col = (long long) blockIdx.y * blockDim.y + threadIdx.y;

        if (row < nrows && col < ncols) {
            dst[row * ld_dst + col] = alpha * src[row * ncols + col];
        }
    }

    extern "C" __global__
    void set_diagonal(
        long long count,
        long long ld,
        double value,
        double* matrix
    ) {
        long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < count) {
            matrix[idx * ld + idx] = value;
        }
    }

    extern "C" __global__
    void make_W(
        long long nmo,
        long long nocc,
        const double* eps,
        const double* dm1,
        double* W
    ) {
        long long p = (long long) blockIdx.x * blockDim.x + threadIdx.x;
        long long q = (long long) blockIdx.y * blockDim.y + threadIdx.y;

        if (p < nmo && q < nmo) {
            double weight;
            if (p >= nocc && q < nocc) {
                weight = eps[q];
            } else if (p < nocc && q >= nocc) {
                weight = eps[p];
            } else {
                weight = 0.5 * (eps[p] + eps[q]);
            }
            W[p * nmo + q] = weight * dm1[p * nmo + q];
        }
    }
    '''

    def __init__(
        self,
        device_id=0,
        threads_per_block=256,
        gpu_arch=None,
        ):
        self.device_id = int(device_id)
        self.threads_per_block = int(threads_per_block)
        self.gpu_arch = gpu_arch

        self._dev = None
        self._ctx = None
        self._module = None

        # Function signatures get stored here
        self._func_make_pair_amplitudes = None
        self._func_make_pair_tau = None
        self._func_transpose_blocks = None
        self._func_apply_denominator = None
        self._func_scale_rows = None
        self._func_scatter_block = None
        self._func_set_diagonal = None
        self._func_make_W = None

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
    def _check_nvrtc(
        result,
        prog=None,
        ):
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

    def _get_nvrtc_ptx(self, prog):
        # compute_* targets only yield PTX; sm_* targets yield a cubin
        is_ptx = self.gpu_arch.startswith(b"compute_")
        get_size = nvrtc.nvrtcGetPTXSize if is_ptx else nvrtc.nvrtcGetCUBINSize
        get_code = nvrtc.nvrtcGetPTX if is_ptx else nvrtc.nvrtcGetCUBIN
        _, ptx_size = get_size(prog)
        ptx_size = int(ptx_size)
        if ptx_size <= 0:
            raise RuntimeError("NVRTC returned empty PTX size")
        buf = bytearray(ptx_size)
        err = get_code(prog, buf)
        if err[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC error: {err[0]}")
        return bytes(buf)

    def _init_cuda(self):
        self._check_driver(driver.cuInit(0))
        self._dev = self._check_driver(driver.cuDeviceGet(self.device_id))
        self._ctx = self._check_driver(driver.cuDevicePrimaryCtxRetain(self._dev))
        self._check_driver(driver.cuCtxSetCurrent(self._ctx))

    def _max_supported_arch(self):
        """Highest virtual architecture (as an integer, e.g. 90 for compute_90)
        that this toolkit's NVRTC can compile for."""
        archs = self._check_nvrtc(nvrtc.nvrtcGetSupportedArchs())
        return max(int(a) for a in archs)

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
        device_arch = major * 10 + minor

        # Emit PTX for the highest arch supported by the current NVRTC
        max_arch = self._max_supported_arch()
        # Cubin (sm_*) needs no PTX ISA support from the driver but only runs on its
        # own arch; if NVRTC is older than the device, emit PTX for the driver to JIT.
        if max_arch < device_arch:
            return f"compute_{max_arch}".encode()
        return f"sm_{device_arch}".encode()

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
        self._func_make_pair_tau = self._check_driver(
            driver.cuModuleGetFunction(self._module, b"make_pair_tau")
        )
        self._func_transpose_blocks = self._check_driver(
            driver.cuModuleGetFunction(self._module, b"transpose_blocks")
        )
        self._func_apply_denominator = self._check_driver(
            driver.cuModuleGetFunction(self._module, b"apply_denominator")
        )
        self._func_scale_rows = self._check_driver(
            driver.cuModuleGetFunction(self._module, b"scale_rows")
        )
        self._func_scatter_block = self._check_driver(
            driver.cuModuleGetFunction(self._module, b"scatter_block")
        )
        self._func_set_diagonal = self._check_driver(
            driver.cuModuleGetFunction(self._module, b"set_diagonal")
        )
        self._func_make_W = self._check_driver(
            driver.cuModuleGetFunction(self._module, b"make_W")
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
    def _as_scalar(value):
        return ctypes.c_double(float(value))

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
    def make_pair_amplitudes(
        self,
        *,
        i,
        j,
        nocc,
        nvir,
        eps_ptr,
        pair_mo_ints_ptr,
        pair_amplitudes_ptr,
        stream=0,
        synchronize=False,
        ):
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
            i_arg,
            j_arg,
            nocc_arg,
            nvir_arg,
            pair_mo_ints_arg,
            eps_arg,
            pair_amplitudes_arg,
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

    def make_pair_tau(
        self,
        *,
        nvir,
        pair_amplitudes_ptr,
        pair_tau_ptr,
        stream=0,
        synchronize=False,
        ):
        """
        Launch:
            pair_tau[a, b] = 2 pair_amplitudes[a, b] - pair_amplitudes[b, a]

        Both buffers are flattened row-major (nvir, nvir) device blocks and must not alias.
        """
        nvir_arg = self._as_count(nvir)
        pair_amplitudes_arg = self._as_device_ptr(pair_amplitudes_ptr)
        pair_tau_arg = self._as_device_ptr(pair_tau_ptr)

        _, params = self._kernel_params(
            nvir_arg,
            pair_amplitudes_arg,
            pair_tau_arg,
            )

        block_xy = min(32, self.threads_per_block)
        blocks_x = (nvir_arg.value + block_xy - 1) // block_xy

        self._check_driver(
            driver.cuLaunchKernel(
                self._func_make_pair_tau,
                blocks_x, blocks_x, 1,
                block_xy, block_xy, 1,
                0,
                self._as_stream(stream),
                params,
                0,
            )
        )
        if synchronize:
            self._check_driver(driver.cuCtxSynchronize())

    def transpose_blocks(
        self,
        *,
        nblocks,
        nrows,
        ncols,
        in_ptr,
        out_ptr,
        stream=0,
        synchronize=False,
        ):
        """
        Launch:
            out[block, col, row] = in[block, row, col]

        The input is nblocks contiguous row-major (nrows, ncols) blocks; the output is the
        matching sequence of (ncols, nrows) blocks. The buffers must not alias.
        """
        nblocks_arg = self._as_count(nblocks)
        nrows_arg = self._as_count(nrows)
        ncols_arg = self._as_count(ncols)
        in_arg = self._as_device_ptr(in_ptr)
        out_arg = self._as_device_ptr(out_ptr)

        _, params = self._kernel_params(
            nblocks_arg,
            nrows_arg,
            ncols_arg,
            in_arg,
            out_arg,
            )

        block_xy = min(32, self.threads_per_block)
        blocks_x = (nrows_arg.value + block_xy - 1) // block_xy
        blocks_y = (ncols_arg.value + block_xy - 1) // block_xy

        self._check_driver(
            driver.cuLaunchKernel(
                self._func_transpose_blocks,
                blocks_x, blocks_y, nblocks_arg.value,
                block_xy, block_xy, 1,
                0,
                self._as_stream(stream),
                params,
                0,
            )
        )
        if synchronize:
            self._check_driver(driver.cuCtxSynchronize())


    def apply_denominator(
        self,
        *,
        nvir,
        nocc,
        invert,
        alpha,
        beta,
        eps_occ_ptr,
        eps_vir_ptr,
        x_ptr,
        y_ptr,
        stream=0,
        synchronize=False,
        ):
        """
        Launch, with d[a, i] = eps_vir[a] - eps_occ[i]:
            y[a, i] = beta * y[a, i] + alpha * x[a, i] * (invert ? 1 / d[a, i] : d[a, i])

        Both x and y are flattened row-major (nvir, nocc) device blocks and may alias.
        """
        nvir_arg = self._as_count(nvir)
        nocc_arg = self._as_count(nocc)
        invert_arg = self._as_count(1 if invert else 0)
        alpha_arg = self._as_scalar(alpha)
        beta_arg = self._as_scalar(beta)
        eps_occ_arg = self._as_device_ptr(eps_occ_ptr)
        eps_vir_arg = self._as_device_ptr(eps_vir_ptr)
        x_arg = self._as_device_ptr(x_ptr)
        y_arg = self._as_device_ptr(y_ptr)

        _, params = self._kernel_params(
            nvir_arg,
            nocc_arg,
            invert_arg,
            alpha_arg,
            beta_arg,
            eps_occ_arg,
            eps_vir_arg,
            x_arg,
            y_arg,
            )

        block_xy = min(32, self.threads_per_block)
        blocks_x = (nvir_arg.value + block_xy - 1) // block_xy
        blocks_y = (nocc_arg.value + block_xy - 1) // block_xy

        self._check_driver(
            driver.cuLaunchKernel(
                self._func_apply_denominator,
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

    def scale_rows(
        self,
        *,
        nrows,
        ncols,
        scales_ptr,
        matrix_ptr,
        stream=0,
        synchronize=False,
        ):
        """
        Launch:
            matrix[row, col] *= scales[row]

        The matrix is a flattened row-major (nrows, ncols) device block.
        """
        nrows_arg = self._as_count(nrows)
        ncols_arg = self._as_count(ncols)
        scales_arg = self._as_device_ptr(scales_ptr)
        matrix_arg = self._as_device_ptr(matrix_ptr)

        _, params = self._kernel_params(
            nrows_arg,
            ncols_arg,
            scales_arg,
            matrix_arg,
            )

        block_xy = min(32, self.threads_per_block)
        blocks_x = (nrows_arg.value + block_xy - 1) // block_xy
        blocks_y = (ncols_arg.value + block_xy - 1) // block_xy

        self._check_driver(
            driver.cuLaunchKernel(
                self._func_scale_rows,
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

    def scatter_block(
        self,
        *,
        nrows,
        ncols,
        ld_dst,
        alpha,
        src_ptr,
        dst_ptr,
        stream=0,
        synchronize=False,
        ):
        """
        Launch:
            dst[row, col] = alpha * src[row, col]

        src is a flattened row-major (nrows, ncols) block; dst is the corresponding block of
        a wider row-major matrix of row stride ld_dst, whose pointer the caller has already
        offset to the corner of the target block.
        """
        nrows_arg = self._as_count(nrows)
        ncols_arg = self._as_count(ncols)
        ld_dst_arg = self._as_count(ld_dst)
        alpha_arg = self._as_scalar(alpha)
        src_arg = self._as_device_ptr(src_ptr)
        dst_arg = self._as_device_ptr(dst_ptr)

        _, params = self._kernel_params(
            nrows_arg,
            ncols_arg,
            ld_dst_arg,
            alpha_arg,
            src_arg,
            dst_arg,
            )

        block_xy = min(32, self.threads_per_block)
        blocks_x = (nrows_arg.value + block_xy - 1) // block_xy
        blocks_y = (ncols_arg.value + block_xy - 1) // block_xy

        self._check_driver(
            driver.cuLaunchKernel(
                self._func_scatter_block,
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

    def set_diagonal(
        self,
        *,
        count,
        ld,
        value,
        matrix_ptr,
        stream=0,
        synchronize=False,
        ):
        """
        Launch:
            matrix[idx, idx] = value   for idx < count
        """
        count_arg = self._as_count(count)
        ld_arg = self._as_count(ld)
        value_arg = self._as_scalar(value)
        matrix_arg = self._as_device_ptr(matrix_ptr)

        _, params = self._kernel_params(
            count_arg,
            ld_arg,
            value_arg,
            matrix_arg,
            )

        blocks = (count_arg.value + self.threads_per_block - 1) // self.threads_per_block

        self._check_driver(
            driver.cuLaunchKernel(
                self._func_set_diagonal,
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

    def make_W(
        self,
        *,
        nmo,
        nocc,
        eps_ptr,
        dm1_ptr,
        W_ptr,
        stream=0,
        synchronize=False,
        ):
        """
        Launch the energy weighting of the relaxed one-particle density:
            W[p, q] = w[p, q] dm1[p, q]
        with w taken as the occupied orbital energy in the two occupied-virtual blocks and
        as the average of the two orbital energies everywhere else.
        """
        nmo_arg = self._as_count(nmo)
        nocc_arg = self._as_count(nocc)
        eps_arg = self._as_device_ptr(eps_ptr)
        dm1_arg = self._as_device_ptr(dm1_ptr)
        W_arg = self._as_device_ptr(W_ptr)

        _, params = self._kernel_params(
            nmo_arg,
            nocc_arg,
            eps_arg,
            dm1_arg,
            W_arg,
            )

        block_xy = min(32, self.threads_per_block)
        blocks = (nmo_arg.value + block_xy - 1) // block_xy

        self._check_driver(
            driver.cuLaunchKernel(
                self._func_make_W,
                blocks, blocks, 1,
                block_xy, block_xy, 1,
                0,
                self._as_stream(stream),
                params,
                0,
            )
        )
        if synchronize:
            self._check_driver(driver.cuCtxSynchronize())
