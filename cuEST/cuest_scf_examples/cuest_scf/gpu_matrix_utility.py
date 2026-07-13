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
import atexit

import nvmath.bindings.cublas as cublas
import nvmath.bindings.cusolverDn as cusolver
import nvmath.bindings.cusolver as cusolver_base
import cuda.bindings.runtime as cuda
import ctypes

from .gpu_matrix import GPUMatrix

# This is a common trick to ensure an active CUDA context exists
cuda.cudaFree(0)
# These handles are module-scoped, making them singletons
cublas_handle = cublas.create()
# Read scalar parameters like alpha and beta from CPU RAM
cublas.set_pointer_mode(
    cublas_handle,
    cublas.PointerMode.HOST
    )
cusolver_handle = cusolver.create()

def _delete_handles():
    global cublas_handle, cusolver_handle
    cublas.destroy(cublas_handle)
    cusolver.destroy(cusolver_handle)

atexit.register(_delete_handles)

class GPUMatrixUtility(object):

    @staticmethod
    def matrix_multiply(
        *,
        mat1,
        mat2,
        transpose1,
        transpose2,
        scale=1.0,
        ):

        alpha = np.array(
            scale,
            dtype=np.double,
            )
        beta = np.array(
            0.0,
            dtype=np.double,
            )
        m = mat1.ncols if transpose1 else mat1.nrows
        n = mat2.nrows if transpose2 else mat2.ncols
        k1 = mat1.nrows if transpose1 else mat1.ncols
        k2 = mat2.ncols if transpose2 else mat2.nrows
        assert k1 == k2

        c = GPUMatrix(
            nrows=m,
            ncols=n,
            )

        cublas.dgemm(
            handle=cublas_handle,
            transa=cublas.Operation.T if transpose2 else cublas.Operation.N,
            transb=cublas.Operation.T if transpose1 else cublas.Operation.N,
            m=n,
            n=m,
            k=k1,
            alpha=alpha.ctypes.data,
            a=mat2.pointer,
            lda=mat2.ncols,
            b=mat1.pointer,
            ldb=mat1.ncols,
            beta=beta.ctypes.data,
            c=c.pointer,
            ldc=c.ncols,
            )

        return c


    @staticmethod
    def ddot(
        *,
        n=None,
        x,
        offx=0,
        incx=1,
        y,
        offy=0,
        incy=1,
        ):
    
        if n is None:
            n = x.size
    
        assert offx + (n - 1) * incx < x.size
        assert offy + (n - 1) * incy < y.size
    
        result = np.empty(1, dtype=np.double)
    
        cublas.ddot(
            handle=cublas_handle,
            n=n,
            x=int(x.pointer) + offx * ctypes.sizeof(ctypes.c_double),
            incx=incx,
            y=int(y.pointer) + offy * ctypes.sizeof(ctypes.c_double),
            incy=incy,
            result=result.ctypes.data,
            )
        return result[0]


    @staticmethod
    def dcopy(
        *,
        n=None,
        x,
        offx=0,
        incx=1,
        y,
        offy=0,
        incy=1,
        ):

        if n is None:
            n=x.size

        assert offx + (n - 1) * incx < x.size
        assert offy + (n - 1) * incy < y.size

        cublas.dcopy(
            handle = cublas_handle,
            n=n,
            x=int(x.pointer) + offx * ctypes.sizeof(ctypes.c_double),
            incx=incx,
            y=int(y.pointer) + offy * ctypes.sizeof(ctypes.c_double),
            incy=incy,
            )


    @staticmethod
    def dgeam(
        *,
        transpose1,
        alpha,
        mat1,
        transpose2,
        beta,
        mat2,
        ):

        assert mat1.nrows == mat1.ncols
        assert mat2.nrows == mat2.ncols
        assert mat1.nrows == mat2.nrows

        C = mat1.clone()

        alpha=np.array(
            alpha,
            dtype=np.double,
            )
        beta=np.array(
            beta,
            dtype=np.double,
            )

        cublas.dgeam(
            handle=cublas_handle,
            transa=cublas.Operation.T if transpose1 else cublas.Operation.N,
            transb=cublas.Operation.T if transpose2 else cublas.Operation.N,
            m=mat1.nrows,
            n=mat2.ncols,
            alpha=alpha.ctypes.data,
            a=mat1.pointer,
            lda=mat1.ncols,
            beta=beta.ctypes.data,
            b=mat2.pointer,
            ldb=mat2.ncols,
            c=C.pointer,
            ldc=C.ncols,
            )

        return C


    @staticmethod
    def daxpy(
        *,
        n=None,
        alpha,
        x,
        offx=0,
        incx=1,
        y,
        offy=0,
        incy=1,
        ):

        alpha = np.array(
            alpha,
            dtype=np.double,
            )
    
        if n is None:
            n = x.size
    
        assert offx + (n - 1) * incx < x.size
        assert offy + (n - 1) * incy < y.size
    
        cublas.daxpy(
            handle=cublas_handle,
            n=n,
            alpha=alpha.ctypes.data,
            x=int(x.pointer) + offx * ctypes.sizeof(ctypes.c_double),
            incx=incx,
            y=int(y.pointer) + offy * ctypes.sizeof(ctypes.c_double),
            incy=incy,
            )


    @staticmethod
    def dscal(
        *,
        n=None,
        alpha,
        x,
        offx=0,
        incx=1,
        ):
    
        alpha = np.array(
            alpha,
            dtype=np.double,
            )
    
        if n is None:
            n = x.size
    
        assert offx + (n - 1) * incx < x.size
    
        cublas.dscal(
            handle=cublas_handle,
            n=n,
            alpha=alpha.ctypes.data,
            x=int(x.pointer) + offx * ctypes.sizeof(ctypes.c_double),
            incx=incx,
            )


    @staticmethod
    def c_dgemm(
        *,
        transa : bool,
        transb : bool,
        m,
        n,
        k,
        alpha,
        a,
        offa,
        lda,
        b,
        offb,
        ldb,
        beta,
        c,
        offc,
        ldc,
        ):
 
        alpha = np.array(
            alpha,
            dtype=np.double,
            )
        beta = np.array(
            beta,
            dtype=np.double,
            )

        cublas.dgemm(
            handle=cublas_handle,
            transa=cublas.Operation.T if transb else cublas.Operation.N,
            transb=cublas.Operation.T if transa else cublas.Operation.N,
            m=n,
            n=m,
            k=k,
            alpha=alpha.ctypes.data,
            a=int(b.pointer) + offb * ctypes.sizeof(ctypes.c_double),
            lda=ldb,
            b=int(a.pointer) + offa * ctypes.sizeof(ctypes.c_double),
            ldb=lda,
            beta=beta.ctypes.data,
            c=int(c.pointer) + offc * ctypes.sizeof(ctypes.c_double),
            ldc=ldc,
            )


    @staticmethod
    def eigh(
        *,
        matrix
        ):

        jobz = cusolver_base.EigMode.VECTOR
        uplo = cublas.FillMode.LOWER

        assert matrix.nrows == matrix.ncols

        evecs = matrix.clone()
        evals = GPUMatrix(
            nrows=matrix.nrows,
            ncols=1,
            dtype=matrix.dtype,
            initialize=False,
            )

        d_info = GPUMatrix(
            nrows=1,
            ncols=1,
            dtype=np.int64,
            )

        # Workspace query
        lwork = np.empty(
            1,
            dtype=np.int64
            )
        lwork = cusolver.dsyevd_buffer_size(
            handle=cusolver_handle,
            jobz=jobz,
            uplo=uplo,
            n=evecs.ncols,
            a=evecs.pointer,
            lda=evecs.ncols,
            w=evals.pointer,
        )

        d_work = GPUMatrix(
            nrows=lwork,
            ncols=1,
            dtype=np.double,
            )

        # Compute eigenvalues/eigenvectors in-place
        cusolver.dsyevd(
            handle=cusolver_handle,
            jobz=jobz,
            uplo=uplo,
            n=evecs.ncols,
            a=evecs.pointer,
            lda=evecs.ncols,
            w=evals.pointer,
            work=d_work.pointer,
            lwork=lwork,
            info=d_info.pointer,
        )

        info_host = d_info.to_numpy()
        if info_host[0, 0] != 0:
            raise RuntimeError(f"dsyevd failed with info = {info_host[0, 0]}")

        evecs = GPUMatrixUtility.dgeam(
            transpose1=True,
            alpha=1.0,
            mat1=evecs,
            transpose2=False,
            beta=0.0,
            mat2=evecs,
            )

        return evals, evecs
