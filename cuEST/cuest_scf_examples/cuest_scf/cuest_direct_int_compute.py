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


import cuest.bindings as ce

from .cuest_handle import CuestHandle
from .cuest_direct_int_plan import CuestDirectIntPlan

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace
from .cuest_utility import CuestUtility, DevicePointerView

from .gpu_matrix import GPUMatrix
from .gpu_matrix_utility import GPUMatrixUtility

import numpy as np


class CuestDirectIntCompute(object):

    @staticmethod
    def coulomb(
        *,
        handle : CuestHandle,
        direct_int_plan : CuestDirectIntPlan,
        Dptr,
        Jptr,
        schwarz_threshold,
        num_density_matrices,
        ):

        Dptr2 = ce.Pointer()
        Dptr2.value = np.intp(Dptr)

        Jptr2 = ce.Pointer()
        Jptr2.value = np.intp(Jptr)

        compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS,
            [(ce.CuestDirectCoulombComputeParametersAttributes.CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS_DENSITY_SCHWARZ_THRESHOLD,
              ce.data_double(schwarz_threshold))],
            )

        # => Workspace query <= #

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestDirectCoulombComputeWorkspaceQuery(
            handle=handle.handle,
            plan=direct_int_plan.direct_int_plan_handle,
            parameters=compute_parameters.parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numDensityMatrices=num_density_matrices,
            densityMatrices=Dptr2,
            outCoulombMatrices=Jptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDirectCoulombComputeWorkspaceQuery failed: %s' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestDirectCoulombCompute(
            handle=handle.handle,
            plan=direct_int_plan.direct_int_plan_handle,
            parameters=compute_parameters.parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            numDensityMatrices=num_density_matrices,
            densityMatrices=Dptr2,
            outCoulombMatrices=Jptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDirectCoulombCompute failed: %s' % status)

    @staticmethod
    def symmetric_exchange(
        *,
        handle : CuestHandle,
        direct_int_plan : CuestDirectIntPlan,
        noccs,
        Coccptrs,
        Kptrs,
        schwarz_threshold,
        ):

        if not (len(noccs) == len(Coccptrs) == len(Kptrs)):
            raise RuntimeError('noccs, Coccptrs, and Kptrs must have the same length')

        nao = direct_int_plan.primary.nao

        # Coefficient-matrix entry point: the symmetric densities
        # D = Cocc^T Cocc are constructed here from the occupied coefficient
        # matrices, then handed to the density entry point
        densities = []
        for nocc, Coccptr in zip(noccs, Coccptrs):
            density = GPUMatrix(
                nrows=nao,
                ncols=nao,
                dtype=np.double,
                )
            GPUMatrixUtility.c_dgemm(
                transa=True,
                transb=False,
                m=nao,
                n=nao,
                k=nocc,
                alpha=1.0,
                a=DevicePointerView(Coccptr, nocc * nao),
                offa=0,
                lda=nao,
                b=DevicePointerView(Coccptr, nocc * nao),
                offb=0,
                ldb=nao,
                beta=0.0,
                c=density,
                offc=0,
                ldc=nao,
                )
            densities.append(density)

        CuestDirectIntCompute.symmetric_exchange_density(
            handle=handle,
            direct_int_plan=direct_int_plan,
            Dptrs=[_.pointer for _ in densities],
            Kptrs=Kptrs,
            schwarz_threshold=schwarz_threshold,
            )

    @staticmethod
    def symmetric_exchange_density(
        *,
        handle : CuestHandle,
        direct_int_plan : CuestDirectIntPlan,
        Dptrs,
        Kptrs,
        schwarz_threshold,
        ):

        if not (len(Dptrs) == len(Kptrs)):
            raise RuntimeError('Dptrs and Kptrs must have the same length')

        nao = direct_int_plan.primary.nao
        nmat = len(Dptrs)

        # Density-matrix entry point. The direct C API takes a contiguous
        # batch of symmetric density matrices (symmetry is required, positive
        # semidefiniteness is not); the input densities are gathered into the
        # batch here
        D_batch = GPUMatrix(
            nrows=nmat * nao,
            ncols=nao,
            dtype=np.double,
            )
        for m, Dptr in enumerate(Dptrs):
            GPUMatrixUtility.dcopy(
                n=nao * nao,
                x=DevicePointerView(Dptr, nao * nao),
                offx=0,
                incx=1,
                y=D_batch,
                offy=m * nao * nao,
                incy=1,
                )

        K_batch = GPUMatrix(
            nrows=nmat * nao,
            ncols=nao,
            dtype=np.double,
            )

        Dptr2 = ce.Pointer()
        Dptr2.value = np.intp(D_batch.pointer)

        Kptr2 = ce.Pointer()
        Kptr2.value = np.intp(K_batch.pointer)

        compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_DIRECTSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
            [(ce.CuestDirectSymmetricExchangeComputeParametersAttributes.CUEST_DIRECTSYMMETRICEXCHANGECOMPUTE_PARAMETERS_DENSITY_SCHWARZ_THRESHOLD,
              ce.data_double(schwarz_threshold))],
            )

        # => Workspace query <= #

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestDirectSymmetricExchangeComputeWorkspaceQuery(
            handle=handle.handle,
            plan=direct_int_plan.direct_int_plan_handle,
            parameters=compute_parameters.parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numDensityMatrices=nmat,
            densityMatrices=Dptr2,
            outExchangeMatrices=Kptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDirectSymmetricExchangeComputeWorkspaceQuery failed: %s' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestDirectSymmetricExchangeCompute(
            handle=handle.handle,
            plan=direct_int_plan.direct_int_plan_handle,
            parameters=compute_parameters.parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            numDensityMatrices=nmat,
            densityMatrices=Dptr2,
            outExchangeMatrices=Kptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDirectSymmetricExchangeCompute failed: %s' % status)

        for m, Kptr in enumerate(Kptrs):
            GPUMatrixUtility.dcopy(
                n=nao * nao,
                x=K_batch,
                offx=m * nao * nao,
                incx=1,
                y=DevicePointerView(Kptr, nao * nao),
                offy=0,
                incy=1,
                )

    @staticmethod
    def coulomb_gradient(
        *,
        handle : CuestHandle,
        direct_int_plan : CuestDirectIntPlan,
        Dptr,
        Gptr,
        schwarz_threshold,
        num_density_matrices,
        ):

        Dptr2 = ce.Pointer()
        Dptr2.value = np.intp(Dptr)

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(Gptr)

        compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_DIRECTCOULOMBDERIVATIVECOMPUTE_PARAMETERS,
            [(ce.CuestDirectCoulombDerivativeComputeParametersAttributes.CUEST_DIRECTCOULOMBDERIVATIVECOMPUTE_PARAMETERS_DENSITY_SCHWARZ_THRESHOLD,
              ce.data_double(schwarz_threshold))],
            )

        # => Workspace query <= #

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestDirectCoulombDerivativeComputeWorkspaceQuery(
            handle=handle.handle,
            plan=direct_int_plan.direct_int_plan_handle,
            parameters=compute_parameters.parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numDensityMatrices=num_density_matrices,
            densityMatrices=Dptr2,
            outCoulombGradients=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDirectCoulombDerivativeComputeWorkspaceQuery failed: %s' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestDirectCoulombDerivativeCompute(
            handle=handle.handle,
            plan=direct_int_plan.direct_int_plan_handle,
            parameters=compute_parameters.parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            numDensityMatrices=num_density_matrices,
            densityMatrices=Dptr2,
            outCoulombGradients=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDirectCoulombDerivativeCompute failed: %s' % status)

    @staticmethod
    def coulomb_bilinear_gradient(
        *,
        handle : CuestHandle,
        direct_int_plan : CuestDirectIntPlan,
        D1ptr,
        D2ptr,
        Gptr,
        schwarz_threshold,
        num_density_matrices,
        ):
        """Nuclear gradient of the bilinear Coulomb form for a batch of
        density pairs: the m-th gradient is d/dR Tr(D1_m J[D2_m]), with D1 and
        D2 stored contiguously as (num_density_matrices, nao, nao).  A separate
        C API entry point (cuestDirectCoulombBilinearDerivativeCompute) from the
        Coulomb-energy gradient coulomb_gradient.  The Schwarz threshold bounds
        each skipped contribution; pass schwarz_threshold = 0 for an unscreened
        result."""

        D1ptr2 = ce.Pointer()
        D1ptr2.value = np.intp(D1ptr)

        D2ptr2 = ce.Pointer()
        D2ptr2.value = np.intp(D2ptr)

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(Gptr)

        compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_DIRECTCOULOMBDERIVATIVECOMPUTE_PARAMETERS,
            [(ce.CuestDirectCoulombDerivativeComputeParametersAttributes.CUEST_DIRECTCOULOMBDERIVATIVECOMPUTE_PARAMETERS_DENSITY_SCHWARZ_THRESHOLD,
              ce.data_double(schwarz_threshold))],
            )

        # => Workspace query <= #

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestDirectCoulombBilinearDerivativeComputeWorkspaceQuery(
            handle=handle.handle,
            plan=direct_int_plan.direct_int_plan_handle,
            parameters=compute_parameters.parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numDensityMatrices=num_density_matrices,
            densityMatrices1=D1ptr2,
            densityMatrices2=D2ptr2,
            outCoulombGradients=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDirectCoulombBilinearDerivativeComputeWorkspaceQuery failed: %s' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestDirectCoulombBilinearDerivativeCompute(
            handle=handle.handle,
            plan=direct_int_plan.direct_int_plan_handle,
            parameters=compute_parameters.parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            numDensityMatrices=num_density_matrices,
            densityMatrices1=D1ptr2,
            densityMatrices2=D2ptr2,
            outCoulombGradients=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDirectCoulombBilinearDerivativeCompute failed: %s' % status)

    @staticmethod
    def symmetric_exchange_gradient(
        *,
        handle : CuestHandle,
        direct_int_plan : CuestDirectIntPlan,
        noccs,
        Coccptrs,
        Gptrs,
        schwarz_threshold,
        ):

        if not (len(noccs) == len(Coccptrs) == len(Gptrs)):
            raise RuntimeError('noccs, Coccptrs, and Gptrs must have the same length')

        nao = direct_int_plan.primary.nao

        # Coefficient-matrix entry point: the symmetric densities
        # D = Cocc^T Cocc are constructed here from the occupied coefficient
        # matrices, then handed to the density entry point
        densities = []
        for nocc, Coccptr in zip(noccs, Coccptrs):
            density = GPUMatrix(
                nrows=nao,
                ncols=nao,
                dtype=np.double,
                )
            GPUMatrixUtility.c_dgemm(
                transa=True,
                transb=False,
                m=nao,
                n=nao,
                k=nocc,
                alpha=1.0,
                a=DevicePointerView(Coccptr, nocc * nao),
                offa=0,
                lda=nao,
                b=DevicePointerView(Coccptr, nocc * nao),
                offb=0,
                ldb=nao,
                beta=0.0,
                c=density,
                offc=0,
                ldc=nao,
                )
            densities.append(density)

        CuestDirectIntCompute.symmetric_exchange_gradient_density(
            handle=handle,
            direct_int_plan=direct_int_plan,
            Dptrs=[_.pointer for _ in densities],
            Gptrs=Gptrs,
            schwarz_threshold=schwarz_threshold,
            )

    @staticmethod
    def symmetric_exchange_gradient_density(
        *,
        handle : CuestHandle,
        direct_int_plan : CuestDirectIntPlan,
        Dptrs,
        Gptrs,
        schwarz_threshold,
        ):

        if not (len(Dptrs) == len(Gptrs)):
            raise RuntimeError('Dptrs and Gptrs must have the same length')

        nao = direct_int_plan.primary.nao
        natom = direct_int_plan.primary.natom
        nmat = len(Dptrs)

        # Density-matrix entry point. The direct C API takes a contiguous
        # batch of symmetric density matrices (symmetry is required, positive
        # semidefiniteness is not) and returns the nuclear gradient of
        # Tr(K(D) D) at fixed density; the input densities are gathered into
        # the batch here
        D_batch = GPUMatrix(
            nrows=nmat * nao,
            ncols=nao,
            dtype=np.double,
            )
        for m, Dptr in enumerate(Dptrs):
            GPUMatrixUtility.dcopy(
                n=nao * nao,
                x=DevicePointerView(Dptr, nao * nao),
                offx=0,
                incx=1,
                y=D_batch,
                offy=m * nao * nao,
                incy=1,
                )

        G_batch = GPUMatrix(
            nrows=nmat * natom,
            ncols=3,
            dtype=np.double,
            )

        Dptr2 = ce.Pointer()
        Dptr2.value = np.intp(D_batch.pointer)

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(G_batch.pointer)

        compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_DIRECTSYMMETRICEXCHANGEDERIVATIVECOMPUTE_PARAMETERS,
            [(ce.CuestDirectSymmetricExchangeDerivativeComputeParametersAttributes.CUEST_DIRECTSYMMETRICEXCHANGEDERIVATIVECOMPUTE_PARAMETERS_DENSITY_SCHWARZ_THRESHOLD,
              ce.data_double(schwarz_threshold))],
            )

        # => Workspace query <= #

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestDirectSymmetricExchangeDerivativeComputeWorkspaceQuery(
            handle=handle.handle,
            plan=direct_int_plan.direct_int_plan_handle,
            parameters=compute_parameters.parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numDensityMatrices=nmat,
            densityMatrices=Dptr2,
            outExchangeGradients=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDirectSymmetricExchangeDerivativeComputeWorkspaceQuery failed: %s' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestDirectSymmetricExchangeDerivativeCompute(
            handle=handle.handle,
            plan=direct_int_plan.direct_int_plan_handle,
            parameters=compute_parameters.parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            numDensityMatrices=nmat,
            densityMatrices=Dptr2,
            outExchangeGradients=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDirectSymmetricExchangeDerivativeCompute failed: %s' % status)

        for m, Gptr in enumerate(Gptrs):
            GPUMatrixUtility.dcopy(
                n=natom * 3,
                x=G_batch,
                offx=m * natom * 3,
                incx=1,
                y=DevicePointerView(Gptr, natom * 3),
                offy=0,
                incy=1,
                )

    @staticmethod
    def symmetric_exchange_bilinear_gradient(
        *,
        handle : CuestHandle,
        direct_int_plan : CuestDirectIntPlan,
        noccs1,
        Coccptrs1,
        noccs2,
        Coccptrs2,
        Gptrs,
        schwarz_threshold,
        ):
        """Nuclear gradient of the bilinear exchange form for a batch of
        density pairs: the m-th gradient is d/dR Tr(D1_m K[D2_m]) with
        D1_m = Cocc1_m^T Cocc1_m and D2_m = Cocc2_m^T Cocc2_m.  The densities
        are built here and passed to symmetric_exchange_bilinear_gradient_density
        (the density-matrix entry point, which also takes indefinite
        densities such as a delta density)."""

        if not (len(noccs1) == len(Coccptrs1) == len(noccs2) == len(Coccptrs2) == len(Gptrs)):
            raise RuntimeError('noccs1, Coccptrs1, noccs2, Coccptrs2, and Gptrs must have the same length')

        nao = direct_int_plan.primary.nao

        def build_densities(noccs_, Coccptrs_):
            densities_ = []
            for nocc, Coccptr in zip(noccs_, Coccptrs_):
                density = GPUMatrix(
                    nrows=nao,
                    ncols=nao,
                    dtype=np.double,
                    )
                GPUMatrixUtility.c_dgemm(
                    transa=True,
                    transb=False,
                    m=nao,
                    n=nao,
                    k=nocc,
                    alpha=1.0,
                    a=DevicePointerView(Coccptr, nocc * nao),
                    offa=0,
                    lda=nao,
                    b=DevicePointerView(Coccptr, nocc * nao),
                    offb=0,
                    ldb=nao,
                    beta=0.0,
                    c=density,
                    offc=0,
                    ldc=nao,
                    )
                densities_.append(density)
            return densities_

        densities1 = build_densities(noccs1, Coccptrs1)
        densities2 = build_densities(noccs2, Coccptrs2)

        CuestDirectIntCompute.symmetric_exchange_bilinear_gradient_density(
            handle=handle,
            direct_int_plan=direct_int_plan,
            D1ptrs=[_.pointer for _ in densities1],
            D2ptrs=[_.pointer for _ in densities2],
            Gptrs=Gptrs,
            schwarz_threshold=schwarz_threshold,
            )

    @staticmethod
    def symmetric_exchange_bilinear_gradient_density(
        *,
        handle : CuestHandle,
        direct_int_plan : CuestDirectIntPlan,
        D1ptrs,
        D2ptrs,
        Gptrs,
        schwarz_threshold,
        ):
        """Nuclear gradient of the bilinear exchange form for a batch of
        symmetric density pairs given as device pointers: the m-th gradient
        is d/dR Tr(D1_m K[D2_m]).  A separate C API entry point
        (cuestDirectSymmetricExchangeBilinearDerivativeCompute) from the
        exchange-energy gradient symmetric_exchange_gradient_density.
        Symmetry is required, positive semidefiniteness is not (a guess or
        delta density against a converged one is the typical use).  Both
        densities of a pair enter the Schwarz screening, each bounding its
        own side of the integral (pass schwarz_threshold = 0 for an
        unscreened result)."""

        if not (len(D1ptrs) == len(D2ptrs) == len(Gptrs)):
            raise RuntimeError('D1ptrs, D2ptrs, and Gptrs must have the same length')

        nao = direct_int_plan.primary.nao
        natom = direct_int_plan.primary.natom
        nmat = len(D1ptrs)

        def gather_density_batch(Dptrs_):
            D_batch_ = GPUMatrix(
                nrows=nmat * nao,
                ncols=nao,
                dtype=np.double,
                )
            for m, Dptr in enumerate(Dptrs_):
                GPUMatrixUtility.dcopy(
                    n=nao * nao,
                    x=DevicePointerView(Dptr, nao * nao),
                    offx=0,
                    incx=1,
                    y=D_batch_,
                    offy=m * nao * nao,
                    incy=1,
                    )
            return D_batch_

        D1_batch = gather_density_batch(D1ptrs)
        D2_batch = gather_density_batch(D2ptrs)

        G_batch = GPUMatrix(
            nrows=nmat * natom,
            ncols=3,
            dtype=np.double,
            )

        D1ptr2 = ce.Pointer()
        D1ptr2.value = np.intp(D1_batch.pointer)

        D2ptr2 = ce.Pointer()
        D2ptr2.value = np.intp(D2_batch.pointer)

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(G_batch.pointer)

        compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_DIRECTSYMMETRICEXCHANGEDERIVATIVECOMPUTE_PARAMETERS,
            [(ce.CuestDirectSymmetricExchangeDerivativeComputeParametersAttributes.CUEST_DIRECTSYMMETRICEXCHANGEDERIVATIVECOMPUTE_PARAMETERS_DENSITY_SCHWARZ_THRESHOLD,
              ce.data_double(schwarz_threshold))],
            )

        # => Workspace query <= #

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestDirectSymmetricExchangeBilinearDerivativeComputeWorkspaceQuery(
            handle=handle.handle,
            plan=direct_int_plan.direct_int_plan_handle,
            parameters=compute_parameters.parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numDensityMatrices=nmat,
            densityMatrices1=D1ptr2,
            densityMatrices2=D2ptr2,
            outExchangeGradients=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDirectSymmetricExchangeBilinearDerivativeComputeWorkspaceQuery failed: %s' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestDirectSymmetricExchangeBilinearDerivativeCompute(
            handle=handle.handle,
            plan=direct_int_plan.direct_int_plan_handle,
            parameters=compute_parameters.parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            numDensityMatrices=nmat,
            densityMatrices1=D1ptr2,
            densityMatrices2=D2ptr2,
            outExchangeGradients=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDirectSymmetricExchangeBilinearDerivativeCompute failed: %s' % status)

        for m, Gptr in enumerate(Gptrs):
            GPUMatrixUtility.dcopy(
                n=natom * 3,
                x=G_batch,
                offx=m * natom * 3,
                incx=1,
                y=DevicePointerView(Gptr, natom * 3),
                offy=0,
                incy=1,
                )
