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
from .cuest_gridk_int_plan import CuestGridKIntPlan

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace
from .cuest_utility import CuestUtility, DevicePointerView

from .gpu_matrix import GPUMatrix
from .gpu_matrix_utility import GPUMatrixUtility

import numpy as np


class CuestGridKIntCompute(object):

    @staticmethod
    def symmetric_exchange(
        *,
        handle : CuestHandle,
        gridk_int_plan : CuestGridKIntPlan,
        noccs,
        Coccptrs,
        Kptrs,
        ):

        if not (len(noccs) == len(Coccptrs) == len(Kptrs)):
            raise RuntimeError('noccs, Coccptrs, and Kptrs must have the same length')

        nao = gridk_int_plan.primary.nao

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

        CuestGridKIntCompute.symmetric_exchange_density(
            handle=handle,
            gridk_int_plan=gridk_int_plan,
            Dptrs=[_.pointer for _ in densities],
            Kptrs=Kptrs,
            )

    @staticmethod
    def symmetric_exchange_density(
        *,
        handle : CuestHandle,
        gridk_int_plan : CuestGridKIntPlan,
        Dptrs,
        Kptrs,
        ):

        if not (len(Dptrs) == len(Kptrs)):
            raise RuntimeError('Dptrs and Kptrs must have the same length')

        # Density-matrix entry point. Each density must be symmetric (the C
        # API restores operator symmetry as K = (K~ + K~^T)/2, which recovers
        # the operator-symmetric exchange only for a symmetric density), but
        # need not be positive semidefinite: an indefinite density such as a
        # difference density is valid here and does not require the roughly
        # twice as expensive nonsymmetric path

        gridk_compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_GRIDKSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
            [],
            )

        # Allow the algorithm to use up to 4GB of DRAM as the grid-blocking budget.
        # This may need to be increased to handle very large systems.
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        exchangeint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=4000000000,
            )

        for Dptr, Kptr in zip(Dptrs, Kptrs):

            Dptr2 = ce.Pointer()
            Dptr2.value = np.intp(Dptr)

            Kptr2 = ce.Pointer()
            Kptr2.value = np.intp(Kptr)

            # => Workspace query <= #

            temporary_workspace_descriptor = CuestWorkspaceDescriptor()

            status = ce.cuestGridKSymmetricExchangeComputeWorkspaceQuery(
                handle=handle.handle,
                plan=gridk_int_plan.gridk_int_plan_handle,
                parameters=gridk_compute_parameters.parameters,
                variableBufferSize=exchangeint_variable_buffer_descriptor.pointer,
                temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
                numDensityMatrices=1,
                densityMatrices=Dptr2,
                outExchangeMatrices=Kptr2,
                )

            if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                raise RuntimeError('cuestGridKSymmetricExchangeComputeWorkspaceQuery failed: %s' % status)

            temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

            status = ce.cuestGridKSymmetricExchangeCompute(
                handle=handle.handle,
                plan=gridk_int_plan.gridk_int_plan_handle,
                parameters=gridk_compute_parameters.parameters,
                variableBufferSize=exchangeint_variable_buffer_descriptor.pointer,
                temporaryWorkspace=temporary_workspace.pointer,
                numDensityMatrices=1,
                densityMatrices=Dptr2,
                outExchangeMatrices=Kptr2,
                )

            if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                raise RuntimeError('cuestGridKSymmetricExchangeCompute failed: %s' % status)

            # Release this matrix's workspace before the next is allocated
            del temporary_workspace

    @staticmethod
    def symmetric_exchange_gradient(
        *,
        handle : CuestHandle,
        gridk_int_plan : CuestGridKIntPlan,
        noccs,
        Coccptrs,
        Gptrs,
        ):

        if not (len(noccs) == len(Coccptrs) == len(Gptrs)):
            raise RuntimeError('noccs, Coccptrs, and Gptrs must have the same length')

        nao = gridk_int_plan.primary.nao

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

        CuestGridKIntCompute.symmetric_exchange_gradient_density(
            handle=handle,
            gridk_int_plan=gridk_int_plan,
            Dptrs=[_.pointer for _ in densities],
            Gptrs=Gptrs,
            )

    @staticmethod
    def symmetric_exchange_gradient_density(
        *,
        handle : CuestHandle,
        gridk_int_plan : CuestGridKIntPlan,
        Dptrs,
        Gptrs,
        ):

        if not (len(Dptrs) == len(Gptrs)):
            raise RuntimeError('Dptrs and Gptrs must have the same length')

        # Density-matrix entry point. Each density must be symmetric but need
        # not be positive semidefinite; the result is the nuclear gradient of
        # the Grid-K exchange energy Tr(K(D) D) at fixed density

        gridk_compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_GRIDKSYMMETRICEXCHANGEDERIVATIVECOMPUTE_PARAMETERS,
            [],
            )

        # Allow the algorithm to use up to 8GB of DRAM as the grid-blocking budget.
        # This may need to be increased to handle very large systems.
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        exchangeint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=8000000000,
            )

        for Dptr, Gptr in zip(Dptrs, Gptrs):

            Dptr2 = ce.Pointer()
            Dptr2.value = np.intp(Dptr)

            Gptr2 = ce.Pointer()
            Gptr2.value = np.intp(Gptr)

            # => Workspace query <= #

            temporary_workspace_descriptor = CuestWorkspaceDescriptor()

            status = ce.cuestGridKSymmetricExchangeDerivativeComputeWorkspaceQuery(
                handle=handle.handle,
                plan=gridk_int_plan.gridk_int_plan_handle,
                parameters=gridk_compute_parameters.parameters,
                variableBufferSize=exchangeint_variable_buffer_descriptor.pointer,
                temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
                numDensityMatrices=1,
                densityMatrices=Dptr2,
                outGradients=Gptr2,
                )

            if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                raise RuntimeError('cuestGridKSymmetricExchangeDerivativeComputeWorkspaceQuery failed: %s' % status)

            temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

            status = ce.cuestGridKSymmetricExchangeDerivativeCompute(
                handle=handle.handle,
                plan=gridk_int_plan.gridk_int_plan_handle,
                parameters=gridk_compute_parameters.parameters,
                variableBufferSize=exchangeint_variable_buffer_descriptor.pointer,
                temporaryWorkspace=temporary_workspace.pointer,
                numDensityMatrices=1,
                densityMatrices=Dptr2,
                outGradients=Gptr2,
                )

            if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                raise RuntimeError('cuestGridKSymmetricExchangeDerivativeCompute failed: %s' % status)

            # Release this matrix's workspace before the next is allocated
            del temporary_workspace

    @staticmethod
    def nonsymmetric_exchange(
        *,
        handle : CuestHandle,
        gridk_int_plan : CuestGridKIntPlan,
        nocc,
        Cocc_left_ptr,
        Cocc_right_ptrs,
        Kptrs,
        ):

        if not (len(Cocc_right_ptrs) == len(Kptrs)):
            raise RuntimeError('Cocc_right_ptrs and Kptrs must have the same length')

        nao = gridk_int_plan.primary.nao
        nRight = len(Cocc_right_ptrs)

        # The grid-K C API takes density matrices; the (generally
        # non-symmetric) densities D_i = L^T R_i are constructed here, stacked
        # contiguously, from the coefficient matrices
        density_matrices = GPUMatrix(
            nrows=nRight,
            ncols=nao * nao,
            dtype=np.double,
            )
        for i, Cocc_right_ptr in enumerate(Cocc_right_ptrs):
            GPUMatrixUtility.c_dgemm(
                transa=True,
                transb=False,
                m=nao,
                n=nao,
                k=nocc,
                alpha=1.0,
                a=DevicePointerView(Cocc_left_ptr, nocc * nao),
                offa=0,
                lda=nao,
                b=DevicePointerView(Cocc_right_ptr, nocc * nao),
                offb=0,
                ldb=nao,
                beta=0.0,
                c=density_matrices,
                offc=i * nao * nao,
                ldc=nao,
                )

        K_out_concatenated = GPUMatrix(
            nrows=nRight,
            ncols=nao * nao,
            dtype=np.double,
            )

        Dptr2 = ce.Pointer()
        Dptr2.value = np.intp(density_matrices.pointer)

        Kptr2 = ce.Pointer()
        Kptr2.value = np.intp(K_out_concatenated.pointer)

        gridk_compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_GRIDKNONSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
            [],
            )

        # Allow the algorithm to use up to 4GB of DRAM as the grid-blocking budget.
        # This may need to be increased to handle very large systems.
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        exchangeint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=4000000000,
            )

        # => Workspace query <= #

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestGridKNonsymmetricExchangeComputeWorkspaceQuery(
            handle=handle.handle,
            plan=gridk_int_plan.gridk_int_plan_handle,
            parameters=gridk_compute_parameters.parameters,
            variableBufferSize=exchangeint_variable_buffer_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numDensityMatrices=nRight,
            densityMatrices=Dptr2,
            outExchangeMatrices=Kptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestGridKNonsymmetricExchangeComputeWorkspaceQuery failed: %s' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestGridKNonsymmetricExchangeCompute(
            handle=handle.handle,
            plan=gridk_int_plan.gridk_int_plan_handle,
            parameters=gridk_compute_parameters.parameters,
            variableBufferSize=exchangeint_variable_buffer_descriptor.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            numDensityMatrices=nRight,
            densityMatrices=Dptr2,
            outExchangeMatrices=Kptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestGridKNonsymmetricExchangeCompute failed: %s' % status)

        for i, Kptr in enumerate(Kptrs):
            GPUMatrixUtility.dcopy(
                n=nao * nao,
                x=K_out_concatenated,
                offx=i * nao * nao,
                incx=1,
                y=DevicePointerView(Kptr, nao * nao),
                offy=0,
                incy=1,
                )
