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
from .cuest_df_int_plan import CuestDFIntPlan

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace
from .cuest_utility import CuestUtility, DevicePointerView

from .gpu_matrix import GPUMatrix
from .gpu_matrix_utility import GPUMatrixUtility

import numpy as np


class CuestDFIntCompute(object):

    @staticmethod
    def coulomb(
        *,
        handle : CuestHandle,
        df_int_plan : CuestDFIntPlan,
        Dptr,
        Jptr,
        ):

        Dptr2 = ce.Pointer()
        Dptr2.value = np.intp(Dptr)

        Jptr2 = ce.Pointer()
        Jptr2.value = np.intp(Jptr)

        dfj_compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_DFCOULOMBCOMPUTE_PARAMETERS,
            [],
            )

        # => Workspace query <= #

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestDFCoulombComputeWorkspaceQuery(
            handle=handle.handle,
            plan=df_int_plan.df_int_plan_handle,
            parameters=dfj_compute_parameters.parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityMatrix=Dptr2,
            outCoulombMatrix=Jptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFCoulombComputeWorkspaceQuery failed: %s' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestDFCoulombCompute(
            handle=handle.handle,
            plan=df_int_plan.df_int_plan_handle,
            parameters=dfj_compute_parameters.parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            densityMatrix=Dptr2,
            outCoulombMatrix=Jptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFCoulombCompute failed: %s' % status)

    @staticmethod
    def symmetric_exchange(
        *,
        handle : CuestHandle,
        df_int_plan : CuestDFIntPlan,
        noccs,
        Coccptrs,
        Kptrs,
        dfk_int8_slice_count,
        dfk_int8_modulus_count,
        ):

        if not (len(noccs) == len(Coccptrs) == len(Kptrs)):
            raise RuntimeError('noccs, Coccptrs, and Kptrs must have the same length')

        dfk_compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
            [(ce.CuestDFSymmetricExchangeComputeParametersAttributes.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_SLICE_COUNT,
              ce.data_uint64_t(dfk_int8_slice_count)),
             (ce.CuestDFSymmetricExchangeComputeParametersAttributes.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_MODULUS_COUNT,
              ce.data_uint64_t(dfk_int8_modulus_count))],
            )

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        exchangeint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        for nocc, Coccptr, Kptr in zip(noccs, Coccptrs, Kptrs):

            Coccptr2 = ce.Pointer()
            Coccptr2.value = np.intp(Coccptr)

            Kptr2 = ce.Pointer()
            Kptr2.value = np.intp(Kptr)

            # => Workspace query <= #

            temporary_workspace_descriptor = CuestWorkspaceDescriptor()

            status = ce.cuestDFSymmetricExchangeComputeWorkspaceQuery(
                handle=handle.handle,
                plan=df_int_plan.df_int_plan_handle,
                parameters=dfk_compute_parameters.parameters,
                variableBufferSize=exchangeint_variable_buffer_descriptor.pointer,
                temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
                numOccupied=nocc,
                coefficientMatrix=Coccptr2,
                outExchangeMatrix=Kptr2,
                )

            if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                raise RuntimeError('cuestDFSymmetricExchangeComputeWorkspaceQuery failed: %s' % status)

            temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

            status = ce.cuestDFSymmetricExchangeCompute(
                handle=handle.handle,
                plan=df_int_plan.df_int_plan_handle,
                parameters=dfk_compute_parameters.parameters,
                variableBufferSize=exchangeint_variable_buffer_descriptor.pointer,
                temporaryWorkspace=temporary_workspace.pointer,
                numOccupied=nocc,
                coefficientMatrix=Coccptr2,
                outExchangeMatrix=Kptr2,
                )

            if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                raise RuntimeError('cuestDFSymmetricExchangeCompute failed: %s' % status)

            # Release this matrix's workspace before the next is allocated
            del temporary_workspace

    @staticmethod
    def nonsymmetric_exchange(
        *,
        handle : CuestHandle,
        df_int_plan : CuestDFIntPlan,
        nocc,
        Cocc_left_ptr,
        Cocc_right_ptrs,
        Kptrs,
        ):

        if not (len(Cocc_right_ptrs) == len(Kptrs)):
            raise RuntimeError('Cocc_right_ptrs and Kptrs must have the same length')

        nao = df_int_plan.primary.nao
        nRight = len(Cocc_right_ptrs)

        # The C API takes the right coefficient matrices and the output
        # exchange matrices stored contiguously
        Cocc_right_concatenated = GPUMatrix(
            nrows=nRight,
            ncols=nocc * nao,
            dtype=np.double,
            )
        for i, Cocc_right_ptr in enumerate(Cocc_right_ptrs):
            GPUMatrixUtility.dcopy(
                n=nocc * nao,
                x=DevicePointerView(Cocc_right_ptr, nocc * nao),
                offx=0,
                incx=1,
                y=Cocc_right_concatenated,
                offy=i * nocc * nao,
                incy=1,
                )

        K_out_concatenated = GPUMatrix(
            nrows=nRight,
            ncols=nao * nao,
            dtype=np.double,
            )

        Cocc_left_ptr2 = ce.Pointer()
        Cocc_left_ptr2.value = np.intp(Cocc_left_ptr)

        Cocc_right_ptr2 = ce.Pointer()
        Cocc_right_ptr2.value = np.intp(Cocc_right_concatenated.pointer)

        Kptr2 = ce.Pointer()
        Kptr2.value = np.intp(K_out_concatenated.pointer)

        dfk_compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_DFNONSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
            [],
            )

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        exchangeint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        # => Workspace query <= #

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestDFNonsymmetricExchangeComputeWorkspaceQuery(
            handle=handle.handle,
            plan=df_int_plan.df_int_plan_handle,
            parameters=dfk_compute_parameters.parameters,
            variableBufferSize=exchangeint_variable_buffer_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numCoefficientMatrices=nRight,
            numOccupied=nocc,
            leftCoefficientMatrix=Cocc_left_ptr2,
            rightCoefficientMatrices=Cocc_right_ptr2,
            outExchangeMatrices=Kptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFNonsymmetricExchangeComputeWorkspaceQuery failed: %s' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestDFNonsymmetricExchangeCompute(
            handle=handle.handle,
            plan=df_int_plan.df_int_plan_handle,
            parameters=dfk_compute_parameters.parameters,
            variableBufferSize=exchangeint_variable_buffer_descriptor.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            numCoefficientMatrices=nRight,
            numOccupied=nocc,
            leftCoefficientMatrix=Cocc_left_ptr2,
            rightCoefficientMatrices=Cocc_right_ptr2,
            outExchangeMatrices=Kptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFNonsymmetricExchangeCompute failed: %s' % status)

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

    @staticmethod
    def coulomb_and_exchange_gradient(
        *,
        handle : CuestHandle,
        df_int_plan : CuestDFIntPlan,
        scaleJ,
        DJptr,
        scaleK,
        noccsK,
        CoccsKptrs,
        Gptr,
        ):

        if not (len(noccsK) == len(CoccsKptrs)):
            raise RuntimeError('noccsK and CoccsKptrs must have the same length')

        nao = df_int_plan.primary.nao

        # The C API takes the coefficient matrices stored contiguously
        CoccsK_concatenated = GPUMatrix(
            nrows=sum(noccsK),
            ncols=nao,
            dtype=np.double,
            )
        offset = 0
        for nocc, CoccKptr in zip(noccsK, CoccsKptrs):
            GPUMatrixUtility.dcopy(
                n=nocc * nao,
                x=DevicePointerView(CoccKptr, nocc * nao),
                offx=0,
                incx=1,
                y=CoccsK_concatenated,
                offy=offset,
                incy=1,
                )
            offset += nocc * nao

        DJptr2 = ce.Pointer()
        DJptr2.value = np.intp(DJptr)

        CoccsKptr2 = ce.Pointer()
        CoccsKptr2.value = np.intp(CoccsK_concatenated.pointer)

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(Gptr)

        df_grad_compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS,
            [(ce.CuestDFSymmetricDerivativeComputeParametersAttributes.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS_MEMORY_POLICY,
              ce.data_cuestDFSymmetricDerivativeComputeMemoryPolicy_t(ce.CuestDFSymmetricDerivativeComputeMemoryPolicy.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_MEMORY_POLICY_OVERWRITE))],
            )

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        # => Workspace query <= #

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestDFSymmetricDerivativeComputeWorkspaceQuery(
            handle=handle.handle,
            plan=df_int_plan.df_int_plan_handle,
            parameters=df_grad_compute_parameters.parameters,
            variableBufferSize=variable_buffer_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityScale=scaleJ,
            densityMatrix=DJptr2,
            coefficientScale=scaleK,
            numCoefficientMatrices=len(noccsK),
            numOccupied=noccsK,
            coefficientMatrices=CoccsKptr2,
            outGradient=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFSymmetricDerivativeComputeWorkspaceQuery failed: %s' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestDFSymmetricDerivativeCompute(
            handle=handle.handle,
            plan=df_int_plan.df_int_plan_handle,
            parameters=df_grad_compute_parameters.parameters,
            variableBufferSize=variable_buffer_descriptor.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            densityScale=scaleJ,
            densityMatrix=DJptr2,
            coefficientScale=scaleK,
            numCoefficientMatrices=len(noccsK),
            numOccupied=noccsK,
            coefficientMatrices=CoccsKptr2,
            outGradient=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFSymmetricDerivativeCompute failed: %s' % status)
