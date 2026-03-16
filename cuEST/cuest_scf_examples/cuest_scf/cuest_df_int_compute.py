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

import cuest.bindings as ce    

from .cuest_handle import CuestHandle
from .cuest_df_int_plan import CuestDFIntPlan

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace
from .cuest_parameters import CuestParameters

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

        # => Workspace query <= #

        dfj_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_DFCOULOMBCOMPUTE_PARAMETERS)

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
            raise RuntimeError('cuestDFCoulombComputeWorkspaceQuery failed')

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
            raise RuntimeError('cuestDFCoulombCompute failed')
        del dfj_compute_parameters


    @staticmethod
    def symmetric_exchange(
        *,
        handle : CuestHandle,
        df_int_plan : CuestDFIntPlan,
        nocc,
        Coccptr,
        Kptr,
        dfk_int8_slice_count,
        dfk_int8_modulus_count,
        ):

        Coccptr2 = ce.Pointer()
        Coccptr2.value = np.intp(Coccptr)

        Kptr2 = ce.Pointer()
        Kptr2.value = np.intp(Kptr)

        # => Workspace query <= #

        dfk_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS)

        slice_count = ce.data_uint64_t(dfk_int8_slice_count)
        modulus_count = ce.data_uint64_t(dfk_int8_modulus_count)

        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
            parameters=dfk_compute_parameters.parameters,
            attribute=ce.CuestDFSymmetricExchangeComputeParametersAttributes.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_SLICE_COUNT,
            attributeValue=slice_count,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestParametersConfigure failed')

        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
            parameters=dfk_compute_parameters.parameters,
            attribute=ce.CuestDFSymmetricExchangeComputeParametersAttributes.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_MODULUS_COUNT,
            attributeValue=modulus_count,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestParametersConfigure failed')

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        exchangeint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

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
            raise RuntimeError('cuestDFExchangeComputeWorkspaceQuery failed')

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
            raise RuntimeError('cuestDFExchangeCompute failed')
        del dfk_compute_parameters


    @staticmethod
    def coulomb_and_exchange_gradient(
        *,
        handle : CuestHandle,
        df_int_plan : CuestDFIntPlan,
        scaleJ,
        DJptr,
        scaleK,
        noccsK,
        CoccsKptr,
        Gptr,
        ):

        DJptr2 = ce.Pointer()
        DJptr2.value = np.intp(DJptr)

        CoccsKptr2 = ce.Pointer()
        CoccsKptr2.value = np.intp(CoccsKptr)

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(Gptr)

        # => Workspace query <= #

        df_grad_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS)

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        memory_policy_data = ce.data_cuestDFSymmetricDerivativeComputeMemoryPolicy_t(ce.CuestDFSymmetricDerivativeComputeMemoryPolicy.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_MEMORY_POLICY_FULL)
        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS,
            parameters=df_grad_compute_parameters.parameters,
            attribute=ce.CuestDFSymmetricDerivativeComputeParametersAttributes.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS_MEMORY_POLICY,
            attributeValue=memory_policy_data,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestParametersConfigure failed')

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
            raise RuntimeError('cuestDFSymmetricDerivativeComputeWorkspaceQuery failed')

        try:
            temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        except RuntimeError:
            memory_policy_data = ce.data_cuestDFSymmetricDerivativeComputeMemoryPolicy_t(ce.CuestDFSymmetricDerivativeComputeMemoryPolicy.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_MEMORY_POLICY_BLOCKED)
            status = ce.cuestParametersConfigure(
                parametersType=ce.CuestParametersType.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS,
                parameters=df_grad_compute_parameters.parameters,
                attribute=ce.CuestDFSymmetricDerivativeComputeParametersAttributes.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS_MEMORY_POLICY,
                attributeValue=memory_policy_data,
                )
            if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                raise RuntimeError('cuestParametersConfigure failed')

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
                raise RuntimeError('cuestDFSymmetricDerivativeComputeWorkspaceQuery failed')

            try:
                temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

            except RuntimeError:
                raise RuntimeError('Cannot allocate memory for cuestDFSymmetricDerivativeCompute')

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
            raise RuntimeError('cuestDFSymmetricDerivativeCompute failed')
        del df_grad_compute_parameters


