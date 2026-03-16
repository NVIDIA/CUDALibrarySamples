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
from .cuest_xc_int_plan import CuestXCIntPlan

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace
from .cuest_parameters import CuestParameters

import numpy as np

class CuestXCIntCompute(object):

    @staticmethod
    def local_exchange_correlation(
        *,
        handle : CuestHandle,
        xc_int_plan : CuestXCIntPlan,
        nocc,
        Coccptr,
        Vxcptr,
        ):

        Exc = ce.data_double()

        Coccptr2 = ce.Pointer()
        Coccptr2.value = np.intp(Coccptr)

        Vxcptr2 = ce.Pointer()
        Vxcptr2.value = np.intp(Vxcptr)

        # => Workspace query <= #
        xc_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_XCPOTENTIALRKSCOMPUTE_PARAMETERS)

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        xcint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestXCPotentialRKSComputeWorkspaceQuery(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numOccupied=nocc,
            coefficientMatrix=Coccptr2,
            outXCEnergy=Exc,
            outXCPotentialMatrix=Vxcptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestXCPotentialRKSComputeWorkspaceQuery failed: %d' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestXCPotentialRKSCompute(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            numOccupied=nocc,
            coefficientMatrix=Coccptr2,
            outXCEnergy=Exc,
            outXCPotentialMatrix=Vxcptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestXCPotentialRKSCompute failed: %d' % status)
        del xc_compute_parameters

        return Exc.value 

    @staticmethod
    def nonlocal_exchange_correlation(
        *,
        handle : CuestHandle,
        xc_int_plan : CuestXCIntPlan,
        nocc,
        Coccptr,
        Vxcptr,
        ):

        Exc = ce.data_double()

        Coccptr2 = ce.Pointer()
        Coccptr2.value = np.intp(Coccptr)

        Vxcptr2 = ce.Pointer()
        Vxcptr2.value = np.intp(Vxcptr)

        # => VV10 parameters <= #
        nonlocal_xc_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS)

        vv10_b = ce.data_double(xc_int_plan.vv10_b)
        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS,
            parameters=nonlocal_xc_compute_parameters.parameters,
            attribute=ce.CuestNonlocalXCPotentialRKSComputeParametersAttributes.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS_VV10_B,
            attributeValue=vv10_b,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestParametersConfigure failed: %d' % status)

        vv10_C = ce.data_double(xc_int_plan.vv10_c)
        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS,
            parameters=nonlocal_xc_compute_parameters.parameters,
            attribute=ce.CuestNonlocalXCPotentialRKSComputeParametersAttributes.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS_VV10_C,
            attributeValue=vv10_C,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestParametersConfigure failed: %d' % status)

        vv10_scale = ce.data_double(xc_int_plan.vv10_scale)
        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS,
            parameters=nonlocal_xc_compute_parameters.parameters,
            attribute=ce.CuestNonlocalXCPotentialRKSComputeParametersAttributes.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS_VV10_SCALE,
            attributeValue=vv10_scale,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestParametersConfigure failed: %d' % status)

        # => Workspace query <= #

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        xcint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestNonlocalXCPotentialRKSComputeWorkspaceQuery(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=nonlocal_xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numOccupied=nocc,
            coefficientMatrix=Coccptr2,
            outXCEnergy=Exc,
            outXCPotentialMatrix=Vxcptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestNonlocalXCPotentialRKSComputeWorkspaceQuery failed: %d' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestNonlocalXCPotentialRKSCompute(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=nonlocal_xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            numOccupied=nocc,
            coefficientMatrix=Coccptr2,
            outXCEnergy=Exc,
            outXCPotentialMatrix=Vxcptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestNonlocalXCPotentialRKSCompute failed: %d' % status)
        del nonlocal_xc_compute_parameters

        return Exc.value 

    @staticmethod
    def local_exchange_correlation_gradient(
        *,
        handle : CuestHandle,
        xc_int_plan : CuestXCIntPlan,
        nocc,
        Coccptr,
        Gptr,
        ):

        Coccptr2 = ce.Pointer()
        Coccptr2.value = np.intp(Coccptr)

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(Gptr)

        # => Workspace query <= #
        xc_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS)

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        xcint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestXCDerivativeRKSComputeWorkspaceQuery(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numOccupied=nocc,
            coefficientMatrix=Coccptr2,
            outGradient=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestXCDerivativeRKSComputeWorkspaceQuery failed: %d' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestXCDerivativeRKSCompute(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            numOccupied=nocc,
            coefficientMatrix=Coccptr2,
            outGradient=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestXCDerivativeRKSCompute failed: %d' % status)
        del xc_compute_parameters


    @staticmethod
    def nonlocal_exchange_correlation_gradient(
        *,
        handle : CuestHandle,
        xc_int_plan : CuestXCIntPlan,
        nocc,
        Coccptr,
        Gptr,
        ):

        Coccptr2 = ce.Pointer()
        Coccptr2.value = np.intp(Coccptr)

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(Gptr)

        # => VV10 parameters <= #
        nonlocal_xc_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS)

        vv10_b = ce.data_double(xc_int_plan.vv10_b)
        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS,
            parameters=nonlocal_xc_compute_parameters.parameters,
            attribute=ce.CuestNonlocalXCDerivativeRKSComputeParametersAttributes.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS_VV10_B,
            attributeValue=vv10_b,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestParametersConfigure failed: %d' % status)

        vv10_C = ce.data_double(xc_int_plan.vv10_c)
        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS,
            parameters=nonlocal_xc_compute_parameters.parameters,
            attribute=ce.CuestNonlocalXCDerivativeRKSComputeParametersAttributes.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS_VV10_C,
            attributeValue=vv10_C,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestParametersConfigure failed: %d' % status)

        vv10_scale = ce.data_double(xc_int_plan.vv10_scale)
        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS,
            parameters=nonlocal_xc_compute_parameters.parameters,
            attribute=ce.CuestNonlocalXCDerivativeRKSComputeParametersAttributes.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS_VV10_SCALE,
            attributeValue=vv10_scale,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestParametersConfigure failed: %d' % status)

        # => Workspace query <= #

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        xcint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestNonlocalXCDerivativeRKSComputeWorkspaceQuery(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=nonlocal_xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numOccupied=nocc,
            coefficientMatrix=Coccptr2,
            outGradient=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestNonlocalXCDerivativeRKSComputeWorkspaceQuery failed: %d' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestNonlocalXCDerivativeRKSCompute(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=nonlocal_xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            numOccupied=nocc,
            coefficientMatrix=Coccptr2,
            outGradient=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestNonlocalXCDerivativeRKSCompute failed: %d' % status)
        del nonlocal_xc_compute_parameters


