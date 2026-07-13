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
from .cuest_ecp_int_plan import CuestECPIntPlan

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace
from .cuest_parameters import CuestParameters

import numpy as np

class CuestECPIntCompute(object):

    @staticmethod
    def compute_ecp_potential(
        *,
        handle : CuestHandle,
        ecp_int_plan : CuestECPIntPlan,
        Vptr,
        ):

        Vptr2 = ce.Pointer()
        Vptr2.value = np.intp(Vptr)

        # => Workspace query <= #

        # ecp parameters
        ecp_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_ECPCOMPUTE_PARAMETERS)

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()
        # Allow the algorithm to use up to 2GB of DRAM
        ecpint_maximum_workspace_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000
            )

        status = ce.cuestECPComputeWorkspaceQuery(
            handle=handle.handle,
            plan=ecp_int_plan.ecp_int_plan_handle,
            parameters=ecp_compute_parameters.parameters,
            variableBufferSize=ecpint_maximum_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outECPMatrix=Vptr2,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestECPComputeWorkspaceQuery failed: %d' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestECPCompute(
            handle=handle.handle,
            plan=ecp_int_plan.ecp_int_plan_handle,
            parameters=ecp_compute_parameters.parameters,
            variableBufferSize=ecpint_maximum_workspace_descriptor.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            outECPMatrix=Vptr2,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestECPCompute failed: %d' % status)
        del ecp_compute_parameters

    @staticmethod
    def compute_ecp_gradient(
        *,
        handle : CuestHandle,
        ecp_int_plan : CuestECPIntPlan,
        Dptr,
        Gptr,
        ):

        Dptr2 = ce.Pointer()
        Dptr2.value = np.intp(Dptr)

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(Gptr)

        # ecp parameters
        ecp_derivative_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_ECPDERIVATIVECOMPUTE_PARAMETERS)

        ecpint_maximum_workspace_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000
            )

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestECPDerivativeComputeWorkspaceQuery(
            handle=handle.handle,
            plan=ecp_int_plan.ecp_int_plan_handle,
            parameters=ecp_derivative_compute_parameters.parameters,
            variableBufferSize=ecpint_maximum_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityMatrix=Dptr2,
            outGradient=Gptr2,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestECPDerivativeComputeWorkspaceQuery failed: %d' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestECPDerivativeCompute(
            handle=handle.handle,
            plan=ecp_int_plan.ecp_int_plan_handle,
            parameters=ecp_derivative_compute_parameters.parameters,
            variableBufferSize=ecpint_maximum_workspace_descriptor.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            densityMatrix=Dptr2,
            outGradient=Gptr2,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestECPDerivativeCompute failed: %d' % status)
        del ecp_derivative_compute_parameters
