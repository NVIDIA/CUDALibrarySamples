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
from .cuest_oe_int_plan import CuestOEIntPlan

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace
from .cuest_parameters import CuestParameters

import numpy as np

class CuestOEIntCompute(object):

    @staticmethod
    def overlap(
        *,
        handle : CuestHandle,
        oe_int_plan : CuestOEIntPlan,
        Sptr,
        ):

        Sptr2 = ce.Pointer()
        Sptr2.value = np.intp(Sptr)

        # => Workspace query <= #

        overlap_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_OVERLAPCOMPUTE_PARAMETERS)

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestOverlapComputeWorkspaceQuery(
            handle=handle.handle,
            plan=oe_int_plan.oe_int_plan_handle,
            parameters=overlap_compute_parameters.parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outSMatrix=Sptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestOverlapComputeWorkspaceQuery failed')

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestOverlapCompute(
            handle=handle.handle,
            plan=oe_int_plan.oe_int_plan_handle,
            parameters=overlap_compute_parameters.parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            outSMatrix=Sptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestOverlapCompute failed')
        del overlap_compute_parameters


    @staticmethod
    def kinetic(
        *,
        handle : CuestHandle,
        oe_int_plan : CuestOEIntPlan,
        Tptr,
        ):

        Tptr2 = ce.Pointer()
        Tptr2.value = np.intp(Tptr)

        # => Workspace query <= #

        kinetic_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_KINETICCOMPUTE_PARAMETERS)

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestKineticComputeWorkspaceQuery(
            handle=handle.handle,
            plan=oe_int_plan.oe_int_plan_handle,
            parameters=kinetic_compute_parameters.parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outTMatrix=Tptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestKineticComputeWorkspaceQuery failed')

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestKineticCompute(
            handle=handle.handle,
            plan=oe_int_plan.oe_int_plan_handle,
            parameters=kinetic_compute_parameters.parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            outTMatrix=Tptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestKineticCompute failed')
        del kinetic_compute_parameters


    @staticmethod
    def potential(
        *,
        handle : CuestHandle,
        oe_int_plan : CuestOEIntPlan,
        ncharge,
        xyz,
        q, 
        Vptr,
        ):

        Vptr2 = ce.Pointer()
        Vptr2.value = np.intp(Vptr)

        xyz2 = ce.Pointer()
        xyz2.value = np.intp(xyz)

        q2 = ce.Pointer()
        q2.value = np.intp(q)

        # => Workspace query <= #

        potential_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_POTENTIALCOMPUTE_PARAMETERS)

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestPotentialComputeWorkspaceQuery(
            handle=handle.handle,
            plan=oe_int_plan.oe_int_plan_handle,
            parameters=potential_compute_parameters.parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numCharges=ncharge,
            xyz=xyz2,
            q=q2,
            outVMatrix=Vptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestPotentialComputeWorkspaceQuery failed')

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestPotentialCompute(
            handle=handle.handle,
            plan=oe_int_plan.oe_int_plan_handle,
            parameters=potential_compute_parameters.parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            numCharges=ncharge,
            xyz=xyz2,
            q=q2,
            outVMatrix=Vptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestPotentialCompute failed')
        del potential_compute_parameters


    @staticmethod
    def overlap_gradient(
        *,
        handle : CuestHandle,
        oe_int_plan : CuestOEIntPlan,
        Gptr,
        Wptr,
        ):

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(Gptr)

        Wptr2 = ce.Pointer()
        Wptr2.value = np.intp(Wptr)

        # => Workspace query <= #

        overlap_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_OVERLAPDERIVATIVECOMPUTE_PARAMETERS)

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestOverlapDerivativeComputeWorkspaceQuery(
            handle=handle.handle,
            plan=oe_int_plan.oe_int_plan_handle,
            parameters=overlap_compute_parameters.parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityMatrix=Wptr2,
            outGradient=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestOverlapDerivativeComputeWorkspaceQuery failed')

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestOverlapDerivativeCompute(
            handle=handle.handle,
            plan=oe_int_plan.oe_int_plan_handle,
            parameters=overlap_compute_parameters.parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            densityMatrix=Wptr2,
            outGradient=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestOverlapDerivativeCompute failed')
        del overlap_compute_parameters


    @staticmethod
    def kinetic_gradient(
        *,
        handle : CuestHandle,
        oe_int_plan : CuestOEIntPlan,
        Gptr,
        Dptr,
        ):

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(Gptr)

        Dptr2 = ce.Pointer()
        Dptr2.value = np.intp(Dptr)

        # => Workspace query <= #

        kinetic_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_KINETICDERIVATIVECOMPUTE_PARAMETERS)

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestKineticDerivativeComputeWorkspaceQuery(
            handle=handle.handle,
            plan=oe_int_plan.oe_int_plan_handle,
            parameters=kinetic_compute_parameters.parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityMatrix=Dptr2,
            outGradient=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestKineticDerivativeComputeWorkspaceQuery failed')

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestKineticDerivativeCompute(
            handle=handle.handle,
            plan=oe_int_plan.oe_int_plan_handle,
            parameters=kinetic_compute_parameters.parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            densityMatrix=Dptr2,
            outGradient=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestKineticDerivativeCompute failed')
        del kinetic_compute_parameters


    @staticmethod
    def potential_gradient(
        *,
        handle : CuestHandle,
        oe_int_plan : CuestOEIntPlan,
        ncharge,
        xyz,
        q, 
        Dptr,
        GptrBasis,
        GptrCharge,
        ):

        Dptr2 = ce.Pointer()
        Dptr2.value = np.intp(Dptr)

        xyz2 = ce.Pointer()
        xyz2.value = np.intp(xyz)

        q2 = ce.Pointer()
        q2.value = np.intp(q)

        GptrBasis2 = ce.Pointer()
        GptrBasis2.value = np.intp(GptrBasis)

        GptrCharge2 = ce.Pointer()
        GptrCharge2.value = np.intp(GptrCharge)

        # => Workspace query <= #

        potential_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_POTENTIALDERIVATIVECOMPUTE_PARAMETERS)

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestPotentialDerivativeComputeWorkspaceQuery(
            handle=handle.handle,
            plan=oe_int_plan.oe_int_plan_handle,
            parameters=potential_compute_parameters.parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numCharges=ncharge,
            xyz=xyz2,
            q=q2,
            densityMatrix=Dptr2,
            outBasisGradient=GptrBasis2,
            outChargeGradient=GptrCharge2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestPotentialDerivativeComputeWorkspaceQuery failed')

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestPotentialDerivativeCompute(
            handle=handle.handle,
            plan=oe_int_plan.oe_int_plan_handle,
            parameters=potential_compute_parameters.parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            numCharges=ncharge,
            xyz=xyz2,
            q=q2,
            densityMatrix=Dptr2,
            outBasisGradient=GptrBasis2,
            outChargeGradient=GptrCharge2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestPotentialDerivativeCompute failed')
        del potential_compute_parameters
            

