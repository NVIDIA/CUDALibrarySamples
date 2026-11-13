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
from .cuest_xc_int_plan import CuestXCIntPlan

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace
from .cuest_utility import CuestUtility

import numpy as np


class CuestXCIntCompute(object):

    @staticmethod
    def local_exchange_correlation_rks(
        *,
        handle : CuestHandle,
        xc_int_plan : CuestXCIntPlan,
        Vxcptr,
        nocc,
        Coccptr,
        ):

        Exc = ce.data_double()

        Coccptr2 = ce.Pointer()
        Coccptr2.value = np.intp(Coccptr)

        Vxcptr2 = ce.Pointer()
        Vxcptr2.value = np.intp(Vxcptr)

        xc_compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_XCPOTENTIALRKSCOMPUTE_PARAMETERS,
            [],
            )

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        xcint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        # => Workspace query <= #

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
            raise RuntimeError('cuestXCPotentialRKSComputeWorkspaceQuery failed: %s' % status)

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
            raise RuntimeError('cuestXCPotentialRKSCompute failed: %s' % status)

        return Exc.value

    @staticmethod
    def local_exchange_correlation_uks(
        *,
        handle : CuestHandle,
        xc_int_plan : CuestXCIntPlan,
        Vxcptr,
        nocc,
        Coccptr,
        Vxcbptr,
        noccb,
        Coccbptr,
        ):

        Exc = ce.data_double()

        Coccptr2 = ce.Pointer()
        Coccptr2.value = np.intp(Coccptr)

        Vxcptr2 = ce.Pointer()
        Vxcptr2.value = np.intp(Vxcptr)

        Coccbptr2 = ce.Pointer()
        Coccbptr2.value = np.intp(Coccbptr)

        Vxcbptr2 = ce.Pointer()
        Vxcbptr2.value = np.intp(Vxcbptr)

        xc_compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_XCPOTENTIALUKSCOMPUTE_PARAMETERS,
            [],
            )

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        xcint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        # => Workspace query <= #

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestXCPotentialUKSComputeWorkspaceQuery(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numOccupiedAlpha=nocc,
            numOccupiedBeta=noccb,
            coefficientMatrixAlpha=Coccptr2,
            coefficientMatrixBeta=Coccbptr2,
            outXCEnergy=Exc,
            outXCPotentialMatrixAlpha=Vxcptr2,
            outXCPotentialMatrixBeta=Vxcbptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestXCPotentialUKSComputeWorkspaceQuery failed: %s' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestXCPotentialUKSCompute(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            numOccupiedAlpha=nocc,
            numOccupiedBeta=noccb,
            coefficientMatrixAlpha=Coccptr2,
            coefficientMatrixBeta=Coccbptr2,
            outXCEnergy=Exc,
            outXCPotentialMatrixAlpha=Vxcptr2,
            outXCPotentialMatrixBeta=Vxcbptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestXCPotentialUKSCompute failed: %s' % status)

        return Exc.value

    @staticmethod
    def nonlocal_exchange_correlation_rks(
        *,
        handle : CuestHandle,
        xc_int_plan : CuestXCIntPlan,
        Vxcptr,
        nocc,
        Coccptr,
        ):

        Exc = ce.data_double()

        Coccptr2 = ce.Pointer()
        Coccptr2.value = np.intp(Coccptr)

        Vxcptr2 = ce.Pointer()
        Vxcptr2.value = np.intp(Vxcptr)

        nonlocal_xc_compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS,
            [(ce.CuestNonlocalXCPotentialRKSComputeParametersAttributes.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS_VV10_B,
              ce.data_double(xc_int_plan.vv10_b)),
             (ce.CuestNonlocalXCPotentialRKSComputeParametersAttributes.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS_VV10_C,
              ce.data_double(xc_int_plan.vv10_c)),
             (ce.CuestNonlocalXCPotentialRKSComputeParametersAttributes.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS_VV10_SCALE,
              ce.data_double(xc_int_plan.vv10_scale))],
            )

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        xcint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        # => Workspace query <= #

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
            raise RuntimeError('cuestNonlocalXCPotentialRKSComputeWorkspaceQuery failed: %s' % status)

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
            raise RuntimeError('cuestNonlocalXCPotentialRKSCompute failed: %s' % status)

        return Exc.value

    @staticmethod
    def nonlocal_exchange_correlation_uks(
        *,
        handle : CuestHandle,
        xc_int_plan : CuestXCIntPlan,
        Vxcptr,
        nocc,
        Coccptr,
        noccb,
        Coccbptr,
        ):

        Exc = ce.data_double()

        Coccptr2 = ce.Pointer()
        Coccptr2.value = np.intp(Coccptr)

        Vxcptr2 = ce.Pointer()
        Vxcptr2.value = np.intp(Vxcptr)

        Coccbptr2 = ce.Pointer()
        Coccbptr2.value = np.intp(Coccbptr)

        nonlocal_xc_compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS,
            [(ce.CuestNonlocalXCPotentialUKSComputeParametersAttributes.CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS_VV10_B,
              ce.data_double(xc_int_plan.vv10_b)),
             (ce.CuestNonlocalXCPotentialUKSComputeParametersAttributes.CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS_VV10_C,
              ce.data_double(xc_int_plan.vv10_c)),
             (ce.CuestNonlocalXCPotentialUKSComputeParametersAttributes.CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS_VV10_SCALE,
              ce.data_double(xc_int_plan.vv10_scale))],
            )

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        xcint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        # => Workspace query <= #

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestNonlocalXCPotentialUKSComputeWorkspaceQuery(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=nonlocal_xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numOccupiedAlpha=nocc,
            numOccupiedBeta=noccb,
            coefficientMatrixAlpha=Coccptr2,
            coefficientMatrixBeta=Coccbptr2,
            outXCEnergy=Exc,
            outXCPotentialMatrix=Vxcptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestNonlocalXCPotentialUKSComputeWorkspaceQuery failed: %s' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestNonlocalXCPotentialUKSCompute(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=nonlocal_xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            numOccupiedAlpha=nocc,
            numOccupiedBeta=noccb,
            coefficientMatrixAlpha=Coccptr2,
            coefficientMatrixBeta=Coccbptr2,
            outXCEnergy=Exc,
            outXCPotentialMatrix=Vxcptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestNonlocalXCPotentialUKSCompute failed: %s' % status)

        return Exc.value

    @staticmethod
    def local_exchange_correlation_rks_gradient(
        *,
        handle : CuestHandle,
        xc_int_plan : CuestXCIntPlan,
        Gptr,
        nocc,
        Coccptr,
        ):

        Coccptr2 = ce.Pointer()
        Coccptr2.value = np.intp(Coccptr)

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(Gptr)

        xc_compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS,
            [],
            )

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        xcint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        # => Workspace query <= #

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
            raise RuntimeError('cuestXCDerivativeRKSComputeWorkspaceQuery failed: %s' % status)

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
            raise RuntimeError('cuestXCDerivativeRKSCompute failed: %s' % status)

    @staticmethod
    def local_exchange_correlation_uks_gradient(
        *,
        handle : CuestHandle,
        xc_int_plan : CuestXCIntPlan,
        Gptr,
        nocc,
        Coccptr,
        noccb,
        Coccbptr,
        ):

        Coccptr2 = ce.Pointer()
        Coccptr2.value = np.intp(Coccptr)

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(Gptr)

        Coccbptr2 = ce.Pointer()
        Coccbptr2.value = np.intp(Coccbptr)

        xc_compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_XCDERIVATIVEUKSCOMPUTE_PARAMETERS,
            [],
            )

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        xcint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        # => Workspace query <= #

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestXCDerivativeUKSComputeWorkspaceQuery(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numOccupiedAlpha=nocc,
            numOccupiedBeta=noccb,
            coefficientMatrixAlpha=Coccptr2,
            coefficientMatrixBeta=Coccbptr2,
            outGradient=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestXCDerivativeUKSComputeWorkspaceQuery failed: %s' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestXCDerivativeUKSCompute(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            numOccupiedAlpha=nocc,
            numOccupiedBeta=noccb,
            coefficientMatrixAlpha=Coccptr2,
            coefficientMatrixBeta=Coccbptr2,
            outGradient=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestXCDerivativeUKSCompute failed: %s' % status)

    @staticmethod
    def nonlocal_exchange_correlation_rks_gradient(
        *,
        handle : CuestHandle,
        xc_int_plan : CuestXCIntPlan,
        Gptr,
        nocc,
        Coccptr,
        ):

        Coccptr2 = ce.Pointer()
        Coccptr2.value = np.intp(Coccptr)

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(Gptr)

        nonlocal_xc_compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS,
            [(ce.CuestNonlocalXCDerivativeRKSComputeParametersAttributes.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS_VV10_B,
              ce.data_double(xc_int_plan.vv10_b)),
             (ce.CuestNonlocalXCDerivativeRKSComputeParametersAttributes.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS_VV10_C,
              ce.data_double(xc_int_plan.vv10_c)),
             (ce.CuestNonlocalXCDerivativeRKSComputeParametersAttributes.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS_VV10_SCALE,
              ce.data_double(xc_int_plan.vv10_scale))],
            )

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        xcint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        # => Workspace query <= #

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
            raise RuntimeError('cuestNonlocalXCDerivativeRKSComputeWorkspaceQuery failed: %s' % status)

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
            raise RuntimeError('cuestNonlocalXCDerivativeRKSCompute failed: %s' % status)

    @staticmethod
    def nonlocal_exchange_correlation_uks_gradient(
        *,
        handle : CuestHandle,
        xc_int_plan : CuestXCIntPlan,
        Gptr,
        nocc,
        Coccptr,
        noccb,
        Coccbptr,
        ):

        Coccptr2 = ce.Pointer()
        Coccptr2.value = np.intp(Coccptr)

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(Gptr)

        Coccbptr2 = ce.Pointer()
        Coccbptr2.value = np.intp(Coccbptr)

        nonlocal_xc_compute_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_NONLOCALXCDERIVATIVEUKSCOMPUTE_PARAMETERS,
            [(ce.CuestNonlocalXCDerivativeUKSComputeParametersAttributes.CUEST_NONLOCALXCDERIVATIVEUKSCOMPUTE_PARAMETERS_VV10_B,
              ce.data_double(xc_int_plan.vv10_b)),
             (ce.CuestNonlocalXCDerivativeUKSComputeParametersAttributes.CUEST_NONLOCALXCDERIVATIVEUKSCOMPUTE_PARAMETERS_VV10_C,
              ce.data_double(xc_int_plan.vv10_c)),
             (ce.CuestNonlocalXCDerivativeUKSComputeParametersAttributes.CUEST_NONLOCALXCDERIVATIVEUKSCOMPUTE_PARAMETERS_VV10_SCALE,
              ce.data_double(xc_int_plan.vv10_scale))],
            )

        # Allow the algorithm to use up to 2GB of DRAM
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        xcint_variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=2000000000,
            )

        # => Workspace query <= #

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestNonlocalXCDerivativeUKSComputeWorkspaceQuery(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=nonlocal_xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numOccupiedAlpha=nocc,
            numOccupiedBeta=noccb,
            coefficientMatrixAlpha=Coccptr2,
            coefficientMatrixBeta=Coccbptr2,
            outGradient=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestNonlocalXCDerivativeUKSComputeWorkspaceQuery failed: %s' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestNonlocalXCDerivativeUKSCompute(
            handle=handle.handle,
            plan=xc_int_plan.xc_int_plan_handle,
            parameters=nonlocal_xc_compute_parameters.parameters,
            variableBufferSize=xcint_variable_buffer_descriptor.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            numOccupiedAlpha=nocc,
            numOccupiedBeta=noccb,
            coefficientMatrixAlpha=Coccptr2,
            coefficientMatrixBeta=Coccbptr2,
            outGradient=Gptr2,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestNonlocalXCDerivativeUKSCompute failed: %s' % status)
