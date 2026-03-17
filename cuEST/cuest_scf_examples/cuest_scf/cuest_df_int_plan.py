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

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace

from .cuest_handle import CuestHandle
from .cuest_ao_basis import CuestAOBasis
from .cuest_ao_pair_list import CuestAOPairList

from .cuest_parameters import CuestParameters

class CuestDFIntPlan(object):

    def __init__(
        self,
        *,
        handle : CuestHandle,
        primary : CuestAOBasis,
        auxiliary : CuestAOBasis,
        ao_pair_list : CuestAOPairList,
        exchange_scale : float = 1.0,
        df_fitting_eigenvalue_cutoff : float = 1.0e-12,
        ):

        self.initialized = False

        self.handle = handle
        self.exchange_scale = exchange_scale
        self.df_fitting_eigenvalue_cutoff = df_fitting_eigenvalue_cutoff

        # NOTE: The CuestDFIntPlan has several defaultable parameters that
        # production codes may wish to override and expose as user-accessible
        # parameters.

        df_int_plan_parameters = CuestParameters(
            parametersType=ce.CuestParametersType.CUEST_DFINTPLAN_PARAMETERS,
            )

        exchange_scale_data = ce.data_double(self.exchange_scale)
        df_fitting_eigenvalue_cutoff_data = ce.data_double(self.df_fitting_eigenvalue_cutoff)

        self.df_int_plan_handle = ce.cuestDFIntPlanHandle()
        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_DFINTPLAN_PARAMETERS,
            parameters=df_int_plan_parameters.parameters,
            attribute=ce.CuestDFIntPlanParametersAttributes.CUEST_DFINTPLAN_PARAMETERS_EXCHANGE_FRACTION,
            attributeValue=exchange_scale_data,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestParametersConfigure failed')

        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_DFINTPLAN_PARAMETERS,
            parameters=df_int_plan_parameters.parameters,
            attribute=ce.CuestDFIntPlanParametersAttributes.CUEST_DFINTPLAN_PARAMETERS_FITTING_CUTOFF,
            attributeValue=df_fitting_eigenvalue_cutoff_data,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestParametersConfigure failed')

        # => Workspace Query <= #

        persistent_workspace_descriptor = CuestWorkspaceDescriptor()
        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestDFIntPlanCreateWorkspaceQuery(
            handle=handle.handle,
            primaryBasis=primary.ao_basis_handle,
            auxiliaryBasis=auxiliary.ao_basis_handle,
            pairList=ao_pair_list.ao_pair_list_handle,
            parameters=df_int_plan_parameters.parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPlan=self.df_int_plan_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFIntPlanCreateWorkspaceQuery failed')

        # Most likely place to run out of device memory, so we will add a custom RuntimeError
        try:
            persistent_workspace = CuestWorkspace(workspaceDescriptor=persistent_workspace_descriptor)
            temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)
        except RuntimeError as e:
            raise RuntimeError("CuestDFIntPlan workspace allocation failed - out of memory for CoreDF") from e

        # => Creation <= #

        status = ce.cuestDFIntPlanCreate(
            handle=handle.handle,
            primaryBasis=primary.ao_basis_handle,
            auxiliaryBasis=auxiliary.ao_basis_handle,
            pairList=ao_pair_list.ao_pair_list_handle,
            parameters=df_int_plan_parameters.parameters,
            persistentWorkspace=persistent_workspace.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            outPlan=self.df_int_plan_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFIntPlanCreate failed')

        # Bind the lifetime of persistent_workspace to this object
        self.persistent_workspace = persistent_workspace

        self.initialized = True

    def __del__(self):

        if not self.initialized: return

        status = ce.cuestDFIntPlanDestroy(
            handle=self.df_int_plan_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFIntPlanDestroy failed')
