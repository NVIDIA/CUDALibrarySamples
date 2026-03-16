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

class CuestOEIntPlan(object):

    def __init__(
        self,
        *,
        handle : CuestHandle,
        basis : CuestAOBasis,
        ao_pair_list : CuestAOPairList,
        ):

        self.initialized = False

        self.handle = handle

        oe_int_plan_parameters = CuestParameters(
            parametersType=ce.CuestParametersType.CUEST_OEINTPLAN_PARAMETERS,
            )

        self.oe_int_plan_handle = ce.cuestOEIntPlanHandle()

        # => Workspace Query <= #

        persistent_workspace_descriptor = CuestWorkspaceDescriptor()
        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestOEIntPlanCreateWorkspaceQuery(
            handle=handle.handle,
            basis=basis.ao_basis_handle,
            pairList=ao_pair_list.ao_pair_list_handle,
            parameters=oe_int_plan_parameters.parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPlan=self.oe_int_plan_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestOEIntPlanCreateWorkspaceQuery failed')

        persistent_workspace = CuestWorkspace(workspaceDescriptor=persistent_workspace_descriptor)
        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        # => Creation <= #

        status = ce.cuestOEIntPlanCreate(
            handle=handle.handle,
            basis=basis.ao_basis_handle,
            pairList=ao_pair_list.ao_pair_list_handle,
            parameters=oe_int_plan_parameters.parameters,
            persistentWorkspace=persistent_workspace.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            outPlan=self.oe_int_plan_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestOEIntPlanCreate failed')

        # Bind the lifetime of persistent_workspace to this object
        self.persistent_workspace = persistent_workspace

        self.initialized = True

    def __del__(self):

        if not self.initialized: return

        status = ce.cuestOEIntPlanDestroy(
            handle=self.oe_int_plan_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestOEIntPlanDestroy failed')
