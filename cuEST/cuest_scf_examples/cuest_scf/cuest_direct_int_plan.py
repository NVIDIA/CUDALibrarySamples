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

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace

from .cuest_handle import CuestHandle
from .cuest_ao_basis import CuestAOBasis
from .cuest_ao_pair_list import CuestAOPairList

from .cuest_utility import CuestUtility

class CuestDirectIntPlan(object):

    def __init__(
        self,
        *,
        handle : CuestHandle,
        primary : CuestAOBasis,
        ao_pair_list : CuestAOPairList,
        exchange_scale : float,
        lrc_exchange_scale : float,
        lrc_omega : float,
        schwarz_threshold : float,
        ):

        self.initialized = False

        self.handle = handle
        self.exchange_scale = exchange_scale
        self.lrc_exchange_scale = lrc_exchange_scale
        self.lrc_omega = lrc_omega

        # The Schwarz threshold is a compute-time parameter of the direct
        # engines rather than a plan parameter; it is kept here so that the
        # plan reports the screening its computes will run with
        self.schwarz_threshold = schwarz_threshold

        direct_int_plan_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_DIRECTINTPLAN_PARAMETERS,
            [(ce.CuestDirectIntPlanParametersAttributes.CUEST_DIRECTINTPLAN_PARAMETERS_EXCHANGE_FRACTION,
              ce.data_double(self.exchange_scale)),
             (ce.CuestDirectIntPlanParametersAttributes.CUEST_DIRECTINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION,
              ce.data_double(self.lrc_exchange_scale)),
             (ce.CuestDirectIntPlanParametersAttributes.CUEST_DIRECTINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA,
              ce.data_double(self.lrc_omega))],
            )

        self.direct_int_plan_handle = ce.cuestDirectIntPlanHandle()
        # => Workspace Query <= #

        persistent_workspace_descriptor = CuestWorkspaceDescriptor()
        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestDirectIntPlanCreateWorkspaceQuery(
            handle=handle.handle,
            primaryBasis=primary.ao_basis_handle,
            pairList=ao_pair_list.ao_pair_list_handle,
            parameters=direct_int_plan_parameters.parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPlan=self.direct_int_plan_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDirectIntPlanCreateWorkspaceQuery failed: %s' % status)

        try:
            persistent_workspace = CuestWorkspace(workspaceDescriptor=persistent_workspace_descriptor)
            temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)
        except RuntimeError as e:
            raise RuntimeError("CuestDirectIntPlan workspace allocation failed - out of memory for the direct J/K plan") from e

        # => Creation <= #

        status = ce.cuestDirectIntPlanCreate(
            handle=handle.handle,
            primaryBasis=primary.ao_basis_handle,
            pairList=ao_pair_list.ao_pair_list_handle,
            parameters=direct_int_plan_parameters.parameters,
            persistentWorkspace=persistent_workspace.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            outPlan=self.direct_int_plan_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDirectIntPlanCreate failed: %s' % status)

        # Bind the lifetime of persistent_workspace to this object
        self.persistent_workspace = persistent_workspace

        # Keep wrapper dependencies alive while the native cuEST object may
        # refer to them; this mirrors the C API lifetime requirements.
        self.primary = primary
        self.ao_pair_list = ao_pair_list

        self.initialized = True

    # => String Details <= #

    def __str__(self):
        return self.string

    @property
    def string(self):
        s = ''
        s += 'DirectIntPlan:\n\n'
        s += '%-19s = %11.3E\n' % ('schwarz threshold', self.schwarz_threshold)
        return s

    def __del__(self):

        if not self.initialized: return

        status = ce.cuestDirectIntPlanDestroy(
            handle=self.direct_int_plan_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDirectIntPlanDestroy failed: %s' % status)
