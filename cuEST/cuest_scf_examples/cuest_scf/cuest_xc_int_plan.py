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

from .memoized_property import memoized_property

import cuest.bindings as ce    

from .xc_functionals import XCFunctionalInfo

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace

from .cuest_handle import CuestHandle
from .cuest_ao_basis import CuestAOBasis
from .cuest_molecular_grid import CuestMolecularGrid

from .cuest_parameters import CuestParameters

class CuestXCIntPlan(object):

    def __init__(
        self,
        *,
        handle : CuestHandle,
        basis : CuestAOBasis,
        grid : CuestMolecularGrid,
        functional,
        xc_threshold_collocation,
        ):

        self.initialized = False

        self.handle = handle
        self.functional = functional
        self.xc_threshold_collocation = xc_threshold_collocation

        xc_int_plan_parameters = CuestParameters(
            parametersType=ce.CuestParametersType.CUEST_XCINTPLAN_PARAMETERS,
            )

        xc_threshold_collocation_data = ce.data_double(self.xc_threshold_collocation)
        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_XCINTPLAN_PARAMETERS,
            parameters=xc_int_plan_parameters.parameters,
            attribute=ce.CuestXCIntPlanParametersAttributes.CUEST_XCINTPLAN_PARAMETERS_THRESHOLD_COLLOCATION,
            attributeValue=xc_threshold_collocation_data,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestXCIntPlanConfigure failed')

        self.xc_int_plan_handle = ce.cuestXCIntPlanHandle()

        # => Workspace Query <= #

        persistent_workspace_descriptor = CuestWorkspaceDescriptor()
        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestXCIntPlanCreateWorkspaceQuery(
            handle=handle.handle,
            basis=basis.ao_basis_handle,
            grid=grid.moleculargrid_handle,
            functional=functional,
            parameters=xc_int_plan_parameters.parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPlan=self.xc_int_plan_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestXCIntPlanCreateWorkspaceQuery failed')

        persistent_workspace = CuestWorkspace(workspaceDescriptor=persistent_workspace_descriptor)
        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        # => Creation <= #

        status = ce.cuestXCIntPlanCreate(
            handle=handle.handle,
            basis=basis.ao_basis_handle,
            grid=grid.moleculargrid_handle,
            functional=functional,
            parameters=xc_int_plan_parameters.parameters,
            persistentWorkspace=persistent_workspace.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            outPlan=self.xc_int_plan_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestXCIntPlanCreate failed')

        # Bind the lifetime of persistent_workspace to this object
        self.persistent_workspace = persistent_workspace

        self.initialized = True

    def __del__(self):

        if not self.initialized: return

        status = ce.cuestXCIntPlanDestroy(
            handle=self.xc_int_plan_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestXCIntPlanDestroy failed')

    @property
    def engine_citation(self):
        query = ce.data_string()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=self.xc_int_plan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_ENGINE_CITATION,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed')

        return query.value

    @property
    def engine_description(self):
        query = ce.data_string()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=self.xc_int_plan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_ENGINE_DESCRIPTION,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed')

        return query.value

    @property
    def functional_citation(self):
        query = ce.data_string()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=self.xc_int_plan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_FUNCTIONAL_CITATION,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed')

        return query.value

    @property
    def functional_description(self):
        query = ce.data_string()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=self.xc_int_plan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_FUNCTIONAL_DESCRIPTION,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed')

        return query.value

    @memoized_property
    def functional_name(self):
        return XCFunctionalInfo.enum_to_string(self.functional)

    @memoized_property
    def exchange_scale(self):
        exchange_scale = ce.data_double()
        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=self.xc_int_plan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_EXCHANGE_SCALE,
            attributeValue=exchange_scale,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestQuery failed')
        return exchange_scale.value

    @memoized_property
    def lrc_exchange_scale(self):
        lrc_exchange_scale = ce.data_double()
        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=self.xc_int_plan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_LRC_EXCHANGE_SCALE,
            attributeValue=lrc_exchange_scale,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestQuery failed')
        return lrc_exchange_scale.value

    @memoized_property
    def lrc_omega(self):
        lrc_omega = ce.data_double()
        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=self.xc_int_plan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_LRC_OMEGA,
            attributeValue=lrc_omega,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestQuery failed')
        return lrc_omega.value

    @memoized_property
    def vv10_scale(self):
        vv10_scale = ce.data_double()
        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=self.xc_int_plan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_VV10_SCALE,
            attributeValue=vv10_scale,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestQuery failed')
        return vv10_scale.value

    @memoized_property
    def vv10_c(self):
        vv10_c = ce.data_double()
        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=self.xc_int_plan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_VV10_C,
            attributeValue=vv10_c,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestQuery failed')
        return vv10_c.value

    @memoized_property
    def vv10_b(self):
        vv10_b = ce.data_double()
        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_XCINTPLAN,
            object=self.xc_int_plan_handle,
            attribute=ce.CuestXCIntPlanAttributes.CUEST_XCINTPLAN_VV10_B,
            attributeValue=vv10_b,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestQuery failed')
        return vv10_b.value

    # => String Details <= #

    def __str__(self):
        return self.string

    @property
    def string(self):
        s = ''
        s += 'XCIntPlan:\n';
        if self.engine_description and self.engine_citation:
            s += '%-10s = %s\n' % ('engine', self.engine_description)
            s += '%-10s = %s\n' % ('citation', self.engine_citation)
        s += '%-10s = %s\n'     % ('XC brief', self.functional_description)
        s += '%-10s = %s\n'     % ('citation', self.functional_citation)
        s += '%-10s = %10s\n'   % ('functional', self.functional_name)
        s += '%-10s = %10.3f\n' % ('hf scale',   self.exchange_scale)
        s += '%-10s = %10.3f\n' % ('lrc scale',  self.lrc_exchange_scale)
        if self.lrc_exchange_scale != 0.0:
            s += '%-10s = %10.3f\n' % ('lrc omega', self.lrc_omega)
        s += '%-10s = %10.3f\n' % ('vv10 scale', self.vv10_scale)
        if self.vv10_scale != 0.0:
            s += '%-10s = %10.3f\n' % ('vv10 b', self.vv10_b)
            s += '%-10s = %10.3f\n' % ('vv10 c', self.vv10_c)
        return s;

