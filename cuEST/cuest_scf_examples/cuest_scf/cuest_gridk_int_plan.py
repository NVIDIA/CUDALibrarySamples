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
from .cuest_molecular_grid import CuestMolecularGrid
from .cuest_oe_int_plan import CuestOEIntPlan

from .cuest_utility import CuestUtility

class CuestGridKIntPlan(object):

    def __init__(
        self,
        *,
        handle : CuestHandle,
        grid : CuestMolecularGrid,
        oe_int_plan : CuestOEIntPlan,
        exchange_scale : float,
        lrc_exchange_scale : float,
        lrc_omega : float,
        threshold_collocation : float,
        fitting_algorithm : str = "none",
        fitting_orthogonalization_cutoff : float = 1.0e-6,
        fitting_cutoff : float = 1.0e-2,
        ):

        self.initialized = False

        # Fitting of the collocated (bra) pair: "none" (raw collocation) or
        # "overlap_symmetric" (symmetric overlap fit; the fitting metric is
        # built at plan creation, from the analytic overlap matrix of the
        # supplied one-electron integral plan, and stored in the plan)
        fitting_algorithm = fitting_algorithm.lower()
        supported_fitting_algorithms = {
            "none": ce.CuestGridKIntPlanParametersFittingAlgorithm.CUEST_GRIDKINTPLAN_PARAMETERS_FITTING_ALGORITHM_NONE,
            "overlap_symmetric": ce.CuestGridKIntPlanParametersFittingAlgorithm.CUEST_GRIDKINTPLAN_PARAMETERS_FITTING_ALGORITHM_OVERLAP_SYMMETRIC,
            }
        if fitting_algorithm not in supported_fitting_algorithms:
            raise RuntimeError(
                f"Unrecognized fitting_algorithm '{fitting_algorithm}'. "
                f"Supported values are {', '.join(supported_fitting_algorithms)}"
                )

        self.handle = handle
        self.exchange_scale = exchange_scale
        self.lrc_exchange_scale = lrc_exchange_scale
        self.lrc_omega = lrc_omega
        self.threshold_collocation = threshold_collocation
        self.fitting_algorithm = fitting_algorithm
        self.fitting_orthogonalization_cutoff = fitting_orthogonalization_cutoff
        self.fitting_cutoff = fitting_cutoff

        gridk_int_plan_parameters = CuestUtility.make_compute_parameters(
            ce.CuestParametersType.CUEST_GRIDKINTPLAN_PARAMETERS,
            [(ce.CuestGridKIntPlanParametersAttributes.CUEST_GRIDKINTPLAN_PARAMETERS_THRESHOLD_COLLOCATION,
              ce.data_double(self.threshold_collocation)),
             (ce.CuestGridKIntPlanParametersAttributes.CUEST_GRIDKINTPLAN_PARAMETERS_EXCHANGE_FRACTION,
              ce.data_double(self.exchange_scale)),
             (ce.CuestGridKIntPlanParametersAttributes.CUEST_GRIDKINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION,
              ce.data_double(self.lrc_exchange_scale)),
             (ce.CuestGridKIntPlanParametersAttributes.CUEST_GRIDKINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA,
              ce.data_double(self.lrc_omega)),
             (ce.CuestGridKIntPlanParametersAttributes.CUEST_GRIDKINTPLAN_PARAMETERS_FITTING_ALGORITHM,
              ce.data_cuestGridKIntPlanParametersFittingAlgorithm_t(supported_fitting_algorithms[self.fitting_algorithm])),
             (ce.CuestGridKIntPlanParametersAttributes.CUEST_GRIDKINTPLAN_PARAMETERS_FITTING_ORTHOGONALIZATION_CUTOFF,
              ce.data_double(self.fitting_orthogonalization_cutoff)),
             (ce.CuestGridKIntPlanParametersAttributes.CUEST_GRIDKINTPLAN_PARAMETERS_FITTING_CUTOFF,
              ce.data_double(self.fitting_cutoff))],
            )

        self.gridk_int_plan_handle = ce.cuestGridKIntPlanHandle()

        # => Workspace Query <= #

        # Allow the algorithm to use up to 4GB of DRAM as the grid-blocking
        # budget.
        # NOTE: Production codes may wish to configure the maximum temporary
        # workspace memory as a user parameter or heuristic
        variable_buffer_descriptor = CuestWorkspaceDescriptor(
            deviceBufferSizeInBytes=4000000000,
            )

        persistent_workspace_descriptor = CuestWorkspaceDescriptor()
        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestGridKIntPlanCreateWorkspaceQuery(
            handle=handle.handle,
            grid=grid.moleculargrid_handle,
            intPlan=oe_int_plan.oe_int_plan_handle,
            parameters=gridk_int_plan_parameters.parameters,
            variableBufferSize=variable_buffer_descriptor.pointer,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPlan=self.gridk_int_plan_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestGridKIntPlanCreateWorkspaceQuery failed: %s' % status)

        # Most likely place to run out of device memory, so we will add a custom RuntimeError
        try:
            persistent_workspace = CuestWorkspace(workspaceDescriptor=persistent_workspace_descriptor)
            temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)
        except RuntimeError as e:
            raise RuntimeError("CuestGridKIntPlan workspace allocation failed - out of memory for GridK") from e

        # => Creation <= #

        status = ce.cuestGridKIntPlanCreate(
            handle=handle.handle,
            grid=grid.moleculargrid_handle,
            intPlan=oe_int_plan.oe_int_plan_handle,
            parameters=gridk_int_plan_parameters.parameters,
            variableBufferSize=variable_buffer_descriptor.pointer,
            persistentWorkspace=persistent_workspace.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            outPlan=self.gridk_int_plan_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestGridKIntPlanCreate failed: %s' % status)

        # Bind the lifetime of persistent_workspace to this object
        self.persistent_workspace = persistent_workspace

        # Keep wrapper dependencies alive while the native cuEST object may
        # refer to them; this mirrors the C API lifetime requirements. The
        # primary basis, coordinates, and pair structure come from the
        # one-electron integral plan.
        self.grid = grid
        self.oe_int_plan = oe_int_plan
        self.primary = oe_int_plan.basis
        self.ao_pair_list = oe_int_plan.ao_pair_list

        self.initialized = True

    @property
    def fitting_num_mo(self):
        # The fitting attributes only exist in the plan when fitting is active;
        # cuestQuery fails for them otherwise
        if self.fitting_algorithm == 'none':
            return None

        query = ce.data_uint64_t()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_GRIDKINTPLAN,
            object=self.gridk_int_plan_handle,
            attribute=ce.CuestGridKIntPlanAttributes.CUEST_GRIDKINTPLAN_FITTING_NUM_MO,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed: %s' % status)

        return int(query.value)

    @property
    def fitting_max_eigenvalue(self):
        if self.fitting_algorithm == 'none':
            return None

        query = ce.data_double()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_GRIDKINTPLAN,
            object=self.gridk_int_plan_handle,
            attribute=ce.CuestGridKIntPlanAttributes.CUEST_GRIDKINTPLAN_FITTING_MAX_EIGENVALUE,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed: %s' % status)

        return float(query.value)

    @property
    def fitting_min_eigenvalue(self):
        if self.fitting_algorithm == 'none':
            return None

        query = ce.data_double()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_GRIDKINTPLAN,
            object=self.gridk_int_plan_handle,
            attribute=ce.CuestGridKIntPlanAttributes.CUEST_GRIDKINTPLAN_FITTING_MIN_EIGENVALUE,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed: %s' % status)

        return float(query.value)

    def query_canonical_orthogonalization_matrix(
        self,
        *,
        Xptr,
        num_mo,
        nao,
        ):

        if self.fitting_algorithm == 'none':
            return None

        # X is num_mo x nao (row-major); Xptr must reference a device array of
        # num_mo * nao doubles
        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_GRIDKINTPLAN,
            object=self.gridk_int_plan_handle,
            attribute=ce.CuestGridKIntPlanAttributes.CUEST_GRIDKINTPLAN_CANONICAL_ORTHOGONALIZATION_MATRIX_GPU,
            attributeValue=ce.Pointer(Xptr),
            attributeValueSize=8 * num_mo * nao,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed: %s' % status)

    # => String Details <= #

    def __str__(self):
        return self.string

    @property
    def string(self):
        s = ''
        s += 'GridKIntPlan:\n\n'
        s += '%-19s = %11.3E\n' % ('collocation', self.threshold_collocation)
        s += '%-19s = %11s\n' % ('fitting', self.fitting_algorithm)
        if self.fitting_algorithm != 'none':
            s += '%-19s = %11.3E\n' % ('fitting cutoff', self.fitting_cutoff)
            s += '%-19s = %11d\n'   % ('nao', self.primary.nao)
            s += '%-19s = %11d\n'   % ('nmo', self.fitting_num_mo)
            s += '%-19s = %11.3E\n' % ('min eigenvalue', self.fitting_min_eigenvalue)
            s += '%-19s = %11.3E\n' % ('max eigenvalue', self.fitting_max_eigenvalue)
        return s

    def __del__(self):

        if not self.initialized: return

        status = ce.cuestGridKIntPlanDestroy(
            handle=self.gridk_int_plan_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestGridKIntPlanDestroy failed: %s' % status)
