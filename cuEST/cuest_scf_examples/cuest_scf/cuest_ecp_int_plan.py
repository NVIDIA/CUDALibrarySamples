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

from .memoized_property import memoized_property

import cuest.bindings as ce    
import numpy as np

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace

from .cuest_handle import CuestHandle
from .cuest_ao_basis import CuestAOBasis

from .cuest_parameters import CuestParameters

from .periodic_table import PeriodicTable
from .unit_conversions import UnitConversions

class CuestECPIntPlan(object):


    def __init__(
        self,
        *,
        handle : CuestHandle,
        basis : CuestAOBasis,
        xyzs : np.ndarray,
        numECPAtoms: int,
        activeIndices: list,
        activeAtoms: list,
        ):

        # NOTE: xyzs is expected in bohr (Molecule.xyz is already converted from
        # angstrom on parse), matching the units of `basis`. Do NOT apply a
        # bohr_per_ang conversion here.

        self.initialized = False

        self.handle = handle
        self.xyzs = xyzs

        # => Input validation <= #
        # Sizes flow directly into the C plan API; a mismatch would otherwise
        # cause an out-of-bounds read in the backend.
        natom = basis.natom
        numECPAtoms = int(numECPAtoms)

        xyz_flat = xyzs.reshape(-1)
        if xyz_flat.size != 3 * natom:
            raise RuntimeError(
                'xyzs has %d entries, expected 3*natom = %d' % (xyz_flat.size, 3 * natom)
            )

        if len(activeIndices) != numECPAtoms:
            raise RuntimeError(
                'len(activeIndices)=%d does not match numECPAtoms=%d'
                % (len(activeIndices), numECPAtoms)
            )

        if len(activeAtoms) != numECPAtoms:
            raise RuntimeError(
                'len(activeAtoms)=%d does not match numECPAtoms=%d'
                % (len(activeAtoms), numECPAtoms)
            )

        if any(idx < 0 or idx >= natom for idx in activeIndices):
            raise RuntimeError(
                'activeIndices contains an out-of-range atom index (natom=%d)' % natom
            )

        ecp_int_plan_parameters = CuestParameters(
            parametersType=ce.CuestParametersType.CUEST_ECPINTPLAN_PARAMETERS,
            )


        self.ecp_int_plan_handle = ce.cuestECPIntPlanHandle()

        # => Workspace Query <= #

        persistent_workspace_descriptor = CuestWorkspaceDescriptor()
        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        xyz = xyz_flat.astype(float).tolist()
        status = ce.cuestECPIntPlanCreateWorkspaceQuery(
            handle=handle.handle,
            basis=basis.ao_basis_handle, 
            xyz=xyz, 
            numECPAtoms=numECPAtoms, 
            activeIndices=activeIndices, 
            activeAtoms=activeAtoms, 
            parameters=ecp_int_plan_parameters.parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPlan=self.ecp_int_plan_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestECPIntPlanCreateWorkspaceQuery failed')

        ecp_int_plan_persistent_workspace = CuestWorkspace(workspaceDescriptor=persistent_workspace_descriptor)
        ecp_int_plan_temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        # => Creation <= #

        status = ce.cuestECPIntPlanCreate(
            handle=handle.handle,
            basis=basis.ao_basis_handle, 
            xyz=xyz, 
            numECPAtoms=numECPAtoms, 
            activeIndices=activeIndices, 
            activeAtoms=activeAtoms, 
            parameters=ecp_int_plan_parameters.parameters,
            persistentWorkspace=ecp_int_plan_persistent_workspace.pointer,
            temporaryWorkspace=ecp_int_plan_temporary_workspace.pointer,
            outPlan=self.ecp_int_plan_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestECPIntPlanCreate failed')

        # Bind the lifetime of persistent_workspace to this object
        self.persistent_workspace = ecp_int_plan_persistent_workspace

        self.initialized = True

    def __del__(self):

        if not self.initialized: return

        status = ce.cuestECPIntPlanDestroy(
            handle=self.ecp_int_plan_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestECPIntPlanDestroy failed')
