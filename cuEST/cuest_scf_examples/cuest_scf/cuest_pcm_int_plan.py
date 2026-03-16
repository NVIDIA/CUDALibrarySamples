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
import numpy as np

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace

from .cuest_handle import CuestHandle
from .cuest_ao_basis import CuestAOBasis
from .cuest_molecular_grid import CuestMolecularGrid

from .cuest_parameters import CuestParameters

from .cuest_oe_int_plan import CuestOEIntPlan
from .periodic_table import PeriodicTable
from .unit_conversions import UnitConversions


# York and Karplus, J. Phys. Chem. A, 103, 11060-11079 (1999) - Table 1.
_zeta_map = {
      14 : 4.865,
      26 : 4.855,
      50 : 4.893,
     110 : 4.901,
     194 : 4.903,
     302 : 4.905,
     434 : 4.906,
     590 : 4.905,
     770 : 4.899,
     974 : 4.907,
    1202 : 4.907,
    }

# First source:
# Truhlar and coworkers, J. Phys. Chem. A, 113, 5806-5812 (2009) - Table 12.
#
# Second source (radii not given in the 2009 paper come from here):
# CRC Handbook of Chemistry and Physics, 95th Edition.
# Pages 9-49 to 9-50.
# Atomic Radii of the Elements
# Manjeera Mantina, Rosendo Valero, Christopher J. Cramer, and Donald G. Truhlar
#
_bondi_radii_ang = {
    'H'  : 1.10,
    'HE' : 1.40,
    'LI' : 1.81,
    'BE' : 1.53,
    'B'  : 1.92,
    'C'  : 1.70,
    'N'  : 1.55,
    'O'  : 1.52,
    'F'  : 1.47,
    'NE' : 1.54,
    'NA' : 2.27,
    'MG' : 1.73,
    'AL' : 1.84,
    'SI' : 2.10,
    'P'  : 1.80,
    'S'  : 1.80,
    'CL' : 1.75,
    'AR' : 1.88,
    'K'  : 2.75,
    'CA' : 2.31,
    'SC' : 2.15,
    'TI' : 2.11,
    'V'  : 2.07,
    'CR' : 2.06,
    'MN' : 2.05,
    'FE' : 2.04,
    'CO' : 2.00,
    'NI' : 1.97,
    'CU' : 1.96,
    'ZN' : 2.01,
    'GA' : 1.87,
    'GE' : 2.11,
    'AS' : 1.85,
    'SE' : 1.90,
    'BR' : 1.83,
    'KR' : 2.02,
    'RB' : 3.03,
    'SR' : 2.49,
    'Y'  : 2.32,
    'ZR' : 2.23,
    'NB' : 2.18,
    'MO' : 2.17,
    'TC' : 2.16,
    'RU' : 2.13,
    'RH' : 2.10,
    'PD' : 2.10,
    'AG' : 2.11,
    'CD' : 2.18,
    'IN' : 1.93,
    'SN' : 2.17,
    'SB' : 2.06,
    'TE' : 2.06,
    'I'  : 1.98,
    'XE' : 2.16,
    'CS' : 3.43,
    'BA' : 2.68,
    'LA' : 2.43,
    'CE' : 2.42,
    'PR' : 2.40,
    'ND' : 2.39,
    'PM' : 2.38,
    'SM' : 2.36,
    'EU' : 2.35,
    'GD' : 2.34,
    'TB' : 2.33,
    'DY' : 2.31,
    'HO' : 2.30,
    'ER' : 2.29,
    'TM' : 2.27,
    'YB' : 2.26,
    'LU' : 2.24,
    'HF' : 2.23,
    'TA' : 2.22,
    'W'  : 2.18,
    'RE' : 2.16,
    'OS' : 2.16,
    'IR' : 2.13,
    'PT' : 2.13,
    'AU' : 2.14,
    'HG' : 2.23,
    'TL' : 1.96,
    'PB' : 2.02,
    'BI' : 2.07,
    'PO' : 1.97,
    'AT' : 2.02,
    'RN' : 2.20,
    'FR' : 3.48,
    'RA' : 2.83,
    'AC' : 2.47,
    'TH' : 2.45,
    'PA' : 2.43,
    'U'  : 2.41,
    'NP' : 2.39,
    'PU' : 2.43,
    'AM' : 2.44,
    'CM' : 2.45,
    'BK' : 2.44,
    'CF' : 2.45,
    'ES' : 2.45,
    'FM' : 2.45,
    'MD' : 2.46,
    'NO' : 2.46,
    'LR' : 2.46,
    }

class CuestPCMIntPlan(object):


    def __init__(
        self,
        *,
        handle : CuestHandle,
        intPlan : CuestOEIntPlan,
        epsilon : float,
        x_prefactor : float,
        num_angular_points_per_hydrogen_atom : int,
        num_angular_points_per_heavy_atom : int,
        Ns : np.ndarray,
        effective_nuclear_charges : np.ndarray,
        cutoff : float,
        convergence_tol : float,
        atomic_radii_scale=1.2
        ):

        bohr_per_ang=UnitConversions.conversions['bohr_per_ang']

        if num_angular_points_per_hydrogen_atom not in _zeta_map:
            raise RuntimeError(f"Invalid num_angular_points_per_hydrogen_atom value ({num_angular_points_per_hydrogen_atom}) is not a valid Lebedev number")
        if num_angular_points_per_heavy_atom not in _zeta_map:
            raise RuntimeError(f"Invalid num_angular_points_per_heavy_atom value ({num_angular_points_per_heavy_atom}) is not a valid Lebedev number")

        self.initialized = False

        self.handle = handle
        self.num_angular_points_per_hydrogen_atom=num_angular_points_per_hydrogen_atom
        self.num_angular_points_per_heavy_atom=num_angular_points_per_heavy_atom
        self.epsilon=epsilon
        self.x_prefactor=x_prefactor
        self.cutoff = cutoff
        self.convergence_tol = convergence_tol

        symbols_upper = [PeriodicTable.N_to_symbol_upper_table[N] for N in Ns]
        for sym in symbols_upper:
            if sym not in _bondi_radii_ang:
                raise RuntimeError(f"No Bondi radius defined for element {sym}")
        grid_sizes = [num_angular_points_per_hydrogen_atom if s == "H" else num_angular_points_per_heavy_atom for s in symbols_upper]
        zetas=[_zeta_map[grid_size] for grid_size in grid_sizes]
        atomic_radii=atomic_radii_scale*np.array([_bondi_radii_ang[symbol_upper] * bohr_per_ang for symbol_upper in symbols_upper])

        pcm_int_plan_parameters = CuestParameters(
            parametersType=ce.CuestParametersType.CUEST_PCMINTPLAN_PARAMETERS,
            )

        pcm_cutoff = ce.data_double(self.cutoff)
        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_PCMINTPLAN_PARAMETERS,
            parameters=pcm_int_plan_parameters.parameters,
            attribute=ce.CuestPCMIntPlanParametersAttributes.CUEST_PCMINTPLAN_PARAMETERS_CUTOFF,
            attributeValue=pcm_cutoff,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestParametersConfigure failed')

        pcm_x_prefactor = ce.data_double(self.x_prefactor)
        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_PCMINTPLAN_PARAMETERS,
            parameters=pcm_int_plan_parameters.parameters,
            attribute=ce.CuestPCMIntPlanParametersAttributes.CUEST_PCMINTPLAN_PARAMETERS_X_PREFACTOR,
            attributeValue=pcm_x_prefactor,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestParametersConfigure failed')

        self.pcm_int_plan_handle = ce.cuestPCMIntPlanHandle()

        # => Workspace Query <= #

        persistent_workspace_descriptor = CuestWorkspaceDescriptor()
        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestPCMIntPlanCreateWorkspaceQuery(
            handle=handle.handle,
            intPlan=intPlan.oe_int_plan_handle,
            parameters=pcm_int_plan_parameters.parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numAngularPointsPerAtom=grid_sizes,
            epsilon=epsilon,
            zetas=zetas,
            atomicRadii=atomic_radii,
            effectiveNuclearCharges=effective_nuclear_charges,
            outPlan=self.pcm_int_plan_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestPCMIntPlanCreateWorkspaceQuery failed')

        persistent_workspace = CuestWorkspace(workspaceDescriptor=persistent_workspace_descriptor)
        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        # => Creation <= #

        status = ce.cuestPCMIntPlanCreate(
            handle=handle.handle,
            intPlan=intPlan.oe_int_plan_handle,
            parameters=pcm_int_plan_parameters.parameters,
            persistentWorkspace=persistent_workspace.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            numAngularPointsPerAtom=grid_sizes,
            epsilon=epsilon,
            zetas=zetas,
            atomicRadii=atomic_radii,
            effectiveNuclearCharges=effective_nuclear_charges,
            outPlan=self.pcm_int_plan_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestPCMIntPlanCreate failed')

        # Bind the lifetime of persistent_workspace to this object
        self.persistent_workspace = persistent_workspace

        self.initialized = True

    def __del__(self):

        if not self.initialized: return

        status = ce.cuestPCMIntPlanDestroy(
            handle=self.pcm_int_plan_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestPCMIntPlanDestroy failed')

    @memoized_property
    def npoint(self):
        npoint = ce.data_uint64_t()
        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_PCMINTPLAN,
            object=self.pcm_int_plan_handle,
            attribute=ce.CuestPCMIntPlanAttributes.CUEST_PCMINTPLAN_NUM_POINT,
            attributeValue=npoint,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestQuery failed')
        return npoint.value

    @memoized_property
    def n_active_point(self):
        npoint = ce.data_uint64_t()
        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_PCMINTPLAN,
            object=self.pcm_int_plan_handle,
            attribute=ce.CuestPCMIntPlanAttributes.CUEST_PCMINTPLAN_NUM_ACTIVE_POINT,
            attributeValue=npoint,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestQuery failed')
        return npoint.value

    # => String Details <= #

    def __str__(self):
        return self.string

    @property
    def string(self):
        s = ''
        s += 'PCMIntPlan:\n'
        s += '%-10s = %10.4f\n' % ('dielectric', self.epsilon)
        s += '%-10s = %10.1e\n' % ('tolerance', self.convergence_tol)
        s += '%-10s = %10.1e\n' % ('cutoff', self.cutoff)
        s += '%-10s = %10d\n'   % ('npoint', self.npoint)
        s += '%-10s = %10d\n'   % ('nactive', self.n_active_point)
        return s

