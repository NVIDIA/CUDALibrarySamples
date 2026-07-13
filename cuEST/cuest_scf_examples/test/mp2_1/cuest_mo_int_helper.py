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

from cuest_scf.ao_basis import AOBasis
from cuest_scf.memoized_property import memoized_property
from cuest_scf.molecule import Molecule
from cuest_scf.gpu_matrix import GPUMatrix
from cuest_scf.gpu_matrix_utility import GPUMatrixUtility

from cuest_scf.cuest_handle import CuestHandle
from cuest_scf.cuest_ao_basis import CuestAOBasis
from cuest_scf.cuest_ao_pair_list import CuestAOPairList
from cuest_scf.cuest_df_int_plan import CuestDFIntPlan
from cuest_scf.cuest_workspace import CuestWorkspace
from cuest_scf.cuest_workspace_descriptor import CuestWorkspaceDescriptor

class CuestMOIntegralHelper(object):
    def __init__(
        self,
        *,
        cuest_handle : CuestHandle,
        molecule : Molecule,
        primary : AOBasis,
        ri_auxiliary : AOBasis,
        primary_name : str,
        ri_auxiliary_name : str,
        threshold_pq : float,
        integral_direct : bool = False,
        df_fitting_eigenvalue_cutoff : float = 1.0e-12,
        ):

        self.cuest_handle = cuest_handle
        self.molecule = molecule
        self.primary = primary
        self.ri_auxiliary = ri_auxiliary
        self.primary_name = primary_name
        self.ri_auxiliary_name = ri_auxiliary_name
        self.threshold_pq = threshold_pq
        self.integral_direct = integral_direct
        self.df_fitting_eigenvalue_cutoff = df_fitting_eigenvalue_cutoff


    @memoized_property
    def cuest_primary(self):
        return CuestAOBasis(
            handle=self.cuest_handle,
            basis=self.primary,
            )

    @memoized_property
    def cuest_ri_auxiliary(self):
        return CuestAOBasis(
            handle=self.cuest_handle,
            basis=self.ri_auxiliary,
            )

    @memoized_property
    def cuest_ao_pair_list(self):
        return CuestAOPairList(
            handle=self.cuest_handle,
            basis=self.cuest_primary,
            xyz=self.molecule.xyz,
            threshold_pq=self.threshold_pq,
            )

    @memoized_property
    def df_int_plan(self):
        return CuestDFIntPlan(
            handle=self.cuest_handle,
            primary=self.cuest_primary,
            auxiliary=self.cuest_ri_auxiliary,
            ao_pair_list=self.cuest_ao_pair_list,
            df_fitting_eigenvalue_cutoff=self.df_fitting_eigenvalue_cutoff,
            integral_direct=self.integral_direct,
            )

    def compute_3c_integral(
        self,
        C : GPUMatrix,
        nmo_c1 : int,
        c1_offset : int,
        nmo_c2 : int,
        c2_offset : int,
        out_mo_integrals : GPUMatrix,
        memory_in_bytes : int = 2 * 1024 * 1024 * 1024,
        ):

        nao = C.ncols
        if self.primary.nao != nao:
            raise RuntimeError('C.ncols != primary.nao')

        dfmo_compute_parameters = ce.cuestDFMOIntegralsComputeParameters()
        status = ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS,
            outParameters=dfmo_compute_parameters,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFMOIntegralsComputeParametersCreate failed')

        variable_buffer_size = CuestWorkspaceDescriptor(deviceBufferSizeInBytes=memory_in_bytes)
        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        C_left_handle = ce.Pointer(C.pointer + c1_offset * nao * C.type_size)
        C_right_handle = ce.Pointer(C.pointer + c2_offset * nao * C.type_size)
        out_mo_integrals_handle = ce.Pointer(out_mo_integrals.pointer)

        status = ce.cuestDFMOIntegralsComputeWorkspaceQuery(
            handle=self.cuest_handle.handle,
            plan=self.df_int_plan.df_int_plan_handle,
            parameters=dfmo_compute_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numCoefficientMatrices=1,
            numLeftOrbitals=[nmo_c1],
            numRightOrbitals=[nmo_c2],
            leftCoefficientMatrices=C_left_handle,
            rightCoefficientMatrices=C_right_handle,
            outTensors=out_mo_integrals_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFMOIntegralsComputeWorkspaceQuery failed', status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestDFMOIntegralsCompute(
            handle=self.cuest_handle.handle,
            plan=self.df_int_plan.df_int_plan_handle,
            parameters=dfmo_compute_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            numCoefficientMatrices=1,
            numLeftOrbitals=[nmo_c1],
            numRightOrbitals=[nmo_c2],
            leftCoefficientMatrices=C_left_handle,
            rightCoefficientMatrices=C_right_handle,
            outTensors=out_mo_integrals_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFMOIntegralsCompute failed')

        status = ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS,
            parameters=dfmo_compute_parameters,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFMOIntegralsComputeParametersDestroy failed')

