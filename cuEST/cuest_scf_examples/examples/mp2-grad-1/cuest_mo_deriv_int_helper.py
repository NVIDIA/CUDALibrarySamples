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


import numpy as np

import cuest.bindings as ce

from cuest_scf.ao_basis import AOBasis
from cuest_scf.ecp_basis import ECPBasis
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

# Core electrons frozen per element, by atomic number (H through Ra)
_FROZEN_CORE_SHELLS = [
  0, # Dummy
  0,                                                                  0, # He
  0,  0,                                          2,  2,  2,  2,  2,  2, # Ne
  2,  2,                                         10, 10, 10, 10, 10, 10, # Ar
 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 18, 18, 18, 18, 18, 18, # Kr
 18, 18, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 36, 36, 36, 36, 36, 36, # Xe
 36, 36,
         36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,         # Lanthanide Series
         46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 68, 68, 68, 68, 68, 68, # Rn
 68, 68] # Ra

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
        ecp_basis : ECPBasis = None,
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
        self.ecp_basis = ecp_basis

    def compute_frozen_core(self):
        """Number of doubly occupied core orbitals to freeze.

        Electrons replaced by an ECP are not orbitals of the calculation, so they are taken
        off the count.
        """

        use_ecp = self.ecp_basis is not None and self.ecp_basis.is_active
        nfrozen = 0
        for i, n in enumerate(self.molecule.N):
            if int(n) >= len(_FROZEN_CORE_SHELLS):
                raise ValueError(
                    f"Frozen core is not available for Z={int(n)}; elements past radium are not supported"
                )
            necp = int(self.ecp_basis.atoms[i].nelectron) if use_ecp else 0
            nfrozen += max(_FROZEN_CORE_SHELLS[int(n)] - necp, 0)
        return nfrozen // 2

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
            exchange_scale=1.0,
            lrc_exchange_scale=0.0,
            lrc_omega=0.0,
            df_fitting_eigenvalue_cutoff=self.df_fitting_eigenvalue_cutoff,
            integral_direct=self.integral_direct,
            )

    def compute_3c_integral(
        self,
        C_left : GPUMatrix,
        nmo_c1,
        c1_offset : int,
        C_right : GPUMatrix,
        nmo_c2,
        c2_offset : int,
        out_mo_integrals : GPUMatrix,
        memory_in_bytes : int = 2 * 1024 * 1024 * 1024,
        ):

        # nmo_c1 and nmo_c2 may be sequences, in which case several subspace pairs are
        # transformed in one pass over the AO three-center integrals. The batches are
        # concatenated: C_left holds the left coefficient matrices back to back from
        # c1_offset, C_right likewise, and out_mo_integrals receives each
        # [naux, nmo_c1[d] * nmo_c2[d]] tensor back to back.
        nao = C_left.ncols
        if self.primary.nao != nao:
            raise RuntimeError('C_left.ncols != primary.nao')
        if C_right.ncols != nao:
            raise RuntimeError('C_right.ncols != C_left.ncols')

        num_left_orbitals = [nmo_c1] if isinstance(nmo_c1, int) else list(nmo_c1)
        num_right_orbitals = [nmo_c2] if isinstance(nmo_c2, int) else list(nmo_c2)
        if len(num_left_orbitals) != len(num_right_orbitals):
            raise RuntimeError('nmo_c1 and nmo_c2 describe different numbers of batches')

        dfmo_compute_parameters = ce.cuestDFMOIntegralsComputeParameters()
        status = ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS,
            outParameters=dfmo_compute_parameters,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFMOIntegralsComputeParametersCreate failed')

        variable_buffer_size = CuestWorkspaceDescriptor(deviceBufferSizeInBytes=memory_in_bytes)
        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        C_left_handle = ce.Pointer(C_left.pointer + c1_offset * nao * C_left.type_size)
        C_right_handle = ce.Pointer(C_right.pointer + c2_offset * nao * C_right.type_size)
        out_mo_integrals_handle = ce.Pointer(out_mo_integrals.pointer)

        status = ce.cuestDFMOIntegralsComputeWorkspaceQuery(
            handle=self.cuest_handle.handle,
            plan=self.df_int_plan.df_int_plan_handle,
            parameters=dfmo_compute_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numCoefficientMatrices=len(num_left_orbitals),
            numLeftOrbitals=num_left_orbitals,
            numRightOrbitals=num_right_orbitals,
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
            numCoefficientMatrices=len(num_left_orbitals),
            numLeftOrbitals=num_left_orbitals,
            numRightOrbitals=num_right_orbitals,
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


    def compute_3c_derivative(
        self,
        C_left : GPUMatrix,
        nmo_c1,
        c1_offset : int,
        C_right : GPUMatrix,
        nmo_c2,
        c2_offset : int,
        three_center_tpdm : GPUMatrix,
        out_gradient : GPUMatrix,
        memory_in_bytes : int = 2 * 1024 * 1024 * 1024,
        ):

        # The nuclear gradient counterpart of compute_3c_integral. three_center_tpdm holds
        # gamma3[A,pq] = sum_rs B[A,rs] Gamma[pq,rs], laid out like the tensor that
        # compute_3c_integral writes, and is assumed bra-ket symmetric. Any pair
        # symmetry or spin prefactor must be folded into it by the caller.
        #
        # nmo_c1 and nmo_c2 may be sequences, in which case several density batches are
        # differentiated in one pass over the derivative integrals. The batches are
        # concatenated: C_left holds the left coefficient matrices back to back from
        # c1_offset, C_right likewise, and three_center_tpdm holds each [naux, nmo_c1[d] *
        # nmo_c2[d]] tensor back to back.
        nao = C_left.ncols
        if self.primary.nao != nao:
            raise RuntimeError('C_left.ncols != primary.nao')
        if C_right.ncols != nao:
            raise RuntimeError('C_right.ncols != C_left.ncols')

        num_left_orbitals = [nmo_c1] if isinstance(nmo_c1, int) else list(nmo_c1)
        num_right_orbitals = [nmo_c2] if isinstance(nmo_c2, int) else list(nmo_c2)
        if len(num_left_orbitals) != len(num_right_orbitals):
            raise RuntimeError('nmo_c1 and nmo_c2 describe different numbers of batches')

        dfmo_derivative_parameters = ce.cuestDFMOIntegralsDerivativeComputeParameters()
        status = ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_DFMOINTEGRALSDERIVATIVECOMPUTE_PARAMETERS,
            outParameters=dfmo_derivative_parameters,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFMOIntegralsDerivativeComputeParametersCreate failed')

        variable_buffer_size = CuestWorkspaceDescriptor(deviceBufferSizeInBytes=memory_in_bytes)
        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        C_left_handle = ce.Pointer(C_left.pointer + c1_offset * nao * C_left.type_size)
        C_right_handle = ce.Pointer(C_right.pointer + c2_offset * nao * C_right.type_size)
        three_center_tpdm_handle = ce.Pointer(three_center_tpdm.pointer)
        out_gradient_handle = ce.Pointer(out_gradient.pointer)

        status = ce.cuestDFMOIntegralsDerivativeComputeWorkspaceQuery(
            handle=self.cuest_handle.handle,
            plan=self.df_int_plan.df_int_plan_handle,
            parameters=dfmo_derivative_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numCoefficientMatrices=len(num_left_orbitals),
            numLeftOrbitals=num_left_orbitals,
            numRightOrbitals=num_right_orbitals,
            leftCoefficientMatrices=C_left_handle,
            rightCoefficientMatrices=C_right_handle,
            threeCenterTPDMs=three_center_tpdm_handle,
            outGradient=out_gradient_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFMOIntegralsDerivativeComputeWorkspaceQuery failed', status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestDFMOIntegralsDerivativeCompute(
            handle=self.cuest_handle.handle,
            plan=self.df_int_plan.df_int_plan_handle,
            parameters=dfmo_derivative_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            numCoefficientMatrices=len(num_left_orbitals),
            numLeftOrbitals=num_left_orbitals,
            numRightOrbitals=num_right_orbitals,
            leftCoefficientMatrices=C_left_handle,
            rightCoefficientMatrices=C_right_handle,
            threeCenterTPDMs=three_center_tpdm_handle,
            outGradient=out_gradient_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFMOIntegralsDerivativeCompute failed')

        status = ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_DFMOINTEGRALSDERIVATIVECOMPUTE_PARAMETERS,
            parameters=dfmo_derivative_parameters,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestDFMOIntegralsDerivativeComputeParametersDestroy failed')

