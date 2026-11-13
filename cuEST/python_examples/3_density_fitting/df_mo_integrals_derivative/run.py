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
import numpy as np
import argparse
from pathlib import Path
import sys
import os

# Inject the directory where the example helper utilities live
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helpers.cuda_utils import cuda_malloc, cuda_free, cuda_memcpy_dtoh, cuda_memcpy_htod, WorkspaceDescriptor, Workspace
from helpers.utilities import normalize_shell_coefficients
from helpers.parsers import simple_xyz_parser, simple_gbs_parser

# This sample shows how to use the density-fitting (DF) MO integral
# derivative routine to compute the nuclear gradient of an energy written
# as a contraction of a four-index two-particle density matrix (TPDM)
# with MO basis integrals,
#
#     E = sum_{pqrs} (pq|rs) Gamma_pq^rs.
#
# The derivative routine does not accept the four-index TPDM directly.
# Instead, it takes the three-index TPDM
#
#     Gamma_pq^A = sum_{rs} B_rs^A Gamma_pq^rs,
#
# where B are the DF integrals without the operator weighting, obtained
# from cuestDFMOIntegralsCompute with the operator weighting set to NONE.
# Two cases are shown below:
#   - Case 1: the simple (pq|rs) Gamma_pq^rs term, which is bra-ket
#     symmetric, Gamma_pq^rs = Gamma_rs^pq.
#   - Case 2: the (ij|ab) Gamma_ij^ab term, with i,j occupied and a,b
#     virtual, which is not bra-ket symmetric.  The three-index TPDMs
#     Gamma_ij^A and Gamma_ab^A, each scaled by 1/2, are passed as a batch
#     of two so that bra-ket symmetry is preserved.
#
# Synthetic data is generated to populate these arrays.  The number of
# occupied orbitals is calculated from the sum of atomic charges.

def run(
    *,
    xyz_filename,
    gbs_filename,
    aux_gbs_filename,
    ):

    # => Parse User Input <= #

    symbols, xyzs, Zs = simple_xyz_parser(
        filename=xyz_filename,
        to_bohr_scale_factor=1.0 / 0.52917720859,
        )

    shellinfo = simple_gbs_parser(
        filename=gbs_filename,
        symbols=symbols,
        )
    aux_shellinfo = simple_gbs_parser(
        filename=aux_gbs_filename,
        symbols=symbols,
        )
    # The screening threshold used to filter out insignificant shell pairs by their overlap
    threshold_pq = 1.0e-14

    sizeof_double = np.dtype('double').itemsize

    # => Set Up cuEST <= #

    def cuest_check(
        title,
        return_code
        ):
        " A simple helper function to call cuEST functions and check the return code. "

        if return_code != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError(f"{title} failed with code {return_code}")

    persistent_workspace_descriptor = WorkspaceDescriptor()
    temporary_workspace_descriptor = WorkspaceDescriptor()

    cuest_handle_parameters = ce.cuestHandleParameters()

    cuest_check('Create Handle Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
            outParameters=cuest_handle_parameters,
            )
        )

    cuest_handle = ce.cuestHandle()
    cuest_check('Create Cuest Handle',
        ce.cuestCreate(
            parameters=cuest_handle_parameters,
            handle=cuest_handle,
            )
        )

    cuest_check('Destroy Handle Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
            parameters=cuest_handle_parameters,
            )
        )

    aoshell_parameters = ce.cuestAOShellParameters()
    cuest_check('Create AO Shell Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_AOSHELL_PARAMETERS,
            outParameters=aoshell_parameters,
            )
        )

    # => Build Gaussian Shells From Basis Parser Info <= #

    # Orbital Basis
    shells = []
    n_shells_per_atom = []
    shell_count = 1
    for atom_shells in shellinfo:
        n_shells_per_atom.append(len(atom_shells))
        for shell in atom_shells:
            L = shell['L']
            exponents = shell['exponents']
            raw_coefficients = shell['coefficients']
            normalized_coefficients = normalize_shell_coefficients(
                coefficients=raw_coefficients,
                exponents=exponents,
                L=L,
                normalization=shell['normalization'],
                )
            aoshell_handle = ce.cuestAOShellHandle()
            cuest_check(f'Create AO Shell {shell_count}',
                ce.cuestAOShellCreate(
                    handle=cuest_handle,
                    isPure=shell['is_pure'],
                    L=L,
                    numPrimitive=len(exponents),
                    exponents=exponents,
                    coefficients=normalized_coefficients,
                    parameters=aoshell_parameters,
                    outShell=aoshell_handle)
                )
            shells.append(aoshell_handle)
            shell_count += 1

    # Auxiliary Basis
    aux_shells = []
    aux_n_shells_per_atom = []
    shell_count = 1
    for atom_shells in aux_shellinfo:
        aux_n_shells_per_atom.append(len(atom_shells))
        for shell in atom_shells:
            L = shell['L']
            exponents = shell['exponents']
            raw_coefficients = shell['coefficients']
            normalized_coefficients = normalize_shell_coefficients(
                coefficients=raw_coefficients,
                exponents=exponents,
                L=L,
                normalization=shell['normalization'],
                )
            aoshell_handle = ce.cuestAOShellHandle()
            cuest_check(f'Create Aux AO Shell {shell_count}',
                ce.cuestAOShellCreate(
                    handle=cuest_handle,
                    isPure=shell['is_pure'],
                    L=L,
                    numPrimitive=len(exponents),
                    exponents=exponents,
                    coefficients=normalized_coefficients,
                    parameters=aoshell_parameters,
                    outShell=aoshell_handle)
                )
            aux_shells.append(aoshell_handle)
            shell_count += 1

    cuest_check('Destroy AO Shell Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_AOSHELL_PARAMETERS,
            parameters=aoshell_parameters,
            )
        )

    # => Build Basis Sets From Gaussian Shells <= #

    aobasis_parameters = ce.cuestAOBasisParameters()
    cuest_check('Create AO Basis Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_AOBASIS_PARAMETERS,
            outParameters=aobasis_parameters,
            )
        )

    # Orbital Basis
    aobasis_handle = ce.cuestAOBasisHandle()
    cuest_check('Create AOBasis Workspace Query',
        ce.cuestAOBasisCreateWorkspaceQuery(
            handle=cuest_handle,
            numAtoms=len(symbols),
            numShellsPerAtom=n_shells_per_atom,
            shells=shells,
            parameters=aobasis_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outBasis=aobasis_handle,
            )
        )

    aobasis_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)
    aobasis_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Create AOBasis',
        ce.cuestAOBasisCreate(
            handle=cuest_handle,
            numAtoms=len(symbols),
            numShellsPerAtom=n_shells_per_atom,
            shells=shells,
            parameters=aobasis_parameters,
            persistentWorkspace=aobasis_persistent_workspace.pointer,
            temporaryWorkspace=aobasis_temporary_workspace.pointer,
            outBasis=aobasis_handle,
            )
        )

    del aobasis_temporary_workspace

    for i, shell in enumerate(shells):
        cuest_check(f'Destroy AO Shell {i+1}',
            ce.cuestAOShellDestroy(
                handle=shell,
                )
            )

    # Auxiliary Basis
    auxbasis_handle = ce.cuestAOBasisHandle()
    cuest_check('Create Aux Basis Workspace Query',
        ce.cuestAOBasisCreateWorkspaceQuery(
            handle=cuest_handle,
            numAtoms=len(symbols),
            numShellsPerAtom=aux_n_shells_per_atom,
            shells=aux_shells,
            parameters=aobasis_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outBasis=auxbasis_handle,
            )
        )

    auxbasis_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)
    auxbasis_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Create Aux AOBasis',
        ce.cuestAOBasisCreate(
            handle=cuest_handle,
            numAtoms=len(symbols),
            numShellsPerAtom=aux_n_shells_per_atom,
            shells=aux_shells,
            parameters=aobasis_parameters,
            persistentWorkspace=auxbasis_persistent_workspace.pointer,
            temporaryWorkspace=auxbasis_temporary_workspace.pointer,
            outBasis=auxbasis_handle,
            )
        )

    del auxbasis_temporary_workspace

    for i, shell in enumerate(aux_shells):
        cuest_check(f'Destroy Aux AO Shell {i+1}',
            ce.cuestAOShellDestroy(
                handle=shell,
                )
            )

    aobasis_num_ao = ce.data_uint64_t()
    cuest_check('Query AO Basis Num AO',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=aobasis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_NUM_AO,
            attributeValue=aobasis_num_ao,
            )
        )
    nao = aobasis_num_ao.value

    auxbasis_num_ao = ce.data_uint64_t()
    cuest_check('Query Aux Basis Num AO',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=auxbasis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_NUM_AO,
            attributeValue=auxbasis_num_ao,
            )
        )
    naux = auxbasis_num_ao.value

    cuest_check('Destroy AO Basis Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_AOBASIS_PARAMETERS,
            parameters=aobasis_parameters,
            )
        )

    # => Build Pair List <= #

    aopairlist_handle = ce.cuestAOPairListHandle()

    aopairlist_parameters = ce.cuestAOPairListParameters()

    cuest_check('Create AOPairList Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_AOPAIRLIST_PARAMETERS,
            outParameters=aopairlist_parameters,
            )
        )

    cuest_check('Create AOPairList Workspace Query',
        ce.cuestAOPairListCreateWorkspaceQuery(
            handle=cuest_handle,
            basis=aobasis_handle,
            numAtoms=len(symbols),
            xyz=xyzs,
            thresholdPQ=threshold_pq,
            parameters=aopairlist_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPairList=aopairlist_handle,
            )
        )

    aopairlist_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)
    aopairlist_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Create AOPairList',
        ce.cuestAOPairListCreate(
            handle=cuest_handle,
            basis=aobasis_handle,
            numAtoms=len(symbols),
            xyz=xyzs,
            thresholdPQ=threshold_pq,
            parameters=aopairlist_parameters,
            persistentWorkspace=aopairlist_persistent_workspace.pointer,
            temporaryWorkspace=aopairlist_temporary_workspace.pointer,
            outPairList=aopairlist_handle,
            )
        )

    del aopairlist_temporary_workspace

    cuest_check('Destroy AOPairList Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_AOPAIRLIST_PARAMETERS,
            parameters=aopairlist_parameters,
            )
        )

    # => Build DF Integral Plan <= #

    dfintplan_handle = ce.cuestDFIntPlanHandle()
    dfintplan_parameters = ce.cuestDFIntPlanParameters()
    cuest_check('Create DFIntPlan Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_DFINTPLAN_PARAMETERS,
            outParameters=dfintplan_parameters,
            )
        )

    # The default behavior of the DFIntPlan is to cache the 3-index integrals so
    # they can be read from disk every transform. A more compute-intensive but
    # more memory-efficient approach is to recompute the 3-index integrals on the fly.
    # This can be achieved by configuring the dfintplan_parameters with the
    # CUEST_DFINTPLAN_PARAMETERS_THREE_INDEX_INTEGRAL_DIRECT attribute set to 1.

    # Integral plans that use cached integrals, have a range-separated component,
    # and are intended for use with the MO integral derivative code must be created
    # with the CUEST_DFINTPLAN_PARAMETERS_CACHE_OPERATOR_WEIGHTING_NONE attribute.
    # While it's not strictly necessary to do this for integral direct plans or those
    # that involve the plain Coulomb operator, there's no performance penalty to doing
    # so in these cases; we therefore strongly recommend that
    # CUEST_DFINTPLAN_PARAMETERS_CACHE_OPERATOR_WEIGHTING_NONE be set unconditionally
    # when making a plan that is to be used for MO derivatives.
    cache_operator_weighting = ce.data_cuestDFIntPlanParametersCacheOperatorWeighting_t(
        ce.CuestDFIntPlanParametersCacheOperatorWeighting.CUEST_DFINTPLAN_PARAMETERS_CACHE_OPERATOR_WEIGHTING_NONE
        )
    cuest_check('Configure DFIntPlan Params',
        ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_DFINTPLAN_PARAMETERS,
            parameters=dfintplan_parameters,
            attribute=ce.CuestDFIntPlanParametersAttributes.CUEST_DFINTPLAN_PARAMETERS_CACHE_OPERATOR_WEIGHTING,
            attributeValue=cache_operator_weighting,
            )
        )

    cuest_check('Create DFIntPlan Workspace Query',
        ce.cuestDFIntPlanCreateWorkspaceQuery(
            handle=cuest_handle,
            primaryBasis=aobasis_handle,
            auxiliaryBasis=auxbasis_handle,
            pairList=aopairlist_handle,
            parameters=dfintplan_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPlan=dfintplan_handle,
            )
        )

    dfintplan_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)
    dfintplan_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Create DFIntPlan',
        ce.cuestDFIntPlanCreate(
            handle=cuest_handle,
            primaryBasis=aobasis_handle,
            auxiliaryBasis=auxbasis_handle,
            pairList=aopairlist_handle,
            parameters=dfintplan_parameters,
            persistentWorkspace=dfintplan_persistent_workspace.pointer,
            temporaryWorkspace=dfintplan_temporary_workspace.pointer,
            outPlan=dfintplan_handle,
            )
        )

    del dfintplan_temporary_workspace

    cuest_check('Destroy DFIntPlan Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_DFINTPLAN_PARAMETERS,
            parameters=dfintplan_parameters,
            )
        )

    # => Orbital and TPDM Set Up <= #

    # Get nocc for a neutral molecule. In the parser, Zs are negated charges,
    # so we sum and divide by two for the closed-shell occupied count.
    nocc = int(np.abs(np.sum(Zs) // 2))

    # Define nvir as the remaining orbitals for this example.
    # In practice, this comes from a separate SCF/MO calculation.
    nvir = nao - nocc
    if nvir == 0:
        raise RuntimeError("No virtual orbitals available in this simple example.")

    # Number of MOs, and the sizes of the composite pq, ij and ab indices
    nmo = nocc + nvir
    npq = nmo * nmo
    noo = nocc * nocc
    nvv = nvir * nvir
    natom = len(symbols)

    # Orbital counts for the coefficient matrices.  Case 1 uses all MOs on
    # both sides of a single pair of coefficient matrices, and Case 2 uses a
    # batch of two pairs, [occ, occ] and [vir, vir].
    num_mo = [nmo]
    num_occ_vir = [nocc, nvir]

    # Allocate space for the MO coefficients, with shape [nmo, nao].  The
    # occupied orbitals are the first nocc rows and the virtual orbitals are
    # the last nvir rows, so the same array is [C_occ, C_vir] when
    # concatenated for the batch of two.
    # Fill with random numbers. In real use cases, these are molecular
    # orbital coefficients.
    np.random.seed(0)
    c_host_array = np.random.normal(size=(nmo, nao)).astype(np.double).ravel()

    c_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nmo * nao
        )
    c_device_handle = ce.Pointer(c_device_pointer)
    cuda_memcpy_htod(
        device_pointer=c_device_pointer,
        host_pointer=c_host_array.ctypes.data,
        size_in_bytes=sizeof_double * nmo * nao,
        )

    # The operator weighting of the integral plan is applied by the
    # derivative routine, so the DF MO integrals used to form the
    # three-index TPDMs must not include it.  By default the weighting
    # is included, so it is turned off here.  The DF integral plan was
    # created with CUEST_DFINTPLAN_PARAMETERS_CACHE_OPERATOR_WEIGHTING_NONE
    # so that it can serve this request.
    dfmo_compute_parameters = ce.cuestDFMOIntegralsComputeParameters()
    cuest_check('Create DFMOIntegrals Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS,
            outParameters=dfmo_compute_parameters,
            )
        )

    operator_weighting = ce.data_cuestDFMOIntegralsComputeParametersOperatorWeighting_t(
        ce.CuestDFMOIntegralsComputeParametersOperatorWeighting.CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS_OPERATOR_WEIGHTING_NONE
        )
    cuest_check('Configure DFMOIntegrals Params',
        ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS,
            parameters=dfmo_compute_parameters,
            attribute=ce.CuestDFMOIntegralsComputeParametersAttributes.CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS_OPERATOR_WEIGHTING,
            attributeValue=operator_weighting,
            )
        )

    dfmo_derivative_parameters = ce.cuestDFMOIntegralsDerivativeComputeParameters()
    cuest_check('Create DFMOIntegralsDerivative Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_DFMOINTEGRALSDERIVATIVECOMPUTE_PARAMETERS,
            outParameters=dfmo_derivative_parameters,
            )
        )

    # Allow the algorithm to use up to 2GB of DRAM for the variable buffer.
    variable_buffer_size = WorkspaceDescriptor(
        device_buffer_size_in_bytes=2000000000
        )

    # => Case 1: sum_{pqrs} (pq|rs) Gamma_pq^rs <= #

    # Allocate space for B_pq^A, with shape [naux, nmo * nmo]
    tensors_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * naux * npq
        )
    tensors_device_handle = ce.Pointer(tensors_device_pointer)

    # Compute DF MO integrals without the operator weighting
    cuest_check('Compute DFMOIntegrals Workspace Query',
        ce.cuestDFMOIntegralsComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=dfmo_compute_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numCoefficientMatrices=1,
            numLeftOrbitals=num_mo,
            numRightOrbitals=num_mo,
            leftCoefficientMatrices=c_device_handle,
            rightCoefficientMatrices=c_device_handle,
            outTensors=tensors_device_handle,
            )
        )

    dfmoint_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Compute DFMOIntegrals',
        ce.cuestDFMOIntegralsCompute(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=dfmo_compute_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspace=dfmoint_temporary_workspace.pointer,
            numCoefficientMatrices=1,
            numLeftOrbitals=num_mo,
            numRightOrbitals=num_mo,
            leftCoefficientMatrices=c_device_handle,
            rightCoefficientMatrices=c_device_handle,
            outTensors=tensors_device_handle,
            )
        )

    del dfmoint_temporary_workspace

    # Form the three-index TPDM, Gamma_pq^A, by contracting the DF integral
    # factors with the four-index TPDM, with shape [naux, nmo * nmo].  The
    # contraction is done on the host with numpy for simplicity.
    tensors_host_array = np.empty(
        (naux, npq),
        dtype=np.double
        )
    cuda_memcpy_dtoh(
        host_pointer=tensors_host_array.ctypes.data,
        device_pointer=tensors_device_pointer,
        size_in_bytes=sizeof_double * naux * npq,
        )

    # Fill the four-index TPDM, with shape [pq, rs], with random numbers.
    # In real use cases, this is a bra-ket symmetric two-particle density
    # matrix.  The derivative routine assumes Gamma_pq^rs = Gamma_rs^pq, so
    # the TPDM must be symmetric.
    gamma_host_array = np.random.normal(size=(npq, npq))
    gamma_host_array = 0.5 * (gamma_host_array + gamma_host_array.T)
    gamma3_host_array = np.ascontiguousarray(
        tensors_host_array @ gamma_host_array.T,
        dtype=np.double
        )
    gamma3_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * naux * npq
        )
    gamma3_device_handle = ce.Pointer(gamma3_device_pointer)
    cuda_memcpy_htod(
        device_pointer=gamma3_device_pointer,
        host_pointer=gamma3_host_array.ctypes.data,
        size_in_bytes=sizeof_double * naux * npq,
        )

    # Allocate space for the gradient, with shape [natom, 3]
    grad1_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * natom * 3
        )
    grad1_device_handle = ce.Pointer(grad1_device_pointer)

    # Compute the gradient of the energy with respect to the nuclear coordinates
    cuest_check('Compute DFMOIntegralsDerivative1 Case 1 Workspace Query',
        ce.cuestDFMOIntegralsDerivativeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=dfmo_derivative_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numCoefficientMatrices=1,
            numLeftOrbitals=num_mo,
            numRightOrbitals=num_mo,
            leftCoefficientMatrices=c_device_handle,
            rightCoefficientMatrices=c_device_handle,
            threeCenterTPDMs=gamma3_device_handle,
            outGradient=grad1_device_handle,
            )
        )

    dfmoderiv_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Compute DFMOIntegralsDerivative1 Case 1',
        ce.cuestDFMOIntegralsDerivativeCompute(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=dfmo_derivative_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspace=dfmoderiv_temporary_workspace.pointer,
            numCoefficientMatrices=1,
            numLeftOrbitals=num_mo,
            numRightOrbitals=num_mo,
            leftCoefficientMatrices=c_device_handle,
            rightCoefficientMatrices=c_device_handle,
            threeCenterTPDMs=gamma3_device_handle,
            outGradient=grad1_device_handle,
            )
        )

    del dfmoderiv_temporary_workspace

    # => Case 2: sum_{ijab} (ij|ab) Gamma_ij^ab <= #

    # This term is not bra-ket symmetric, so the three-index TPDMs are
    # built from two terms, each with a scale factor of 1/2,
    #
    #     Gamma_ij^A = 1/2 sum_ab B_ab^A Gamma_ij^ab
    #     Gamma_ab^A = 1/2 sum_ij B_ij^A Gamma_ij^ab
    #
    # and passed as a batch of two.

    # Allocate space for B_ij^A, with shape [naux, noo], followed by
    # B_ab^A, with shape [naux, nvv]
    tensors_ov_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * naux * (noo + nvv)
        )
    tensors_ov_device_handle = ce.Pointer(tensors_ov_device_pointer)

    # Compute DF MO integrals (B_ij^A, B_ab^A) without the operator weighting
    cuest_check('Compute DFMOIntegrals Case 2 Workspace Query',
        ce.cuestDFMOIntegralsComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=dfmo_compute_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numCoefficientMatrices=2,
            numLeftOrbitals=num_occ_vir,
            numRightOrbitals=num_occ_vir,
            leftCoefficientMatrices=c_device_handle,
            rightCoefficientMatrices=c_device_handle,
            outTensors=tensors_ov_device_handle,
            )
        )

    dfmoint_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Compute DFMOIntegrals Case 2',
        ce.cuestDFMOIntegralsCompute(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=dfmo_compute_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspace=dfmoint_temporary_workspace.pointer,
            numCoefficientMatrices=2,
            numLeftOrbitals=num_occ_vir,
            numRightOrbitals=num_occ_vir,
            leftCoefficientMatrices=c_device_handle,
            rightCoefficientMatrices=c_device_handle,
            outTensors=tensors_ov_device_handle,
            )
        )

    del dfmoint_temporary_workspace

    tensors_ov_host_array = np.empty(
        (naux * (noo + nvv),),
        dtype=np.double
        )
    cuda_memcpy_dtoh(
        host_pointer=tensors_ov_host_array.ctypes.data,
        device_pointer=tensors_ov_device_pointer,
        size_in_bytes=sizeof_double * naux * (noo + nvv),
        )
    b_ij_host_array = tensors_ov_host_array[:naux * noo].reshape(naux, noo)
    b_ab_host_array = tensors_ov_host_array[naux * noo:].reshape(naux, nvv)

    # Fill the four-index TPDM, with shape [ij, ab], with random numbers.
    # This is the only block of the TPDM that is needed.
    gamma_ov_host_array = np.random.normal(size=(noo, nvv))

    # Form the three-index TPDMs, Gamma_ij^A with shape [naux, noo],
    # followed by Gamma_ab^A with shape [naux, nvv], including the scale
    # factor of 1/2.
    gamma3_ov_host_array = np.ascontiguousarray(
        np.concatenate((
            (0.5 * b_ab_host_array @ gamma_ov_host_array.T).ravel(),
            (0.5 * b_ij_host_array @ gamma_ov_host_array).ravel(),
            )),
        dtype=np.double
        )
    gamma3_ov_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * naux * (noo + nvv)
        )
    gamma3_ov_device_handle = ce.Pointer(gamma3_ov_device_pointer)
    cuda_memcpy_htod(
        device_pointer=gamma3_ov_device_pointer,
        host_pointer=gamma3_ov_host_array.ctypes.data,
        size_in_bytes=sizeof_double * naux * (noo + nvv),
        )

    # Allocate space for the gradient, with shape [natom, 3]
    grad2_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * natom * 3
        )
    grad2_device_handle = ce.Pointer(grad2_device_pointer)

    # Compute the gradient.  The orbitals [C_occ, C_vir] are used on both
    # the left and the right for the batch of two.
    cuest_check('Compute DFMOIntegralsDerivative2 Case 2 Workspace Query',
        ce.cuestDFMOIntegralsDerivativeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=dfmo_derivative_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numCoefficientMatrices=2,
            numLeftOrbitals=num_occ_vir,
            numRightOrbitals=num_occ_vir,
            leftCoefficientMatrices=c_device_handle,
            rightCoefficientMatrices=c_device_handle,
            threeCenterTPDMs=gamma3_ov_device_handle,
            outGradient=grad2_device_handle,
            )
        )

    dfmoderiv_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Compute DFMOIntegralsDerivative2 Case 2',
        ce.cuestDFMOIntegralsDerivativeCompute(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=dfmo_derivative_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspace=dfmoderiv_temporary_workspace.pointer,
            numCoefficientMatrices=2,
            numLeftOrbitals=num_occ_vir,
            numRightOrbitals=num_occ_vir,
            leftCoefficientMatrices=c_device_handle,
            rightCoefficientMatrices=c_device_handle,
            threeCenterTPDMs=gamma3_ov_device_handle,
            outGradient=grad2_device_handle,
            )
        )

    del dfmoderiv_temporary_workspace

    cuest_check('Destroy DFMOIntegrals Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS,
            parameters=dfmo_compute_parameters,
            )
        )

    cuest_check('Destroy DFMOIntegralsDerivative Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_DFMOINTEGRALSDERIVATIVECOMPUTE_PARAMETERS,
            parameters=dfmo_derivative_parameters,
            )
        )

    # Free MO coefficient, TPDM, DF MO tensor and gradient buffers
    cuda_free(
        array=c_device_pointer
        )
    cuda_free(
        array=tensors_device_pointer
        )
    cuda_free(
        array=gamma3_device_pointer
        )
    cuda_free(
        array=grad1_device_pointer
        )
    cuda_free(
        array=tensors_ov_device_pointer
        )
    cuda_free(
        array=gamma3_ov_device_pointer
        )
    cuda_free(
        array=grad2_device_pointer
        )

    # => Cleanup <= #

    cuest_check('Destroy DFIntPlan',
        ce.cuestDFIntPlanDestroy(
            handle=dfintplan_handle,
            )
        )

    cuest_check('Destroy AOPairList',
        ce.cuestAOPairListDestroy(
            handle=aopairlist_handle,
            )
        )

    cuest_check('Destroy Aux AOBasis',
        ce.cuestAOBasisDestroy(
            handle=auxbasis_handle,
            )
        )

    cuest_check('Destroy AOBasis',
        ce.cuestAOBasisDestroy(
            handle=aobasis_handle,
            )
        )

    del dfintplan_persistent_workspace
    del aopairlist_persistent_workspace
    del auxbasis_persistent_workspace
    del aobasis_persistent_workspace

    cuest_check('Destroy Cuest Handle',
        ce.cuestDestroy(
            handle=cuest_handle,
            )
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run using XYZ and GBS files."
        )
    parser.add_argument(
        "xyz_filename",
        type=str,
        help="The file with molecular coordinates (Angstrom) in basic xyz format."
    )
    parser.add_argument(
        "gbs_filename",
        type=str,
        help="The basis set file in G94/Psi4 GBS format."
    )
    parser.add_argument(
        "aux_gbs_filename",
        type=str,
        help="The auxiliary basis set file in G94/Psi4 GBS format."
    )

    args = parser.parse_args()

    run(
        xyz_filename=args.xyz_filename,
        gbs_filename=args.gbs_filename,
        aux_gbs_filename=args.aux_gbs_filename,
        )
