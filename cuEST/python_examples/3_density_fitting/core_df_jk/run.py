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

    # These are user-provided functions that are responsible for allocating
    # host and device workspace arrays.  We use 
    persistent_workspace_descriptor = WorkspaceDescriptor()
    temporary_workspace_descriptor = WorkspaceDescriptor()

    cuest_handle_parameters = ce.cuestHandleParameters()

    cuest_check('Create Handle Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
            outParameters=cuest_handle_parameters,
            )
        )

    # Here we allow cuEST to use the default stream, cuBLAS, etc., but those
    # parameters should be set in the handle parameters before this call.
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

    # Auxilliary Basis
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

    # Auxilliary Basis
    auxbasis_handle = ce.cuestAOBasisHandle()
    cuest_check('Create AOBasis Workspace Query',
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

    cuest_check('Create AOBasis',
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

    # => Build Fake Density Matrix / Occupied Orbitals <= #

    densitymatrix_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nao**2
        )
    densitymatrix_device_handle = ce.Pointer()
    densitymatrix_device_handle.value = densitymatrix_device_pointer
    densitymatrix_host_array = np.random.normal(size=(nao, nao)).astype(np.double)
    densitymatrix_host_array = np.ravel(densitymatrix_host_array + densitymatrix_host_array.T)
    cuda_memcpy_htod(
        device_pointer=densitymatrix_device_pointer,
        host_pointer=densitymatrix_host_array.ctypes.data,
        size_in_bytes=sizeof_double * nao**2,
        )

    nocc = int(np.abs(np.sum(Zs)//2))
    occorbitals_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nocc * nao
        )
    occorbitals_device_handle = ce.Pointer()
    occorbitals_device_handle.value = occorbitals_device_pointer
    occorbitals_host_array = np.random.normal(size=nocc*nao).astype(np.double)
    cuda_memcpy_htod(
        device_pointer=occorbitals_device_pointer,
        host_pointer=occorbitals_host_array.ctypes.data,
        size_in_bytes=sizeof_double * nocc * nao,
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

    print("DFIntPlan Persistent sizes:", persistent_workspace_descriptor)
    print("DFIntPlan Temporary sizes:", temporary_workspace_descriptor)

    dfintplan_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)
    dfintplan_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)

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

    # => Coulomb Matrix <= #

    coulombmatrix_device_handle = ce.Pointer()
    compute_coulomb_matrix_parameters = ce.cuestDFCoulombComputeParameters()
    cuest_check('Create Coulomb Compute Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_DFCOULOMBCOMPUTE_PARAMETERS,
            outParameters=compute_coulomb_matrix_parameters,
            )
        )

    cuest_check('Compute Coulomb Compute Workspace Query',
        ce.cuestDFCoulombComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=compute_coulomb_matrix_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityMatrix=densitymatrix_device_handle,
            outCoulombMatrix=coulombmatrix_device_handle,
            )
        )

    print("CoulombInt Temporary sizes:", temporary_workspace_descriptor)

    coulombint_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    coulombmatrix_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nao**2
        )
    coulombmatrix_device_handle.value = np.intp(coulombmatrix_device_pointer)
    cuest_check('Compute Coulomb Compute',
        ce.cuestDFCoulombCompute(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=compute_coulomb_matrix_parameters,
            temporaryWorkspace=coulombint_temporary_workspace.pointer,
            densityMatrix=densitymatrix_device_handle,
            outCoulombMatrix=coulombmatrix_device_handle,
            )
        )

    del coulombint_temporary_workspace

    cuest_check('Destroy Coulomb Compute Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_DFCOULOMBCOMPUTE_PARAMETERS,
            parameters=compute_coulomb_matrix_parameters,
            )
        )

    coulombmatrix_host_array = np.empty(
        (nao * nao),
        dtype=np.double
        )

    cuda_memcpy_dtoh(
        host_pointer=coulombmatrix_host_array.ctypes.data,
        device_pointer=coulombmatrix_device_pointer,
        size_in_bytes=sizeof_double * nao**2,
        )

    cuda_free(
        array=coulombmatrix_device_pointer
        )

    # => Exchange Matrix <= #

    # Allow the algorithm to use up to 2GB of DRAM
    variable_buffer_size = WorkspaceDescriptor(
        device_buffer_size_in_bytes=2000000000
        )
    compute_exchange_matrix_parameters = ce.cuestDFSymmetricExchangeComputeParameters()
    cuest_check('Create Exchange Compute Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
            outParameters=compute_exchange_matrix_parameters,
            )
        )
    exchangematrix_device_handle = ce.Pointer()

    cuest_check('Compute Exchange Compute Workspace Query',
        ce.cuestDFSymmetricExchangeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=compute_exchange_matrix_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numOccupied=nocc,
            coefficientMatrix=occorbitals_device_handle,
            outExchangeMatrix=exchangematrix_device_handle,
            )
        )

    print("ExchangeInt Temporary sizes:", temporary_workspace_descriptor)

    exchangeint_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    exchangematrix_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nao**2
        )
    exchangematrix_device_handle.value = np.intp(exchangematrix_device_pointer)
    cuest_check('Compute Exchange Compute',
        ce.cuestDFSymmetricExchangeCompute(
            handle=cuest_handle,
            plan=dfintplan_handle,
            parameters=compute_exchange_matrix_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspace=exchangeint_temporary_workspace.pointer,
            numOccupied=nocc,
            coefficientMatrix=occorbitals_device_handle,
            outExchangeMatrix=exchangematrix_device_handle,
            )
        )

    del exchangeint_temporary_workspace

    cuest_check('Destroy Exchange Compute Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
            parameters=compute_exchange_matrix_parameters,
            )
        )

    exchangematrix_host_array = np.empty(
        (nao * nao),
        dtype=np.double
        )

    cuda_memcpy_dtoh(
        host_pointer=exchangematrix_host_array.ctypes.data,
        device_pointer=exchangematrix_device_pointer,
        size_in_bytes=sizeof_double * nao**2,
        )

    cuda_free(
        array=exchangematrix_device_pointer
        )

    # => Cleanup <= #

    cuda_free(
        array=densitymatrix_device_pointer,
        )

    cuda_free(
        array=occorbitals_device_pointer,
        )

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

    cuest_check('Destroy AOBasis',
        ce.cuestAOBasisDestroy(
            handle=aobasis_handle,
            )
        )

    cuest_check('Destroy Aux AOBasis',
        ce.cuestAOBasisDestroy(
            handle=auxbasis_handle,
            )
        )

    del dfintplan_persistent_workspace
    del aopairlist_persistent_workspace
    del aobasis_persistent_workspace
    del auxbasis_persistent_workspace

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
        help="The auxilliary basis set file in G94/Psi4 GBS format."
    )

    args = parser.parse_args()

    run(
        xyz_filename=args.xyz_filename,
        gbs_filename=args.gbs_filename,
        aux_gbs_filename=args.aux_gbs_filename,
        )

