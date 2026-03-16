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

# Inject the directory where the example helper utilities live
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helpers.cuda_utils import cuda_malloc, cuda_free, cuda_memcpy_dtoh, cuda_memcpy_htod, WorkspaceDescriptor, Workspace
from helpers.utilities import normalize_shell_coefficients
from helpers.parsers import simple_xyz_parser, simple_gbs_parser, simple_ecp_parser

def run(
    *,
    xyz_filename,
    gbs_filename,
    ecp_filename,
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

    ecp_metadata = simple_ecp_parser(
        filename=ecp_filename,
        symbols=symbols,
        )

    num_atoms = len(symbols)
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
    # host and device workspace arrays.
    persistent_workspace_descriptor = WorkspaceDescriptor()
    temporary_workspace_descriptor = WorkspaceDescriptor()
    # Allow 2 GB of DRAM for ECP evaluation routines
    variable_buffer_size = WorkspaceDescriptor(
        host_buffer_size_in_bytes=0,
        device_buffer_size_in_bytes=2000000000,
        )

    # => cuEST Handle Setup <= #

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

    # => Build Gaussian Shells From Basis Parser Info <= #

    aoshell_parameters = ce.cuestAOShellParameters()
    cuest_check('Create AO Shell Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_AOSHELL_PARAMETERS,
            outParameters=aoshell_parameters,
            )
        )

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

    cuest_check('Destroy AO Shell Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_AOSHELL_PARAMETERS,
            parameters=aoshell_parameters,
            )
        )

    # => Build Basis Set From Gaussian Shells <= #

    aobasis_parameters = ce.cuestAOBasisParameters()

    cuest_check('Create AO Basis Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_AOBASIS_PARAMETERS,
            outParameters=aobasis_parameters,
            )
        )

    aobasis_handle = ce.cuestAOBasisHandle()

    cuest_check('Create AOBasis Workspace Query',
        ce.cuestAOBasisCreateWorkspaceQuery(
            handle=cuest_handle,
            numAtoms=num_atoms,
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
            numAtoms=num_atoms,
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

    # => Build ECP Shells and Atoms <= #

    # Create ECP shell parameters (shared for all ECP shell creation calls)
    ecpshell_parameters = ce.cuestECPShellParameters()
    cuest_check('Create ECP Shell Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_ECPSHELL_PARAMETERS,
            outParameters=ecpshell_parameters,
            )
        )

    # Create ECP shells once per unique element type.  When multiple atoms of the
    # same element are present, their ECP data is identical so the shell handles
    # are shared across all atoms of that element.
    unique_symbol_shells = {}  # symbol -> {'top_shell': handle, 'shells': [handles]}
    for symbol, info in zip(symbols, ecp_metadata):
        if info is None or symbol in unique_symbol_shells:
            continue

        # Create the top shell (the max-L projector)
        top_shell_info = info['top_shell']
        top_shell_handle = ce.cuestECPShellHandle()
        cuest_check(f'Create ECP Top Shell for {symbol}',
            ce.cuestECPShellCreate(
                handle=cuest_handle,
                L=top_shell_info.L,
                numPrimitive=len(top_shell_info.ns),
                radialPowers=top_shell_info.ns,
                coefficients=top_shell_info.ws,
                exponents=top_shell_info.es,
                parameters=ecpshell_parameters,
                outECPShell=top_shell_handle,
                )
            )

        # Create the angular-momentum projected shells
        shell_handles = []
        for k, shell_info in enumerate(info['shells']):
            shell_handle = ce.cuestECPShellHandle()
            cuest_check(f'Create ECP Shell {k} for {symbol}',
                ce.cuestECPShellCreate(
                    handle=cuest_handle,
                    L=shell_info.L,
                    numPrimitive=len(shell_info.ns),
                    radialPowers=shell_info.ns,
                    coefficients=shell_info.ws,
                    exponents=shell_info.es,
                    parameters=ecpshell_parameters,
                    outECPShell=shell_handle,
                    )
                )
            shell_handles.append(shell_handle)

        unique_symbol_shells[symbol] = {
            'top_shell': top_shell_handle,
            'shells': shell_handles,
            }

    cuest_check('Destroy ECP Shell Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_ECPSHELL_PARAMETERS,
            parameters=ecpshell_parameters,
            )
        )

    # Create ECP atom parameters (shared for all ECP atom creation calls)
    ecpatom_parameters = ce.cuestECPAtomParameters()
    cuest_check('Create ECP Atom Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_ECPATOM_PARAMETERS,
            outParameters=ecpatom_parameters,
            )
        )

    # Create one ECP atom handle per active atom (atoms that carry an ECP)
    ecp_atom_indices = []   # indices into the full atom list
    ecp_atom_handles = []   # corresponding cuestECPAtomHandle objects
    for atom_index, (symbol, info) in enumerate(zip(symbols, ecp_metadata)):
        if info is None:
            continue
        shell_data = unique_symbol_shells[symbol]
        ecpatom_handle = ce.cuestECPAtomHandle()
        cuest_check(f'Create ECP Atom {atom_index}',
            ce.cuestECPAtomCreate(
                handle=cuest_handle,
                numElectrons=info['nelectron'],
                numShells=len(shell_data['shells']),
                shells=shell_data['shells'],
                topShell=shell_data['top_shell'],
                parameters=ecpatom_parameters,
                outECPAtom=ecpatom_handle,
                )
            )
        ecp_atom_indices.append(atom_index)
        ecp_atom_handles.append(ecpatom_handle)

    # ECP shell handles are no longer needed once the atoms have been created
    for symbol, shell_data in unique_symbol_shells.items():
        cuest_check(f'Destroy ECP Top Shell for {symbol}',
            ce.cuestECPShellDestroy(
                handle=shell_data['top_shell'],
                )
            )
        for k, shell_handle in enumerate(shell_data['shells']):
            cuest_check(f'Destroy ECP Shell {k} for {symbol}',
                ce.cuestECPShellDestroy(
                    handle=shell_handle,
                    )
                )

    cuest_check('Destroy ECP Atom Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_ECPATOM_PARAMETERS,
            parameters=ecpatom_parameters,
            )
        )

    # => Build ECP Integral Plan <= #

    ecpintplan_parameters = ce.cuestECPIntPlanParameters()
    cuest_check('Create ECPIntPlan Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_ECPINTPLAN_PARAMETERS,
            outParameters=ecpintplan_parameters,
            )
        )

    ecpintplan_handle = ce.cuestECPIntPlanHandle()

    cuest_check('Create ECPIntPlan Workspace Query',
        ce.cuestECPIntPlanCreateWorkspaceQuery(
            handle=cuest_handle,
            basis=aobasis_handle,
            xyz=xyzs,
            numECPAtoms=len(ecp_atom_indices),
            activeIndices=ecp_atom_indices,
            activeAtoms=ecp_atom_handles,
            parameters=ecpintplan_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPlan=ecpintplan_handle,
            )
        )

    ecpintplan_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)
    ecpintplan_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Create ECPIntPlan',
        ce.cuestECPIntPlanCreate(
            handle=cuest_handle,
            basis=aobasis_handle,
            xyz=xyzs,
            numECPAtoms=len(ecp_atom_indices),
            activeIndices=ecp_atom_indices,
            activeAtoms=ecp_atom_handles,
            parameters=ecpintplan_parameters,
            persistentWorkspace=ecpintplan_persistent_workspace.pointer,
            temporaryWorkspace=ecpintplan_temporary_workspace.pointer,
            outPlan=ecpintplan_handle,
            )
        )

    del ecpintplan_temporary_workspace

    for i, ecpatom_handle in enumerate(ecp_atom_handles):
        cuest_check(f'Destroy ECP Atom {i}',
            ce.cuestECPAtomDestroy(
                handle=ecpatom_handle,
                )
            )

    cuest_check('Destroy ECPIntPlan Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_ECPINTPLAN_PARAMETERS,
            parameters=ecpintplan_parameters,
            )
        )

    # => Compute ECP Gradients <= #

    # The ECP derivative routines contract the integral derivatives against a density
    # matrix and accumulate the result into a [num_atoms, 3] gradient array.  A real
    # calculation would supply the SCF density matrix here; we use a random symmetric
    # matrix as a stand-in.
    np.random.seed(0)
    density_host = np.random.normal(size=(nao, nao)).astype(np.double)
    density_host = 0.5 * (density_host + density_host.T)  # Symmetrize

    density_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nao**2
        )
    density_device_handle = ce.Pointer(density_device_pointer)
    cuda_memcpy_htod(
        device_pointer=density_device_pointer,
        host_pointer=density_host.ctypes.data,
        size_in_bytes=sizeof_double * nao**2,
        )

    gradient_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * num_atoms * 3
        )
    gradient_device_handle = ce.Pointer(gradient_device_pointer)

    ecpderivcompute_parameters = ce.cuestECPDerivativeComputeParameters()
    cuest_check('Create ECPDerivativeCompute Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_ECPDERIVATIVECOMPUTE_PARAMETERS,
            outParameters=ecpderivcompute_parameters,
            )
        )

    # Find the temporary workspace requirements for the ECP gradient evaluation
    cuest_check('ECPDerivativeCompute Workspace Query',
        ce.cuestECPDerivativeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=ecpintplan_handle,
            parameters=ecpderivcompute_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityMatrix=density_device_handle,
            outGradient=gradient_device_handle,
            )
        )

    ecp_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    # Compute dECP/dR contracted with the density matrix; result is [num_atoms, 3]
    cuest_check('ECPDerivativeCompute',
        ce.cuestECPDerivativeCompute(
            handle=cuest_handle,
            plan=ecpintplan_handle,
            parameters=ecpderivcompute_parameters,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspace=ecp_temporary_workspace.pointer,
            densityMatrix=density_device_handle,
            outGradient=gradient_device_handle,
            )
        )

    del ecp_temporary_workspace

    cuest_check('Destroy ECPDerivativeCompute Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_ECPDERIVATIVECOMPUTE_PARAMETERS,
            parameters=ecpderivcompute_parameters,
            )
        )

    gradient_host_array = np.empty(
        num_atoms * 3,
        dtype=np.double
        )

    cuda_memcpy_dtoh(
        host_pointer=gradient_host_array.ctypes.data,
        device_pointer=gradient_device_pointer,
        size_in_bytes=sizeof_double * num_atoms * 3,
        )

    cuda_free(array=density_device_pointer)
    cuda_free(array=gradient_device_pointer)

    # => Cleanup <= #

    cuest_check('Destroy ECPIntPlan',
        ce.cuestECPIntPlanDestroy(
            handle=ecpintplan_handle,
            )
        )
    del ecpintplan_persistent_workspace

    cuest_check('Destroy AOBasis',
        ce.cuestAOBasisDestroy(
            handle=aobasis_handle,
            )
        )
    del aobasis_persistent_workspace

    cuest_check('Destroy Cuest Handle',
        ce.cuestDestroy(
            handle=cuest_handle,
            )
        )
        
    return gradient_host_array.reshape(num_atoms, 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute ECP integral gradients contracted with a density matrix."
        )
    parser.add_argument(
        "xyz_filename",
        type=str,
        help="Molecular coordinates (Angstrom) in basic XYZ format."
        )
    parser.add_argument(
        "gbs_filename",
        type=str,
        help="Orbital basis set in G94/Psi4 GBS format."
        )
    parser.add_argument(
        "ecp_filename",
        type=str,
        help="Effective core potential basis set file."
        )

    args = parser.parse_args()

    run(
        xyz_filename=args.xyz_filename,
        gbs_filename=args.gbs_filename,
        ecp_filename=args.ecp_filename,
        )
