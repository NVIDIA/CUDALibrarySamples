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
from helpers.parsers import simple_xyz_parser, simple_gbs_parser
from helpers.grid_utils import build_ahlrichs_radial_quadrature, symbol_to_ahlrichs_radius

def run(
    *,
    xyz_filename,
    gbs_filename,
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
    # Allow 2Gb of DRAM for XC evaluation routines
    variable_buffer_size = WorkspaceDescriptor(
        host_buffer_size_in_bytes=0,
        device_buffer_size_in_bytes=2000000000,
        )

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

    stream_handle = ce.data_cudaStream_t()
    cuest_check('Query Handle Stream',
        ce.cuestParametersQuery(
            parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
            parameters=cuest_handle_parameters,
            attribute=ce.CuestHandleParametersAttributes.CUEST_HANDLE_PARAMETERS_CUDASTREAM,
            attributeValue=stream_handle,
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


    # => Build Molecular Integration Grid <= #

    atomgrid_parameters = ce.cuestAtomGridParameters()

    cuest_check('Create AtomGrid Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_ATOMGRID_PARAMETERS,
            outParameters=atomgrid_parameters,
            )
        )

    atomgrids = []
    for atom,symbol in enumerate(symbols):
        # For simplicity, build an unpruned 75/302 grid using the Ahlrichs radial quadrature scheme
        ahlrichs_radius = symbol_to_ahlrichs_radius(
            symbol=symbol
            )
        radial_nodes, radial_weights = build_ahlrichs_radial_quadrature(
            npoint=75,
            R=ahlrichs_radius,
            )
        num_angular_points = [302]*len(radial_nodes)
        atomgrid_handle = ce.cuestAtomGridHandle()
        cuest_check(f'Create AtomGrid {atom}',
            ce.cuestAtomGridCreate(
                handle=cuest_handle,
                numRadialPoints=75,
                radialNodes=radial_nodes,
                radialWeights=radial_weights,
                numAngularPoints=num_angular_points,
                parameters=atomgrid_parameters,
                outAtomGrid=atomgrid_handle,
                )
            )
        atomgrids.append(atomgrid_handle)

    cuest_check('Destroy AtomGrid Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_ATOMGRID_PARAMETERS,
            parameters=atomgrid_parameters,
            )
        )

    moleculargrid_parameters = ce.cuestMolecularGridParameters()

    cuest_check('Create MolecularGrid Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_MOLECULARGRID_PARAMETERS,
            outParameters=moleculargrid_parameters,
            )
        )

    cuest_check('Create MolecularGrid Workspace Query',
        ce.cuestMolecularGridCreateWorkspaceQuery(
            handle=cuest_handle,
            numAtoms=len(Zs),
            atomGrid=atomgrids,
            xyz=xyzs,
            parameters=moleculargrid_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outGrid=ce.cuestMolecularGridHandle(),
            )
        )

    grid_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)
    grid_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    moleculargrid=ce.cuestMolecularGridHandle()

    cuest_check('Create MolecularGrid',
        ce.cuestMolecularGridCreate(
            handle=cuest_handle,
            numAtoms=len(Zs),
            atomGrid=atomgrids,
            xyz=xyzs,
            parameters=moleculargrid_parameters,
            persistentWorkspace=grid_persistent_workspace.pointer,
            temporaryWorkspace=grid_temporary_workspace.pointer,
            outGrid=moleculargrid,
            )
        )

    del grid_temporary_workspace

    for atom,grid in enumerate(atomgrids):
        cuest_check(f'Destroy AtomGrid{atom}',
            ce.cuestAtomGridDestroy(
                atomGrid=grid,
                )
            )

    cuest_check('Destroy MolecularGrid Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_MOLECULARGRID_PARAMETERS,
            parameters=moleculargrid_parameters,
            )
        )

    # => Build XC Integral Plan <= #

    xcintplan_parameters = ce.cuestXCIntPlanParameters()
    cuest_check('Create XCIntPlan Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_XCINTPLAN_PARAMETERS,
            outParameters=xcintplan_parameters,
            )
        )

    cuest_check('Create XCIntPlan Workspace Query',
        ce.cuestXCIntPlanCreateWorkspaceQuery(
            handle=cuest_handle,
            basis=aobasis_handle,
            grid=moleculargrid,
            functional=ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE,
            parameters=xcintplan_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPlan=ce.cuestXCIntPlanHandle(),
            )
        )

    xcintplan = ce.cuestXCIntPlanHandle()
    xcintplan_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)
    xcintplan_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Create XCIntPlan',
        ce.cuestXCIntPlanCreate(
            handle=cuest_handle,
            basis=aobasis_handle,
            grid=moleculargrid,
            functional=ce.CuestXCIntPlanParametersFunctional.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE,
            parameters=xcintplan_parameters,
            persistentWorkspace=xcintplan_persistent_workspace.pointer,
            temporaryWorkspace=xcintplan_temporary_workspace.pointer,
            outPlan=xcintplan,
            )
        )

    del xcintplan_temporary_workspace

    cuest_check('Destroy XCIntPlan Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_XCINTPLAN_PARAMETERS,
            parameters=xcintplan_parameters,
            )
        )

    # => RKS Potential Matrix Computation <= #

    # => Build Fake Occupied Orbitals <= #

    nocc = np.abs(int(np.sum(Zs)//2))

    Cocc_host = np.random.normal(size=(nocc, nao)).astype(np.double)
    Cocc_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nocc * nao
        )
    Cocc_device_handle = ce.Pointer(Cocc_device_pointer)
    cuda_memcpy_htod(
        device_pointer=Cocc_device_pointer,
        host_pointer=Cocc_host.ctypes.data,
        size_in_bytes=sizeof_double * nocc * nao,
        stream=stream_handle.value,
        )

    # => Call the XC Potential Compute Routines <= #

    Vxc_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nao**2
        )
    Vxc_device_handle = ce.Pointer(Vxc_device_pointer)
    compute_xc_potential_parameters = ce.cuestXCPotentialRKSComputeParameters()
    cuest_check('XCPotentialRKS Parameters Create',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_XCPOTENTIALRKSCOMPUTE_PARAMETERS,
            outParameters=compute_xc_potential_parameters,
            )
        )

    Exc = ce.data_double()
    cuest_check('XCPotentialRKSCompute Workspace Query',
        ce.cuestXCPotentialRKSComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=xcintplan,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            parameters=compute_xc_potential_parameters,
            numOccupied=nocc,
            coefficientMatrix=Cocc_device_handle,
            outXCEnergy=Exc,
            outXCPotentialMatrix=Vxc_device_handle,
            )
        )

    Vxc_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('XCPotentialRKSCompute',
        ce.cuestXCPotentialRKSCompute(
            handle=cuest_handle,
            plan=xcintplan,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspace=Vxc_workspace.pointer,
            parameters=compute_xc_potential_parameters,
            numOccupied=nocc,
            coefficientMatrix=Cocc_device_handle,
            outXCEnergy=Exc,
            outXCPotentialMatrix=Vxc_device_handle,
            )
        )

    del Vxc_workspace

    cuda_free(
        array=Cocc_device_pointer
        )

    cuest_check('Destroy XCPotentialRKS Parameters',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_XCPOTENTIALRKSCOMPUTE_PARAMETERS,
            parameters=compute_xc_potential_parameters,
            )
        )

    Vxc_host_array = np.empty(
        (nao * nao),
        dtype=np.double
        )
    cuda_memcpy_dtoh(
        host_pointer=Vxc_host_array.ctypes.data,
        device_pointer=Vxc_device_pointer,
        size_in_bytes=sizeof_double * nao**2,
        stream=stream_handle.value,
        )

    cuda_free(
        array=Vxc_device_pointer
        )

    # => UKS Potential Matrix Computation <= #

    # => Build Fake Occupied Orbitals <= #

    # Assume a cation
    nocca = np.abs(int(np.sum(Zs)//2))
    noccb = nocca - 1

    # Alpha and Beta occupied molecular orbitals
    Cocca_host = np.random.normal(size=(nocca, nao)).astype(np.double)
    Cocca_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nocca * nao
        )
    Cocca_device_handle = ce.Pointer(Cocca_device_pointer)
    cuda_memcpy_htod(
        device_pointer=Cocca_device_pointer,
        host_pointer=Cocca_host.ctypes.data,
        size_in_bytes=sizeof_double * nocca * nao,
        stream=stream_handle.value,
        )

    Coccb_host = np.random.normal(size=(noccb, nao)).astype(np.double)
    Coccb_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * noccb * nao
        )
    Coccb_device_handle = ce.Pointer(Coccb_device_pointer)
    cuda_memcpy_htod(
        device_pointer=Coccb_device_pointer,
        host_pointer=Coccb_host.ctypes.data,
        size_in_bytes=sizeof_double * noccb * nao,
        stream=stream_handle.value,
        )

    # => Call the XC Potential Compute Routines <= #

    Vxca_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nao**2
        )
    Vxca_device_handle = ce.Pointer(Vxca_device_pointer)

    Vxcb_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nao**2
        )
    Vxcb_device_handle = ce.Pointer(Vxcb_device_pointer)

    compute_xc_potential_parameters = ce.cuestXCPotentialUKSComputeParameters()
    cuest_check('XCPotentialUKS Parameters Create',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_XCPOTENTIALUKSCOMPUTE_PARAMETERS,
            outParameters=compute_xc_potential_parameters,
            )
        )

    Exc = ce.data_double()
    cuest_check('XCPotentialUKSCompute Workspace Query',
        ce.cuestXCPotentialUKSComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=xcintplan,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            parameters=compute_xc_potential_parameters,
            numOccupiedAlpha=nocca,
            numOccupiedBeta=noccb,
            coefficientMatrixAlpha=Cocca_device_handle,
            coefficientMatrixBeta=Coccb_device_handle,
            outXCEnergy=Exc,
            outXCPotentialMatrixAlpha=Vxca_device_handle,
            outXCPotentialMatrixBeta=Vxcb_device_handle,
            )
        )

    Vxc_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('XCPotentialUKSCompute',
        ce.cuestXCPotentialUKSCompute(
            handle=cuest_handle,
            plan=xcintplan,
            variableBufferSize=variable_buffer_size.pointer,
            temporaryWorkspace=Vxc_workspace.pointer,
            parameters=compute_xc_potential_parameters,
            numOccupiedAlpha=nocca,
            numOccupiedBeta=noccb,
            coefficientMatrixAlpha=Cocca_device_handle,
            coefficientMatrixBeta=Coccb_device_handle,
            outXCEnergy=Exc,
            outXCPotentialMatrixAlpha=Vxca_device_handle,
            outXCPotentialMatrixBeta=Vxcb_device_handle,
            )
        )

    del Vxc_workspace

    cuest_check('Destroy XCPotentialUKS Parameters',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_XCPOTENTIALUKSCOMPUTE_PARAMETERS,
            parameters=compute_xc_potential_parameters,
            )
        )

    cuda_free(
        array=Cocca_device_pointer
        )
    cuda_free(
        array=Coccb_device_pointer
        )

    Vxca_host_array = np.empty(
        (nao * nao),
        dtype=np.double
        )
    cuda_memcpy_dtoh(
        host_pointer=Vxca_host_array.ctypes.data,
        device_pointer=Vxca_device_pointer,
        size_in_bytes=sizeof_double * nao**2,
        stream=stream_handle.value,
        )

    Vxcb_host_array = np.empty(
        (nao * nao),
        dtype=np.double
        )
    cuda_memcpy_dtoh(
        host_pointer=Vxcb_host_array.ctypes.data,
        device_pointer=Vxcb_device_pointer,
        size_in_bytes=sizeof_double * nao**2,
        stream=stream_handle.value,
        )

    cuda_free(
        array=Vxca_device_pointer
        )
    cuda_free(
        array=Vxcb_device_pointer
        )

    # => Cleanup <= #

    cuest_check('Destroy XCIntPlan',
        ce.cuestXCIntPlanDestroy(
            handle=xcintplan,
            )
        )
    del xcintplan_persistent_workspace

    cuest_check('Destroy MolecularGrid',
        ce.cuestMolecularGridDestroy(
            grid=moleculargrid,
            )
        )
    del grid_persistent_workspace


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

    args = parser.parse_args()

    run(
        xyz_filename=args.xyz_filename,
        gbs_filename=args.gbs_filename,
        )

