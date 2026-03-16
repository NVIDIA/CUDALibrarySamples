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

    # => Build One Electron Integral Plan <= #

    oeintplan_handle = ce.cuestOEIntPlanHandle()
    oeintplan_parameters = ce.cuestOEIntPlanParameters()

    cuest_check('Create OEIntPlan Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_OEINTPLAN_PARAMETERS,
            outParameters=oeintplan_parameters,
            )
        )

    cuest_check('Create OEIntPlan Workspace Query',
        ce.cuestOEIntPlanCreateWorkspaceQuery(
            handle=cuest_handle,
            basis=aobasis_handle,
            pairList=aopairlist_handle,
            parameters=oeintplan_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPlan=oeintplan_handle,
            )
        )

    oeintplan_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)
    oeintplan_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Create OEIntPlan',
        ce.cuestOEIntPlanCreate(
            handle=cuest_handle,
            basis=aobasis_handle,
            pairList=aopairlist_handle,
            parameters=oeintplan_parameters,
            persistentWorkspace=oeintplan_persistent_workspace.pointer,
            temporaryWorkspace=oeintplan_temporary_workspace.pointer,
            outPlan=oeintplan_handle,
            )
        )

    del oeintplan_temporary_workspace

    cuest_check('Destroy OEIntPlan Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_OEINTPLAN_PARAMETERS,
            parameters=oeintplan_parameters,
            )
        )

    # => Overlap Integrals <= #

    overlapint_device_handle = ce.Pointer()
    compute_overlap_parameters = ce.cuestOverlapComputeParameters()
    cuest_check('Create Overlap Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_OVERLAPCOMPUTE_PARAMETERS,
            outParameters=compute_overlap_parameters,
            )
        )

    # Find Memory Requirements
    cuest_check('Compute Overlap Ints Workspace Query',
        ce.cuestOverlapComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_overlap_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outSMatrix=overlapint_device_handle,
            )
        )

    overlapint_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    overlapint_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nao**2
        )
    overlapint_device_handle.value = np.intp(overlapint_device_pointer)

    # Compute Integrals
    cuest_check('Compute Overlap Ints',
        ce.cuestOverlapCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_overlap_parameters,
            temporaryWorkspace=overlapint_temporary_workspace.pointer,
            outSMatrix=overlapint_device_handle,
            )
        )

    del overlapint_temporary_workspace

    cuest_check('Destroy Overlap Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_OVERLAPCOMPUTE_PARAMETERS,
            parameters=compute_overlap_parameters,
            )
        )

    overlapint_host_array = np.empty(
        (nao * nao),
        dtype=np.double
        )

    cuda_memcpy_dtoh(
        host_pointer=overlapint_host_array.ctypes.data,
        device_pointer=overlapint_device_pointer,
        size_in_bytes=sizeof_double * nao**2,
        )

    cuda_free(
        array=overlapint_device_pointer
        )

    # => Kinetic Integrals <= #

    kineticint_device_handle = ce.Pointer()
    compute_kinetic_parameters = ce.cuestKineticComputeParameters()
    cuest_check('Create Kinetic Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_KINETICCOMPUTE_PARAMETERS,
            outParameters=compute_kinetic_parameters,
            )
        )

    # Find Memory Requirements
    cuest_check('Compute Kinetic Ints Workspace Query',
        ce.cuestKineticComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_kinetic_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outTMatrix=kineticint_device_handle,
            )
        )

    kineticint_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    kineticint_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nao**2
        )
    kineticint_device_handle.value = np.intp(kineticint_device_pointer)

    # Compute Integrals
    cuest_check('Compute Kinetic Ints',
        ce.cuestKineticCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_kinetic_parameters,
            temporaryWorkspace=kineticint_temporary_workspace.pointer,
            outTMatrix=kineticint_device_handle,
            )
        )

    del kineticint_temporary_workspace

    cuest_check('Destroy Kinetic Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_KINETICCOMPUTE_PARAMETERS,
            parameters=compute_kinetic_parameters,
            )
        )

    kineticint_host_array = np.empty(
        (nao * nao),
        dtype=np.double
        )

    cuda_memcpy_dtoh(
        host_pointer=kineticint_host_array.ctypes.data,
        device_pointer=kineticint_device_pointer,
        size_in_bytes=sizeof_double * nao**2,
        )

    cuda_free(
        array=kineticint_device_pointer
        )

    # => Potential Integrals <= #

    potentialint_device_handle = ce.Pointer()

    # Move the position and (negated) charges of the atoms to the GPU
    xyzs_device_handle = ce.Pointer()
    xyzs = np.array(xyzs)
    xyzs_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * len(xyzs)
        )
    xyzs_device_handle.value = np.intp(xyzs_device_pointer)
    cuda_memcpy_htod(
        device_pointer=xyzs_device_pointer,
        host_pointer=xyzs.ctypes.data,
        size_in_bytes=sizeof_double * len(xyzs),
        )

    # The integral compute in cuEST are electron repulsion integrals but we're intersted
    # in a nucleus in the ket here, so the Z values have been scaled by -1 in the parser
    Zs_device_handle = ce.Pointer()
    Zs = np.array(Zs)
    Zs_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * len(Zs)
        )
    Zs_device_handle.value = np.intp(Zs_device_pointer)
    cuda_memcpy_htod(
        device_pointer=Zs_device_pointer,
        host_pointer=Zs.ctypes.data,
        size_in_bytes=sizeof_double * len(Zs),
        )

    potential_compute_parameters = ce.cuestPotentialComputeParameters()
    cuest_check('Create Potential Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_POTENTIALCOMPUTE_PARAMETERS,
            outParameters=potential_compute_parameters,
            )
        )

    # Find Memory Requirements
    cuest_check('Compute potential Ints Workspace Query',
        ce.cuestPotentialComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=potential_compute_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numCharges=len(symbols),
            xyz=xyzs_device_handle,
            q=Zs_device_handle,
            outVMatrix=potentialint_device_handle,
            )
        )

    potentialint_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    potentialint_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nao**2
        )
    potentialint_device_handle.value = np.intp(potentialint_device_pointer)

    # Compute Integrals
    cuest_check('Compute potential Ints',
        ce.cuestPotentialCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=potential_compute_parameters,
            temporaryWorkspace=potentialint_temporary_workspace.pointer,
            numCharges=len(symbols),
            xyz=xyzs_device_handle,
            q=Zs_device_handle,
            outVMatrix=potentialint_device_handle,
            )
        )
    cuda_free(
        array=xyzs_device_pointer,
        )
    cuda_free(
        array=Zs_device_pointer,
        )

    del potentialint_temporary_workspace

    cuest_check('Destroy Potential Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_POTENTIALCOMPUTE_PARAMETERS,
            parameters=potential_compute_parameters,
            )
        )

    potentialint_host_array = np.empty(
        (nao * nao),
        dtype=np.double
        )

    cuda_memcpy_dtoh(
        host_pointer=potentialint_host_array.ctypes.data,
        device_pointer=potentialint_device_pointer,
        size_in_bytes=sizeof_double * nao**2,
        )

    cuda_free(
        array=potentialint_device_pointer
        )


    # => Cleanup <= #

    cuest_check('Destroy OEIntPlan',
        ce.cuestOEIntPlanDestroy(
            handle=oeintplan_handle,
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

    del oeintplan_persistent_workspace
    del aopairlist_persistent_workspace
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

