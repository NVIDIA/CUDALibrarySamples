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

    # => Fake Density Matrix <= #

    densitymatrix_device_handle = ce.Pointer()
    densitymatrix_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nao**2
        )
    densitymatrix_device_handle.value = np.intp(densitymatrix_device_pointer)
    densitymatrix_host_array = np.random.normal(size=(nao, nao)).astype(np.double)
    densitymatrix_host_array = np.ravel(densitymatrix_host_array + densitymatrix_host_array.T)
    cuda_memcpy_htod(
        device_pointer=densitymatrix_device_pointer,
        host_pointer=densitymatrix_host_array.ctypes.data,
        size_in_bytes=sizeof_double * nao**2,
        )

    # The Hessian skeleton contraction is computed over two lists of atom
    # perturbations; here we request the full Hessian by using every atom as
    # both the row (A) and column (B) perturbation list.
    perturbations = list(range(len(symbols)))

    # => Overlap Hessian Skeleton <= #

    compute_overlap_hessian_skeleton_parameters = ce.cuestOverlapHessianSkeletonComputeParameters()
    cuest_check('Create Overlap Hessian Skeleton Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_OVERLAPHESSIANSKELETONCOMPUTE_PARAMETERS,
            outParameters=compute_overlap_hessian_skeleton_parameters,
            )
        )

    overlaphessianskeleton_device_handle = ce.Pointer()
    cuest_check('Compute Overlap Hessian Skeleton Workspace Query',
        ce.cuestOverlapHessianSkeletonComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_overlap_hessian_skeleton_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            perturbationsA=perturbations,
            perturbationsB=perturbations,
            densityMatrix=densitymatrix_device_handle,
            outHessianBlock=overlaphessianskeleton_device_handle,
            )
        )

    overlaphessianskeleton_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    overlaphessianskeleton_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * (3 * len(symbols)) * (3 * len(symbols))
        )
    overlaphessianskeleton_device_handle.value = np.intp(overlaphessianskeleton_device_pointer)

    cuest_check('Compute Overlap Hessian Skeleton',
        ce.cuestOverlapHessianSkeletonCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_overlap_hessian_skeleton_parameters,
            temporaryWorkspace=overlaphessianskeleton_temporary_workspace.pointer,
            perturbationsA=perturbations,
            perturbationsB=perturbations,
            densityMatrix=densitymatrix_device_handle,
            outHessianBlock=overlaphessianskeleton_device_handle,
            )
        )

    del overlaphessianskeleton_temporary_workspace
    cuest_check('Destroy Overlap Hessian Skeleton Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_OVERLAPHESSIANSKELETONCOMPUTE_PARAMETERS,
            parameters=compute_overlap_hessian_skeleton_parameters,
            )
        )

    overlaphessianskeleton_host_array = np.empty(
        (3 * len(symbols), 3 * len(symbols)),
        dtype=np.double
        )

    cuda_memcpy_dtoh(
        host_pointer=overlaphessianskeleton_host_array.ctypes.data,
        device_pointer=overlaphessianskeleton_device_pointer,
        size_in_bytes=sizeof_double * (3 * len(symbols)) * (3 * len(symbols)),
        )

    cuda_free(
        array=overlaphessianskeleton_device_pointer
        )

    # => Kinetic Hessian Skeleton <= #

    compute_kinetic_hessian_skeleton_parameters = ce.cuestKineticHessianSkeletonComputeParameters()
    cuest_check('Create Kinetic Hessian Skeleton Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_KINETICHESSIANSKELETONCOMPUTE_PARAMETERS,
            outParameters=compute_kinetic_hessian_skeleton_parameters,
            )
        )

    kinetichessianskeleton_device_handle = ce.Pointer()
    cuest_check('Compute Kinetic Hessian Skeleton Workspace Query',
        ce.cuestKineticHessianSkeletonComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_kinetic_hessian_skeleton_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            perturbationsA=perturbations,
            perturbationsB=perturbations,
            densityMatrix=densitymatrix_device_handle,
            outHessianBlock=kinetichessianskeleton_device_handle,
            )
        )

    kinetichessianskeleton_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    kinetichessianskeleton_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * (3 * len(symbols)) * (3 * len(symbols))
        )
    kinetichessianskeleton_device_handle.value = np.intp(kinetichessianskeleton_device_pointer)

    cuest_check('Compute Kinetic Hessian Skeleton',
        ce.cuestKineticHessianSkeletonCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_kinetic_hessian_skeleton_parameters,
            temporaryWorkspace=kinetichessianskeleton_temporary_workspace.pointer,
            perturbationsA=perturbations,
            perturbationsB=perturbations,
            densityMatrix=densitymatrix_device_handle,
            outHessianBlock=kinetichessianskeleton_device_handle,
            )
        )

    del kinetichessianskeleton_temporary_workspace
    cuest_check('Destroy Kinetic Hessian Skeleton Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_KINETICHESSIANSKELETONCOMPUTE_PARAMETERS,
            parameters=compute_kinetic_hessian_skeleton_parameters,
            )
        )

    kinetichessianskeleton_host_array = np.empty(
        (3 * len(symbols), 3 * len(symbols)),
        dtype=np.double
        )

    cuda_memcpy_dtoh(
        host_pointer=kinetichessianskeleton_host_array.ctypes.data,
        device_pointer=kinetichessianskeleton_device_pointer,
        size_in_bytes=sizeof_double * (3 * len(symbols)) * (3 * len(symbols)),
        )

    cuda_free(
        array=kinetichessianskeleton_device_pointer
        )

    cuda_free(
        array=densitymatrix_device_pointer
        )

    # => Fake MO Coefficient Matrices <= #

    # The Hessian RHS contraction transforms the AO-basis first derivatives
    # into the MO basis using a left and a right coefficient matrix. Here we
    # use a single (square, nao x nao) coefficient-matrix pair for both the
    # overlap and the kinetic RHS demonstrations below.
    leftcoefficient_device_handle = ce.Pointer()
    leftcoefficient_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nao * nao
        )
    leftcoefficient_device_handle.value = np.intp(leftcoefficient_device_pointer)
    leftcoefficient_host_array = np.ravel(np.random.normal(size=(nao, nao)).astype(np.double))
    cuda_memcpy_htod(
        device_pointer=leftcoefficient_device_pointer,
        host_pointer=leftcoefficient_host_array.ctypes.data,
        size_in_bytes=sizeof_double * nao * nao,
        )

    rightcoefficient_device_handle = ce.Pointer()
    rightcoefficient_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nao * nao
        )
    rightcoefficient_device_handle.value = np.intp(rightcoefficient_device_pointer)
    rightcoefficient_host_array = np.ravel(np.random.normal(size=(nao, nao)).astype(np.double))
    cuda_memcpy_htod(
        device_pointer=rightcoefficient_device_pointer,
        host_pointer=rightcoefficient_host_array.ctypes.data,
        size_in_bytes=sizeof_double * nao * nao,
        )

    num_left_orbitals = [nao]
    num_right_orbitals = [nao]

    # => Overlap Hessian RHS <= #

    compute_overlap_hessian_rhs_parameters = ce.cuestOverlapHessianRHSComputeParameters()
    cuest_check('Create Overlap Hessian RHS Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_OVERLAPHESSIANRHSCOMPUTE_PARAMETERS,
            outParameters=compute_overlap_hessian_rhs_parameters,
            )
        )

    overlaphessianrhs_device_handle = ce.Pointer()
    cuest_check('Compute Overlap Hessian RHS Workspace Query',
        ce.cuestOverlapHessianRHSComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_overlap_hessian_rhs_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            perturbations=perturbations,
            numLeftOrbitals=num_left_orbitals,
            numRightOrbitals=num_right_orbitals,
            leftCoefficientMatrices=leftcoefficient_device_handle,
            rightCoefficientMatrices=rightcoefficient_device_handle,
            outTensors=overlaphessianrhs_device_handle,
            )
        )

    overlaphessianrhs_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    overlaphessianrhs_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * (3 * len(symbols)) * nao * nao
        )
    overlaphessianrhs_device_handle.value = np.intp(overlaphessianrhs_device_pointer)

    cuest_check('Compute Overlap Hessian RHS',
        ce.cuestOverlapHessianRHSCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_overlap_hessian_rhs_parameters,
            temporaryWorkspace=overlaphessianrhs_temporary_workspace.pointer,
            perturbations=perturbations,
            numLeftOrbitals=num_left_orbitals,
            numRightOrbitals=num_right_orbitals,
            leftCoefficientMatrices=leftcoefficient_device_handle,
            rightCoefficientMatrices=rightcoefficient_device_handle,
            outTensors=overlaphessianrhs_device_handle,
            )
        )

    del overlaphessianrhs_temporary_workspace
    cuest_check('Destroy Overlap Hessian RHS Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_OVERLAPHESSIANRHSCOMPUTE_PARAMETERS,
            parameters=compute_overlap_hessian_rhs_parameters,
            )
        )

    overlaphessianrhs_host_array = np.empty(
        (3 * len(symbols) * nao * nao,),
        dtype=np.double
        )

    cuda_memcpy_dtoh(
        host_pointer=overlaphessianrhs_host_array.ctypes.data,
        device_pointer=overlaphessianrhs_device_pointer,
        size_in_bytes=sizeof_double * (3 * len(symbols)) * nao * nao,
        )

    cuda_free(
        array=overlaphessianrhs_device_pointer
        )

    # => Kinetic Hessian RHS <= #

    compute_kinetic_hessian_rhs_parameters = ce.cuestKineticHessianRHSComputeParameters()
    cuest_check('Create Kinetic Hessian RHS Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_KINETICHESSIANRHSCOMPUTE_PARAMETERS,
            outParameters=compute_kinetic_hessian_rhs_parameters,
            )
        )

    kinetichessianrhs_device_handle = ce.Pointer()
    cuest_check('Compute Kinetic Hessian RHS Workspace Query',
        ce.cuestKineticHessianRHSComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_kinetic_hessian_rhs_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            perturbations=perturbations,
            numLeftOrbitals=num_left_orbitals,
            numRightOrbitals=num_right_orbitals,
            leftCoefficientMatrices=leftcoefficient_device_handle,
            rightCoefficientMatrices=rightcoefficient_device_handle,
            outTensors=kinetichessianrhs_device_handle,
            )
        )

    kinetichessianrhs_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    kinetichessianrhs_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * (3 * len(symbols)) * nao * nao
        )
    kinetichessianrhs_device_handle.value = np.intp(kinetichessianrhs_device_pointer)

    cuest_check('Compute Kinetic Hessian RHS',
        ce.cuestKineticHessianRHSCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=compute_kinetic_hessian_rhs_parameters,
            temporaryWorkspace=kinetichessianrhs_temporary_workspace.pointer,
            perturbations=perturbations,
            numLeftOrbitals=num_left_orbitals,
            numRightOrbitals=num_right_orbitals,
            leftCoefficientMatrices=leftcoefficient_device_handle,
            rightCoefficientMatrices=rightcoefficient_device_handle,
            outTensors=kinetichessianrhs_device_handle,
            )
        )

    del kinetichessianrhs_temporary_workspace
    cuest_check('Destroy Kinetic Hessian RHS Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_KINETICHESSIANRHSCOMPUTE_PARAMETERS,
            parameters=compute_kinetic_hessian_rhs_parameters,
            )
        )

    kinetichessianrhs_host_array = np.empty(
        (3 * len(symbols) * nao * nao,),
        dtype=np.double
        )

    cuda_memcpy_dtoh(
        host_pointer=kinetichessianrhs_host_array.ctypes.data,
        device_pointer=kinetichessianrhs_device_pointer,
        size_in_bytes=sizeof_double * (3 * len(symbols)) * nao * nao,
        )

    cuda_free(
        array=kinetichessianrhs_device_pointer
        )

    # => Cleanup <= #

    cuda_free(
        array=leftcoefficient_device_pointer
        )
    cuda_free(
        array=rightcoefficient_device_pointer
        )

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
