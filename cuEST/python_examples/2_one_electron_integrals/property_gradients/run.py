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


# This sample demonstrates how to compute the derivatives of the
# one-electron integrals contracted with a density matrix. In cuEST,
# the derivative integrals cannot be computed individually, but are
# always contracted with a density (or pseudo-density) matrix. The
# result of this contraction is stored in a number of atoms by 3 array.
#
# In this sample, a "real" density matrix is not availble, so a random
# non-symmetric matrix is substituted.
#
# The use of the one-electron integral derivative routines follows very
# closely with the underlying one-electron integrals.

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

    # => Fake Non-Symmetric Density Matrix <= #

    # The property derivative routines contract with a (pseudo-)density matrix.
    # In real use cases this comes from a calculation; here we use random data.
    np.random.seed(0)
    densitymatrix_host_array = np.random.normal(size=(nao, nao)).astype(np.double).ravel()

    densitymatrix_device_handle = ce.Pointer()
    densitymatrix_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * nao**2
        )
    densitymatrix_device_handle.value = np.intp(densitymatrix_device_pointer)
    cuda_memcpy_htod(
        device_pointer=densitymatrix_device_pointer,
        host_pointer=densitymatrix_host_array.ctypes.data,
        size_in_bytes=sizeof_double * nao**2,
        )

    # Example operator settings
    multipole_order = [1, 0, 0]
    origin = np.array([0.0, 0.0, 0.0])

    # => Angular Momentum Derivative <= #

    angular_momentum_grad_device_handle = ce.Pointer()
    angular_momentum_compute_parameters = ce.cuestAngularMomentumDerivativeComputeParameters()
    cuest_check('Create AngularMomentum Deriv Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_ANGULARMOMENTUMDERIVATIVECOMPUTE_PARAMETERS,
            outParameters=angular_momentum_compute_parameters,
            )
        )

    cuest_check('Compute AngularMomentum Deriv Workspace Query',
        ce.cuestAngularMomentumDerivativeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=angular_momentum_compute_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            component=ce.CuestAngularMomentumComputeParametersComponent.CUEST_ANGULARMOMENTUMCOMPUTE_PARAMETERS_COMPONENT_LZ,
            origin=origin,
            densityMatrix=densitymatrix_device_handle,
            outGradient=angular_momentum_grad_device_handle,
            )
        )

    angular_momentum_grad_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    angular_momentum_grad_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * len(symbols) * 3
        )
    angular_momentum_grad_device_handle.value = np.intp(angular_momentum_grad_device_pointer)

    cuest_check('Compute AngularMomentum Deriv',
        ce.cuestAngularMomentumDerivativeCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=angular_momentum_compute_parameters,
            temporaryWorkspace=angular_momentum_grad_temporary_workspace.pointer,
            component=ce.CuestAngularMomentumComputeParametersComponent.CUEST_ANGULARMOMENTUMCOMPUTE_PARAMETERS_COMPONENT_LZ,
            origin=origin,
            densityMatrix=densitymatrix_device_handle,
            outGradient=angular_momentum_grad_device_handle,
            )
        )

    del angular_momentum_grad_temporary_workspace

    cuest_check('Destroy AngularMomentum Deriv Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_ANGULARMOMENTUMDERIVATIVECOMPUTE_PARAMETERS,
            parameters=angular_momentum_compute_parameters,
            )
        )

    angular_momentum_grad_host_array = np.empty(
        (len(symbols), 3),
        dtype=np.double
        )

    cuda_memcpy_dtoh(
        host_pointer=angular_momentum_grad_host_array.ctypes.data,
        device_pointer=angular_momentum_grad_device_pointer,
        size_in_bytes=sizeof_double * len(symbols) * 3,
        )

    cuda_free(
        array=angular_momentum_grad_device_pointer
        )

    # => Nabla Derivative <= #

    nabla_grad_device_handle = ce.Pointer()
    nabla_compute_parameters = ce.cuestNablaDerivativeComputeParameters()
    cuest_check('Create Nabla Deriv Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_NABLADERIVATIVECOMPUTE_PARAMETERS,
            outParameters=nabla_compute_parameters,
            )
        )

    cuest_check('Compute Nabla Deriv Workspace Query',
        ce.cuestNablaDerivativeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=nabla_compute_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            component=ce.CuestNablaComputeParametersComponent.CUEST_NABLACOMPUTE_PARAMETERS_COMPONENT_X,
            densityMatrix=densitymatrix_device_handle,
            outGradient=nabla_grad_device_handle,
            )
        )

    nabla_grad_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    nabla_grad_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * len(symbols) * 3
        )
    nabla_grad_device_handle.value = np.intp(nabla_grad_device_pointer)

    cuest_check('Compute Nabla Deriv',
        ce.cuestNablaDerivativeCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=nabla_compute_parameters,
            temporaryWorkspace=nabla_grad_temporary_workspace.pointer,
            component=ce.CuestNablaComputeParametersComponent.CUEST_NABLACOMPUTE_PARAMETERS_COMPONENT_X,
            densityMatrix=densitymatrix_device_handle,
            outGradient=nabla_grad_device_handle,
            )
        )

    del nabla_grad_temporary_workspace

    cuest_check('Destroy Nabla Deriv Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_NABLADERIVATIVECOMPUTE_PARAMETERS,
            parameters=nabla_compute_parameters,
            )
        )

    nabla_grad_host_array = np.empty(
        (len(symbols), 3),
        dtype=np.double
        )

    cuda_memcpy_dtoh(
        host_pointer=nabla_grad_host_array.ctypes.data,
        device_pointer=nabla_grad_device_pointer,
        size_in_bytes=sizeof_double * len(symbols) * 3,
        )

    cuda_free(
        array=nabla_grad_device_pointer
        )

    # => Multipole Derivative <= #

    multipole_grad_device_handle = ce.Pointer()
    multipole_compute_parameters = ce.cuestMultipoleDerivativeComputeParameters()
    cuest_check('Create Multipole Deriv Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_MULTIPOLEDERIVATIVECOMPUTE_PARAMETERS,
            outParameters=multipole_compute_parameters,
            )
        )

    cuest_check('Compute Multipole Deriv Workspace Query',
        ce.cuestMultipoleDerivativeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=multipole_compute_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            multipoleOrder=multipole_order,
            origin=origin,
            densityMatrix=densitymatrix_device_handle,
            outGradient=multipole_grad_device_handle,
            )
        )

    multipole_grad_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    multipole_grad_device_pointer = cuda_malloc(
        size_in_bytes=sizeof_double * len(symbols) * 3
        )
    multipole_grad_device_handle.value = np.intp(multipole_grad_device_pointer)

    cuest_check('Compute Multipole Deriv',
        ce.cuestMultipoleDerivativeCompute(
            handle=cuest_handle,
            plan=oeintplan_handle,
            parameters=multipole_compute_parameters,
            temporaryWorkspace=multipole_grad_temporary_workspace.pointer,
            multipoleOrder=multipole_order,
            origin=origin,
            densityMatrix=densitymatrix_device_handle,
            outGradient=multipole_grad_device_handle,
            )
        )

    del multipole_grad_temporary_workspace

    cuest_check('Destroy Multipole Deriv Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_MULTIPOLEDERIVATIVECOMPUTE_PARAMETERS,
            parameters=multipole_compute_parameters,
            )
        )

    multipole_grad_host_array = np.empty(
        (len(symbols), 3),
        dtype=np.double
        )

    cuda_memcpy_dtoh(
        host_pointer=multipole_grad_host_array.ctypes.data,
        device_pointer=multipole_grad_device_pointer,
        size_in_bytes=sizeof_double * len(symbols) * 3,
        )

    cuda_free(
        array=multipole_grad_device_pointer
        )

    # => Cleanup <= #

    cuda_free(
        array=densitymatrix_device_pointer
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
