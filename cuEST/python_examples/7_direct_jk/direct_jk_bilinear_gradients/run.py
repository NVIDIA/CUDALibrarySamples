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


# This sample shows how to use the direct (integral-direct, 4-center)
# algorithm to evaluate nuclear gradients of the bilinear Coulomb and
# exchange forms
#
#     E_J(D1, D2) = sum_pq D1_pq J[D2]_pq      E_K(D1, D2) = sum_pq D1_pq K[D2]_pq
#
# at fixed density coefficients, for a pair of density matrices (D1, D2),
# symmetric for the exchange form. The single-density gradient routines (see the direct_jk_gradients
# sample) evaluate the quadratic forms E_J(D, D) and E_K(D, D); the bilinear
# routines take two densities and are the tool of choice whenever a cross
# term between two different densities is needed -- for example a guess or
# delta density against a converged one, or the contraction of a response
# density with the SCF density in a coupled-perturbed calculation.
#
# The bilinear forms are symmetric in their arguments and related to the
# single-density gradients by the polarization identity
#
#     grad E_J(D1, D2) = 1/2 [ grad E_J(D1+D2) - grad E_J(D1) - grad E_J(D2) ]
#
# (and likewise for K); the bilinear routine replaces the three gradients of
# the identity by one call, and the exchange form's screening benefits from
# sparsity in either density.
#
# A batch of density pairs can be processed in a single bilinear call: the
# first densities are stacked contiguously in densityMatrices1, the second
# ones in densityMatrices2, and the number of pairs is passed as
# numDensityMatrices. One gradient (numAtoms x 3) is produced per pair; the
# output buffers are overwritten, not accumulated. For the exchange form both
# densities of every pair must be symmetric.
#
# Since density matrices are not available, synthetic data is generated
# to populate these matrices.
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
    # => Build Fake Density Matrices <= #

    # A single pair of symmetric density matrices (D1, D2) is used here (batching: see the header).
    num_density_matrices = 1
    natom = len(symbols)

    # In real use cases, these are density matrices obtained during an SCF
    # procedure, e.g. the initial guess density and the converged density.
    rng = np.random.default_rng(0)
    D1 = rng.normal(size=(nao, nao))
    D1 = np.ravel(D1 + D1.T)
    D2 = rng.normal(size=(nao, nao))
    D2 = np.ravel(D2 + D2.T)

    density1_device_pointer = cuda_malloc(size_in_bytes=sizeof_double * nao**2)
    density1_device_handle = ce.Pointer()
    density1_device_handle.value = np.intp(density1_device_pointer)
    cuda_memcpy_htod(
        device_pointer=density1_device_pointer,
        host_pointer=D1.ctypes.data,
        size_in_bytes=sizeof_double * nao**2,
        )

    density2_device_pointer = cuda_malloc(size_in_bytes=sizeof_double * nao**2)
    density2_device_handle = ce.Pointer()
    density2_device_handle.value = np.intp(density2_device_pointer)
    cuda_memcpy_htod(
        device_pointer=density2_device_pointer,
        host_pointer=D2.ctypes.data,
        size_in_bytes=sizeof_double * nao**2,
        )

    # => Build Direct Integral Plan <= #

    # The exchange operator is defined when the plan is created. The defaults
    # (exchange fraction 1.0, LRC exchange fraction 0.0) request full-range
    # exact exchange.
    directintplan_handle = ce.cuestDirectIntPlanHandle()
    directintplan_parameters = ce.cuestDirectIntPlanParameters()
    cuest_check('Create DirectIntPlan Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_DIRECTINTPLAN_PARAMETERS,
            outParameters=directintplan_parameters,
            )
        )

    cuest_check('Create DirectIntPlan Workspace Query',
        ce.cuestDirectIntPlanCreateWorkspaceQuery(
            handle=cuest_handle,
            primaryBasis=aobasis_handle,
            pairList=aopairlist_handle,
            parameters=directintplan_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outPlan=directintplan_handle,
            )
        )

    print("DirectIntPlan Persistent sizes:", persistent_workspace_descriptor)
    print("DirectIntPlan Temporary sizes:", temporary_workspace_descriptor)

    directintplan_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)
    directintplan_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)

    cuest_check('Create DirectIntPlan',
        ce.cuestDirectIntPlanCreate(
            handle=cuest_handle,
            primaryBasis=aobasis_handle,
            pairList=aopairlist_handle,
            parameters=directintplan_parameters,
            persistentWorkspace=directintplan_persistent_workspace.pointer,
            temporaryWorkspace=directintplan_temporary_workspace.pointer,
            outPlan=directintplan_handle,
            )
        )

    del directintplan_temporary_workspace

    cuest_check('Destroy DirectIntPlan Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_DIRECTINTPLAN_PARAMETERS,
            parameters=directintplan_parameters,
            )
        )

    # => Bilinear Coulomb Gradient <= #

    # The bilinear routine shares its parameters type with the single-density routine.
    compute_coulomb_gradient_parameters = ce.cuestDirectCoulombDerivativeComputeParameters()
    cuest_check('Create Coulomb Derivative Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_DIRECTCOULOMBDERIVATIVECOMPUTE_PARAMETERS,
            outParameters=compute_coulomb_gradient_parameters,
            )
        )

    coulombgrad_device_pointer = cuda_malloc(size_in_bytes=sizeof_double * num_density_matrices * natom * 3)
    coulombgrad_device_handle = ce.Pointer()
    coulombgrad_device_handle.value = np.intp(coulombgrad_device_pointer)

    cuest_check('Compute Bilinear Coulomb Derivative Workspace Query',
        ce.cuestDirectCoulombBilinearDerivativeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=directintplan_handle,
            parameters=compute_coulomb_gradient_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numDensityMatrices=num_density_matrices,
            densityMatrices1=density1_device_handle,
            densityMatrices2=density2_device_handle,
            outCoulombGradients=coulombgrad_device_handle,
            )
        )

    print("Bilinear CoulombGrad Temporary sizes:", temporary_workspace_descriptor)

    coulombgrad_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    # Compute the bilinear J gradient d/dR sum_pq D1_pq J[D2]_pq
    cuest_check('Compute Bilinear Coulomb Derivative',
        ce.cuestDirectCoulombBilinearDerivativeCompute(
            handle=cuest_handle,
            plan=directintplan_handle,
            parameters=compute_coulomb_gradient_parameters,
            temporaryWorkspace=coulombgrad_temporary_workspace.pointer,
            numDensityMatrices=num_density_matrices,
            densityMatrices1=density1_device_handle,
            densityMatrices2=density2_device_handle,
            outCoulombGradients=coulombgrad_device_handle,
            )
        )

    del coulombgrad_temporary_workspace

    coulombgrad_host_array = np.empty((num_density_matrices * natom * 3,), dtype=np.double)
    cuda_memcpy_dtoh(
        host_pointer=coulombgrad_host_array.ctypes.data,
        device_pointer=coulombgrad_device_pointer,
        size_in_bytes=sizeof_double * num_density_matrices * natom * 3,
        )
    cuda_free(array=coulombgrad_device_pointer)

    cuest_check('Destroy Coulomb Derivative Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_DIRECTCOULOMBDERIVATIVECOMPUTE_PARAMETERS,
            parameters=compute_coulomb_gradient_parameters,
            )
        )

    # => Bilinear Exchange Gradient <= #

    compute_exchange_gradient_parameters = ce.cuestDirectSymmetricExchangeDerivativeComputeParameters()
    cuest_check('Create Exchange Derivative Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_DIRECTSYMMETRICEXCHANGEDERIVATIVECOMPUTE_PARAMETERS,
            outParameters=compute_exchange_gradient_parameters,
            )
        )

    exchangegrad_device_pointer = cuda_malloc(size_in_bytes=sizeof_double * num_density_matrices * natom * 3)
    exchangegrad_device_handle = ce.Pointer()
    exchangegrad_device_handle.value = np.intp(exchangegrad_device_pointer)

    cuest_check('Compute Bilinear Exchange Derivative Workspace Query',
        ce.cuestDirectSymmetricExchangeBilinearDerivativeComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=directintplan_handle,
            parameters=compute_exchange_gradient_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numDensityMatrices=num_density_matrices,
            densityMatrices1=density1_device_handle,
            densityMatrices2=density2_device_handle,
            outExchangeGradients=exchangegrad_device_handle,
            )
        )

    print("Bilinear ExchangeGrad Temporary sizes:", temporary_workspace_descriptor)

    exchangegrad_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    # Compute the bilinear K gradient d/dR sum_pq D1_pq K[D2]_pq
    cuest_check('Compute Bilinear Exchange Derivative',
        ce.cuestDirectSymmetricExchangeBilinearDerivativeCompute(
            handle=cuest_handle,
            plan=directintplan_handle,
            parameters=compute_exchange_gradient_parameters,
            temporaryWorkspace=exchangegrad_temporary_workspace.pointer,
            numDensityMatrices=num_density_matrices,
            densityMatrices1=density1_device_handle,
            densityMatrices2=density2_device_handle,
            outExchangeGradients=exchangegrad_device_handle,
            )
        )

    del exchangegrad_temporary_workspace

    exchangegrad_host_array = np.empty((num_density_matrices * natom * 3,), dtype=np.double)
    cuda_memcpy_dtoh(
        host_pointer=exchangegrad_host_array.ctypes.data,
        device_pointer=exchangegrad_device_pointer,
        size_in_bytes=sizeof_double * num_density_matrices * natom * 3,
        )
    cuda_free(array=exchangegrad_device_pointer)

    cuest_check('Destroy Exchange Derivative Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_DIRECTSYMMETRICEXCHANGEDERIVATIVECOMPUTE_PARAMETERS,
            parameters=compute_exchange_gradient_parameters,
            )
        )

    # => Cleanup <= #

    cuda_free(array=density1_device_pointer)
    cuda_free(array=density2_device_pointer)

    cuest_check('Destroy DirectIntPlan',
        ce.cuestDirectIntPlanDestroy(
            handle=directintplan_handle,
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

    del directintplan_persistent_workspace
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
