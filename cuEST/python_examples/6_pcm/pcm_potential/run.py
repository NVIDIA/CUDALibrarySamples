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

# Inject the directory where the example helper utilities live
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helpers.cuda_utils import cuda_malloc, cuda_free, cuda_memcpy_dtoh, cuda_memcpy_htod, WorkspaceDescriptor, Workspace
from helpers.utilities import normalize_shell_coefficients
from helpers.parsers import simple_xyz_parser, simple_gbs_parser
from helpers.pcm_utils import pcm_cavity_parameters

def run(
    *,
    xyz_filename,
    gbs_filename,
    ):
    """
    This sample shows how to use cuEST to compute the PCM potential
    matrix and the PCM dislectric energy. The PCM potential matrix
    is obtained by solving for the induced surface charges on the
    cavity via a preconditioned conjugate gradient (PCG) solver.

    In this sample, a random symmetric matrix stands in for a real
    density matrix. Molecular cavity parameters are looked up
    per-element from a helper_pcm.h utility file.
    """

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
        return_code,
        ):
        "A simple helper function to call cuEST functions and check the return code."

        if return_code != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError(f"{title} failed with code {return_code}")

    persistent_workspace_descriptor = WorkspaceDescriptor()
    temporary_workspace_descriptor = WorkspaceDescriptor()

    epsilon = 80.0
    num_angular_points_per_atom, zetas, atomic_radii, effective_nuclear_charges = \
        pcm_cavity_parameters(symbols=symbols, Zs=Zs)

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

    # => Build AO Pair List <= #

    aopairlist_handle = ce.cuestAOPairListHandle()
    aopairlist_parameters = ce.cuestAOPairListParameters()

    cuest_check('Create AO Pair List Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_AOPAIRLIST_PARAMETERS,
            outParameters=aopairlist_parameters,
            )
        )

    cuest_check('Create AOPairList Workspace Query',
        ce.cuestAOPairListCreateWorkspaceQuery(
            handle=cuest_handle,
            basis=aobasis_handle,
            numAtoms=num_atoms,
            xyz=xyzs,
            thresholdPQ=1.0e-14,
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
            numAtoms=num_atoms,
            xyz=xyzs,
            thresholdPQ=1.0e-14,
            parameters=aopairlist_parameters,
            persistentWorkspace=aopairlist_persistent_workspace.pointer,
            temporaryWorkspace=aopairlist_temporary_workspace.pointer,
            outPairList=aopairlist_handle,
            )
        )

    del aopairlist_temporary_workspace

    cuest_check('Destroy AO Pair List Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_AOPAIRLIST_PARAMETERS,
            parameters=aopairlist_parameters,
            )
        )

    # => Build OE Integral Plan <= #

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

    # => Build PCM Integral Plan <= #

    pcmintplan_handle = ce.cuestPCMIntPlanHandle()
    pcmintplan_parameters = ce.cuestPCMIntPlanParameters()

    cuest_check('Create PCMIntPlan Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_PCMINTPLAN_PARAMETERS,
            outParameters=pcmintplan_parameters,
            )
        )

    cuest_check('Create PCMIntPlan Workspace Query',
        ce.cuestPCMIntPlanCreateWorkspaceQuery(
            handle=cuest_handle,
            intPlan=oeintplan_handle,
            parameters=pcmintplan_parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            numAngularPointsPerAtom=num_angular_points_per_atom,
            epsilon=epsilon,
            zetas=zetas,
            atomicRadii=atomic_radii,
            effectiveNuclearCharges=effective_nuclear_charges,
            outPlan=pcmintplan_handle,
            )
        )

    pcmintplan_persistent_workspace = Workspace(workspaceDescriptor=persistent_workspace_descriptor)
    pcmintplan_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    cuest_check('Create PCMIntPlan',
        ce.cuestPCMIntPlanCreate(
            handle=cuest_handle,
            intPlan=oeintplan_handle,
            parameters=pcmintplan_parameters,
            persistentWorkspace=pcmintplan_persistent_workspace.pointer,
            temporaryWorkspace=pcmintplan_temporary_workspace.pointer,
            numAngularPointsPerAtom=num_angular_points_per_atom,
            epsilon=epsilon,
            zetas=zetas,
            atomicRadii=atomic_radii,
            effectiveNuclearCharges=effective_nuclear_charges,
            outPlan=pcmintplan_handle,
            )
        )

    del pcmintplan_temporary_workspace

    cuest_check('Destroy PCMIntPlan Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_PCMINTPLAN_PARAMETERS,
            parameters=pcmintplan_parameters,
            )
        )

    # => Compute PCM Potential Matrix <= #

    pcm_npoints = ce.data_uint64_t()
    cuest_check('Query PCMIntPlan Num Points',
        ce.cuestQuery(
            handle=cuest_handle,
            type=ce.CuestType.CUEST_PCMINTPLAN,
            object=pcmintplan_handle,
            attribute=ce.CuestPCMIntPlanAttributes.CUEST_PCMINTPLAN_NUM_POINT,
            attributeValue=pcm_npoints,
            )
        )
    npoints = pcm_npoints.value

    # A real calculation would supply the total (alpha + beta) SCF density matrix here;
    # we use a random symmetric matrix as a stand-in.
    np.random.seed(0)
    density_host = np.random.normal(size=(nao, nao)).astype(np.double)
    density_host = 0.5 * (density_host + density_host.T)

    density_device_pointer = cuda_malloc(size_in_bytes=sizeof_double * nao**2)
    density_device_handle = ce.Pointer(density_device_pointer)
    cuda_memcpy_htod(
        device_pointer=density_device_pointer,
        host_pointer=density_host.ctypes.data,
        size_in_bytes=sizeof_double * nao**2,
        )

    inq_device_pointer = cuda_malloc(size_in_bytes=sizeof_double * npoints)
    inq_device_handle = ce.Pointer(inq_device_pointer)
    # Zero the initial charge guess; the solver will find the converged charges.
    zeros = np.zeros(npoints, dtype=np.double)
    cuda_memcpy_htod(
        device_pointer=inq_device_pointer,
        host_pointer=zeros.ctypes.data,
        size_in_bytes=sizeof_double * npoints,
        )

    outq_device_pointer = cuda_malloc(size_in_bytes=sizeof_double * npoints)
    outq_device_handle = ce.Pointer(outq_device_pointer)

    pcm_potential_device_pointer = cuda_malloc(size_in_bytes=sizeof_double * nao**2)
    pcm_potential_device_handle = ce.Pointer(pcm_potential_device_pointer)

    pcm_results = ce.cuestPCMResultsHandle()
    cuest_check('Create PCM Results',
        ce.cuestResultsCreate(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            outResults=pcm_results,
            )
        )

    pcm_potential_parameters = ce.cuestPCMPotentialComputeParameters()
    cuest_check('Create PCMPotentialCompute Params',
        ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS,
            outParameters=pcm_potential_parameters,
            )
        )

    cuest_check('PCMPotentialCompute Workspace Query',
        ce.cuestPCMPotentialComputeWorkspaceQuery(
            handle=cuest_handle,
            plan=pcmintplan_handle,
            parameters=pcm_potential_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityMatrix=density_device_handle,
            inQ=inq_device_handle,
            outQ=outq_device_handle,
            outPCMResults=pcm_results,
            outPCMPotentialMatrix=pcm_potential_device_handle,
            )
        )

    pcm_potential_temporary_workspace = Workspace(workspaceDescriptor=temporary_workspace_descriptor)

    # Solve for the induced surface charges and compute the PCM potential matrix;
    # result is (nao, nao).
    cuest_check('PCMPotentialCompute',
        ce.cuestPCMPotentialCompute(
            handle=cuest_handle,
            plan=pcmintplan_handle,
            parameters=pcm_potential_parameters,
            temporaryWorkspace=pcm_potential_temporary_workspace.pointer,
            densityMatrix=density_device_handle,
            inQ=inq_device_handle,
            outQ=outq_device_handle,
            outPCMResults=pcm_results,
            outPCMPotentialMatrix=pcm_potential_device_handle,
            )
        )

    # At this point d_outQ contains the (hopefully) converged surface charges.
    # The inQ and outQ pointers can be swapped here to provide a good guess for the
    # PCM potential matrix solve in the next SCF cycle.

    del pcm_potential_temporary_workspace

    cuest_check('Destroy PCMPotentialCompute Params',
        ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS,
            parameters=pcm_potential_parameters,
            )
        )

    # Query convergence information from the PCM results object.
    pcm_energy = ce.data_double()
    cuest_check('Query PCM Dielectric Energy',
        ce.cuestResultsQuery(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results,
            attribute=ce.CuestPCMResultAttributes.CUEST_PCMRESULT_PCM_DIELECTRIC_ENERGY,
            attributeValue=pcm_energy,
            )
        )

    converged_residual = ce.data_double()
    cuest_check('Query PCM Converged Residual',
        ce.cuestResultsQuery(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results,
            attribute=ce.CuestPCMResultAttributes.CUEST_PCMRESULT_CONVERGED_RESIDUAL,
            attributeValue=converged_residual,
            )
        )

    num_iterations = ce.data_uint64_t()
    cuest_check('Query PCM Num Iterations',
        ce.cuestResultsQuery(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results,
            attribute=ce.CuestPCMResultAttributes.CUEST_PCMRESULT_NUM_ITERATIONS_TAKEN,
            attributeValue=num_iterations,
            )
        )

    converged = ce.data_int32_t()
    cuest_check('Query PCM Converged',
        ce.cuestResultsQuery(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results,
            attribute=ce.CuestPCMResultAttributes.CUEST_PCMRESULT_CONVERGED,
            attributeValue=converged,
            )
        )

    print(f"PCM dielectric energy:  {pcm_energy.value:20.12f} Eh")
    print(f"PCG converged residual: {converged_residual.value:20.12e}")
    print(f"PCG iterations taken:   {num_iterations.value}")
    print(f"PCG converged:          {'yes' if converged.value else 'no'}")

    cuda_free(array=density_device_pointer)
    cuda_free(array=inq_device_pointer)
    cuda_free(array=outq_device_pointer)
    cuda_free(array=pcm_potential_device_pointer)

    # => Cleanup <= #

    cuest_check('Destroy PCM Results',
        ce.cuestResultsDestroy(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results,
            )
        )

    cuest_check('Destroy PCMIntPlan',
        ce.cuestPCMIntPlanDestroy(
            handle=pcmintplan_handle,
            )
        )
    del pcmintplan_persistent_workspace

    cuest_check('Destroy OEIntPlan',
        ce.cuestOEIntPlanDestroy(
            handle=oeintplan_handle,
            )
        )
    del oeintplan_persistent_workspace

    cuest_check('Destroy AOPairList',
        ce.cuestAOPairListDestroy(
            handle=aopairlist_handle,
            )
        )
    del aopairlist_persistent_workspace

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
        description="Compute the PCM potential matrix contracted with a density matrix."
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

    args = parser.parse_args()

    run(
        xyz_filename=args.xyz_filename,
        gbs_filename=args.gbs_filename,
        )
