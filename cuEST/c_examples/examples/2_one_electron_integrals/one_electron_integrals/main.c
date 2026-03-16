/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuest.h>

#include <helper_status.h>
#include <helper_workspaces.h>
#include <helper_xyz_parser.h>
#include <helper_gbs_parser.h>
#include <helper_ao_shells.h>

/*
 * This sample shows how to use cuEST to compute
 * one-electron integrals. In particular, overlap (S),
 * kinetic (T) and potential (V) integrals are computed.
 * 
 * This introduces the AO pair list handle, which accounts 
 * for sparsity in the pair space and the one-electron
 * integral plan, which stores the necessary intermediate
 * data required to calculate one-electron integrals. The
 * integrals themselves are generated directly into a user
 * provided device array allocated with cudaMalloc.
 */
int main(int argc, char **argv)
{
    /* Check that an xyz file has been provided */
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <xyz_file_path> <gbs_file_path>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const char* xyzFilePath = argv[1];
    const char* gbsFilePath = argv[2];

    /* Parse the XYZ file */
    parsedXYZFile_t* xyzData = parseXYZFile(xyzFilePath, 1.0 / 0.52917720859);
    if (!xyzData) {
        fprintf(stderr, "Error: failed to parse file '%s'\n", xyzFilePath);
        exit(EXIT_FAILURE);
    }

    /**********************/
    /* cuEST Handle Setup */
    /**********************/

    /* Create the cuEST handle. */
    cuestHandle_t handle;
    cuestHandleParameters_t handle_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_HANDLE_PARAMETERS, 
        &handle_parameters));
    checkCuestErrors(cuestCreate(
        handle_parameters, 
        &handle));
    checkCuestErrors(cuestParametersDestroy(
        CUEST_HANDLE_PARAMETERS, 
        handle_parameters));

    /************************/
    /* cuEST AO Shell Setup */
    /************************/

    /* Use the AO shell helper to build the array of AO shells */
    AtomShellData_t* shell_data = formAOShells(handle, xyzData, gbsFilePath, 1);

    /* Unpack the AtomShellData_t struct */
    uint64_t numAtoms = shell_data->numAtoms;
    uint64_t numShellsTotal = shell_data->numShellsTotal;
    uint64_t* numShellsPerAtom = shell_data->numShellsPerAtom;
    cuestAOShell_t* shells = shell_data->shells;

    /************************/
    /* cuEST AO Basis Setup */
    /************************/

    /* Declare the AO basis handle. */
    cuestAOBasis_t basis;

    /* Declare the AO basis parameter handle. */
    cuestAOBasisParameters_t basis_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_AOBASIS_PARAMETERS, 
        &basis_parameters));

    /* Allocate space for workspace descriptors. Will be used to determine how large the workspace needs to be. */
    cuestWorkspaceDescriptor_t* persistentWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    cuestWorkspaceDescriptor_t* temporaryWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));

    /* Make the workspace query to populate the workspace descriptors. */
    checkCuestErrors(cuestAOBasisCreateWorkspaceQuery(
        handle, 
        numAtoms, 
        numShellsPerAtom, 
        (const cuestAOShell_t*) shells, 
        basis_parameters, 
        persistentWorkspaceDescriptor, 
        temporaryWorkspaceDescriptor, 
        &basis));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentBasisWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryBasisWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

    /* Create the AO basis handle. */
    checkCuestErrors(cuestAOBasisCreate(
        handle, 
        numAtoms, 
        numShellsPerAtom, 
        (const cuestAOShell_t*) shells, 
        basis_parameters, 
        persistentBasisWorkspace, 
        temporaryBasisWorkspace, 
        &basis));

    /* The temporary workspace is no longer needed. */
    freeWorkspace(temporaryBasisWorkspace);

    /* The AO basis parameters are no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_AOBASIS_PARAMETERS, 
        basis_parameters));

    /* Free all the shell data */
    freeAOShellData(shell_data);

    /****************************/
    /* cuEST AO Pair List Setup */
    /****************************/

    /* Declare the AO pair list handle. */
    cuestAOPairList_t pair_list;

    /* Declare the AO pair list parameter handle. */
    cuestAOPairListParameters_t pair_list_parameters;
    checkCuestErrors(cuestParametersCreate(CUEST_AOPAIRLIST_PARAMETERS, &pair_list_parameters));

    /* Determine how large the workspace must be to form and store the pair list. */
    checkCuestErrors(cuestAOPairListCreateWorkspaceQuery(
        handle, 
        basis, 
        numAtoms, 
        xyzData->xyzCPU, 
        1.0e-14, 
        pair_list_parameters, 
        persistentWorkspaceDescriptor, 
        temporaryWorkspaceDescriptor, 
        &pair_list));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentAOPairListWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryAOPairListWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

    /* Create the pair list. */
    checkCuestErrors(cuestAOPairListCreate(
        handle, 
        basis, 
        numAtoms,
        xyzData->xyzCPU, 
        1.0e-14, 
        pair_list_parameters, 
        persistentAOPairListWorkspace, 
        temporaryAOPairListWorkspace, 
        &pair_list));

    /* The AO pair list parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_AOPAIRLIST_PARAMETERS, 
        pair_list_parameters));

    /* The temporary workspace can be freed. */
    freeWorkspace(temporaryAOPairListWorkspace);

    /******************************************/
    /* cuEST One-Electron Integral Plan Setup */
    /******************************************/

    /* Declare the one-electron integral plan handle. */
    cuestOEIntPlan_t oeint_plan;

    /* Declare and create the one-electron integral plan parameter handle. */
    cuestOEIntPlanParameters_t oeint_plan_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_OEINTPLAN_PARAMETERS, 
        &oeint_plan_parameters));

    /* Determine how large the workspace must be to form and store the one-electron integral plan. */
    checkCuestErrors(cuestOEIntPlanCreateWorkspaceQuery(
        handle, 
        basis, 
        pair_list, 
        oeint_plan_parameters, 
        persistentWorkspaceDescriptor, 
        temporaryWorkspaceDescriptor, 
        &oeint_plan));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentOEIntPlanWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryOEIntPlanWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Create the one-electron integral plan. */
    checkCuestErrors(cuestOEIntPlanCreate(
        handle, 
        basis, 
        pair_list, 
        oeint_plan_parameters, 
        persistentOEIntPlanWorkspace, 
        temporaryOEIntPlanWorkspace, 
        &oeint_plan));

    /* The one-electron integral plan parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_OEINTPLAN_PARAMETERS, 
        oeint_plan_parameters));

    /* The temporary workspace can be freed. */
    freeWorkspace(temporaryOEIntPlanWorkspace);

    /**********************************/
    /* Compute One-Electron Integrals */
    /**********************************/

    /* Query the AO basis for the number of basis functions */
    uint64_t nao = 0;
    checkCuestErrors(cuestQuery(
        handle, 
        CUEST_AOBASIS, 
        basis, 
        CUEST_AOBASIS_NUM_AO,        
        &nao,        
        sizeof(uint64_t)));

    /* Allocate space for S, T, and V matrices */
    double *d_S, *d_T, *d_V;
    if (cudaMalloc((void**) &d_S, nao * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void**) &d_T, nao * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void**) &d_V, nao * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }

    /* Declare and create the overlap compute parameter handle. */
    cuestOverlapComputeParameters_t overlap_compute_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_OVERLAPCOMPUTE_PARAMETERS, 
        &overlap_compute_parameters));

    /* Workspace query to find the temporary space needed to form the S integrals. */
    checkCuestErrors(cuestOverlapComputeWorkspaceQuery(
        handle, 
        oeint_plan, 
        overlap_compute_parameters,
        temporaryWorkspaceDescriptor, 
        d_S));

    cuestWorkspace_t* temporarySWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Compute S Integrals */
    checkCuestErrors(cuestOverlapCompute(
        handle, 
        oeint_plan, 
        overlap_compute_parameters,
        temporarySWorkspace, 
        d_S));

    freeWorkspace(temporarySWorkspace);

    /* The overlap compute parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_OVERLAPCOMPUTE_PARAMETERS, 
        overlap_compute_parameters));

    /* Declare and create the kinetic compute parameter handle. */
    cuestKineticComputeParameters_t kinetic_compute_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_KINETICCOMPUTE_PARAMETERS, 
        &kinetic_compute_parameters));

    /* Workspace query to find the temporary space needed to form the T integrals. */
    checkCuestErrors(cuestKineticComputeWorkspaceQuery(
        handle, 
        oeint_plan, 
        kinetic_compute_parameters,
        temporaryWorkspaceDescriptor, 
        d_T));

    cuestWorkspace_t* temporaryTWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Compute T Integrals */
    checkCuestErrors(cuestKineticCompute(
        handle, 
        oeint_plan, 
        kinetic_compute_parameters,
        temporaryTWorkspace, 
        d_T));

    freeWorkspace(temporaryTWorkspace);

    /* The kinetic compute parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_KINETICCOMPUTE_PARAMETERS, 
        kinetic_compute_parameters));

    /* Declare and create the potential compute parameter handle. */
    cuestPotentialComputeParameters_t potential_compute_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_POTENTIALCOMPUTE_PARAMETERS, 
        &potential_compute_parameters));

    /* Workspace query to find the temporary space needed to form the V integrals. */
    checkCuestErrors(cuestPotentialComputeWorkspaceQuery(
        handle, 
        oeint_plan, 
        potential_compute_parameters,
        temporaryWorkspaceDescriptor, 
        numAtoms, 
        xyzData->xyzGPU, 
        xyzData->chargesGPU, 
        d_V));

    cuestWorkspace_t* temporaryVWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Compute V Integrals */
    checkCuestErrors(cuestPotentialCompute(
        handle, 
        oeint_plan, 
        potential_compute_parameters,
        temporaryVWorkspace, 
        numAtoms, 
        xyzData->xyzGPU, 
        xyzData->chargesGPU, 
        d_V));

    freeWorkspace(temporaryVWorkspace);

    /* The potential compute parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_POTENTIALCOMPUTE_PARAMETERS, 
        potential_compute_parameters));

    /* Free S, T, and V matrices */
    if (cudaFree((void*) d_S) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_T) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_V) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }

    /*****************************************/
    /* Destroy cuEST handles and free memory */
    /*****************************************/

    /* The workspace descriptors are no longer needed. */
    free(persistentWorkspaceDescriptor);
    free(temporaryWorkspaceDescriptor);

    /* Destroy the OE integral plan handle. */
    checkCuestErrors(cuestOEIntPlanDestroy(oeint_plan));
    freeWorkspace(persistentOEIntPlanWorkspace);

    /* Destroy the AO pair list handle. */
    checkCuestErrors(cuestAOPairListDestroy(pair_list));
    freeWorkspace(persistentAOPairListWorkspace);

    /* Destroy the AO basis handle. */
    checkCuestErrors(cuestAOBasisDestroy(basis));
    freeWorkspace(persistentBasisWorkspace);

    /* Destroy the cuEST handle. */
    checkCuestErrors(cuestDestroy(handle));

    /* Free the XYZ data */
    freeParsedXYZFile(xyzData);

    return 0;
}
