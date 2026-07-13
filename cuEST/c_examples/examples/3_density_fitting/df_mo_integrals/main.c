/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <math.h>

#include <cuest.h>

#include <helper_status.h>
#include <helper_workspaces.h>
#include <helper_xyz_parser.h>
#include <helper_gbs_parser.h>
#include <helper_ao_shells.h>

/*
 * This sample shows how to use the density-fitting (DF) MO integral
 * transformation routine to compute DF MO tensors in three blocks:
 *   - A_ij : auxiliary by occupied-by-occupied
 *   - A_ia : auxiliary by occupied-by-virtual
 *   - A_ab : auxiliary by virtual-by-virtual
 *
 * The DF integral plan is built using a primary AO basis and an
 * auxiliary fitting basis. Synthetic occupied/virtual coefficient
 * matrices are generated to populate the left and right coefficient
 * arrays. The AO 3-index DF tensor is then transformed to the MO
 * basis using cuestDFMOIntegralsCompute.
 */

/*
 * This is a helper function to generate synthetic data to populate
 * a matrix.
 */
void fill_matrix(double *A, uint64_t M, uint64_t N)
{
    double *Atmp = (double*) malloc(M * N * sizeof(double));
    if (!Atmp) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    /* Force this function to return the same values on mulitple calls */
    srand(0);
    for (uint64_t i=0, ij=0; i<M; i++) {
        for (uint64_t j=0; j<N; j++, ij++) {
            /* Box-Muller transform with mu=0 and stddev=1.0 */
            double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
            double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
            Atmp[ij] = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        }
    }
    cudaError_t err = cudaMemcpy(A, Atmp,  M * N * sizeof(double), cudaMemcpyHostToDevice);
    free(Atmp);
    if (err != cudaSuccess) {
        fprintf(stderr, "Host to device copy failed\n");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv)
{
    /* Check that an xyz file has been provided */
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <xyz_file_path> <primary_gbs_file_path> <auxiliary_gbs_file_path>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const char* xyzFilePath = argv[1];
    const char* primaryGBSFilePath = argv[2];
    const char* auxiliaryGBSFilePath = argv[3];

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

    /****************************************/
    /* cuEST AO Shell Setup (primary basis) */
    /****************************************/

    /* Use the AO shell helper to build the array of AO shells */
    AtomShellData_t* primaryShellData = formAOShells(handle, xyzData, primaryGBSFilePath, 1);

    /* Unpack the AtomShellData_t struct */
    uint64_t numAtoms = primaryShellData->numAtoms;
    uint64_t numPrimaryShellsTotal = primaryShellData->numShellsTotal;
    uint64_t* numPrimaryShellsPerAtom = primaryShellData->numShellsPerAtom;
    cuestAOShell_t* primaryShells = primaryShellData->shells;

    /****************************************/
    /* cuEST AO Basis Setup (primary basis) */
    /****************************************/

    /* Declare the AO basis handle. */
    cuestAOBasis_t primaryBasis;

    /* Declare the AO basis parameter handle. */
    cuestAOBasisParameters_t primaryBasisParameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_AOBASIS_PARAMETERS,
        &primaryBasisParameters));

    /* Allocate space for workspace descriptors. Will be used to determine how large the workspace needs to be. */
    cuestWorkspaceDescriptor_t* persistentWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    cuestWorkspaceDescriptor_t* temporaryWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));

    /* Make the workspace query to populate the workspace descriptors. */
    checkCuestErrors(cuestAOBasisCreateWorkspaceQuery(
        handle,
        numAtoms,
        numPrimaryShellsPerAtom,
        (const cuestAOShell_t*) primaryShells,
        primaryBasisParameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &primaryBasis));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentPrimaryBasisWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryPrimaryBasisWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

    /* Create the AO basis handle. */
    checkCuestErrors(cuestAOBasisCreate(
        handle,
        numAtoms,
        numPrimaryShellsPerAtom,
        (const cuestAOShell_t*) primaryShells,
        primaryBasisParameters,
        persistentPrimaryBasisWorkspace,
        temporaryPrimaryBasisWorkspace,
        &primaryBasis));

    /* The temporary workspace is no longer needed. */
    freeWorkspace(temporaryPrimaryBasisWorkspace);

    /* The AO basis parameters are no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_AOBASIS_PARAMETERS,
        primaryBasisParameters));

    /* Free all the shell data */
    freeAOShellData(primaryShellData);

    /******************************************/
    /* cuEST AO Shell Setup (auxiliary basis) */
    /******************************************/

    /* Use the AO shell helper to build the array of AO shells */
    AtomShellData_t* auxiliaryShellData = formAOShells(handle, xyzData, auxiliaryGBSFilePath, 1);

    /* Unpack the AtomShellData_t struct */
    uint64_t numAuxiliaryShellsTotal = auxiliaryShellData->numShellsTotal;
    uint64_t* numAuxiliaryShellsPerAtom = auxiliaryShellData->numShellsPerAtom;
    cuestAOShell_t* auxiliaryShells = auxiliaryShellData->shells;

    /******************************************/
    /* cuEST AO Basis Setup (auxiliary basis) */
    /******************************************/

    /* Declare the AO basis handle. */
    cuestAOBasis_t auxiliaryBasis;

    /* Declare the AO basis parameter handle. */
    cuestAOBasisParameters_t auxiliaryBasisParameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_AOBASIS_PARAMETERS,
        &auxiliaryBasisParameters));

    /* Make the workspace query to populate the workspace descriptors. */
    checkCuestErrors(cuestAOBasisCreateWorkspaceQuery(
        handle,
        numAtoms,
        numAuxiliaryShellsPerAtom,
        (const cuestAOShell_t*) auxiliaryShells,
        auxiliaryBasisParameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &auxiliaryBasis));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentAuxiliaryBasisWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryAuxiliaryBasisWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

    /* Create the AO basis handle. */
    checkCuestErrors(cuestAOBasisCreate(
        handle,
        numAtoms,
        numAuxiliaryShellsPerAtom,
        (const cuestAOShell_t*) auxiliaryShells,
        auxiliaryBasisParameters,
        persistentAuxiliaryBasisWorkspace,
        temporaryAuxiliaryBasisWorkspace,
        &auxiliaryBasis));

    /* The temporary workspace is no longer needed. */
    freeWorkspace(temporaryAuxiliaryBasisWorkspace);

    /* The AO basis parameters are no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_AOBASIS_PARAMETERS,
        auxiliaryBasisParameters));

    /* Free all the shell data */
    freeAOShellData(auxiliaryShellData);

    /****************************/
    /* cuEST AO Pair List Setup */
    /****************************/

    /* Declare the AO pair list handle. */
    cuestAOPairList_t pair_list;

    /* Declare the AO pair list parameter handle. */
    cuestAOPairListParameters_t pair_list_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_AOPAIRLIST_PARAMETERS,
        &pair_list_parameters));

    /* Determine how large the workspace must be to form and store the pair list. */
    checkCuestErrors(cuestAOPairListCreateWorkspaceQuery(
        handle,
        primaryBasis,
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
        primaryBasis,
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

    /********************************/
    /* cuEST DF Integral Plan Setup */
    /********************************/

    /* Declare the DF integral plan handle. */
    cuestDFIntPlan_t dfint_plan;

    /* Declare the DF integral plan parameter handle. */
    cuestDFIntPlanParameters_t dfint_plan_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_DFINTPLAN_PARAMETERS,
        &dfint_plan_parameters));

    /*
     * The default behavior of the DFIntPlan is to cache the 3-index integrals so
     * they can be read from disk every transform. A more compute-intensive but
     * more memory-efficient approach is to recompute the 3-index integrals on the fly.
     * This can be achieved by configuring the dfint_plan_parameters with the
     * CUEST_DFINTPLAN_PARAMETERS_THREE_INDEX_INTEGRAL_DIRECT attribute set to 1.
     */

    /* Determine how large the workspace must be to form and store the DF integral plan. */
    checkCuestErrors(cuestDFIntPlanCreateWorkspaceQuery(
        handle,
        primaryBasis,
        auxiliaryBasis,
        pair_list,
        dfint_plan_parameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &dfint_plan));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentDFIntPlanWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryDFIntPlanWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

    /* Create the DF integral plan. */
    checkCuestErrors(cuestDFIntPlanCreate(
        handle,
        primaryBasis,
        auxiliaryBasis,
        pair_list,
        dfint_plan_parameters,
        persistentDFIntPlanWorkspace,
        temporaryDFIntPlanWorkspace,
        &dfint_plan));

    /* The DF integral plan parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_DFINTPLAN_PARAMETERS,
        dfint_plan_parameters));

    /* The temporary workspace can be freed. */
    freeWorkspace(temporaryDFIntPlanWorkspace);

    /************************************/
    /* DF MO Integral Transformation    */
    /************************************/

    /* Query the AO basis for the number of basis functions */
    uint64_t nao = 0;
    checkCuestErrors(cuestQuery(
        handle,
        CUEST_AOBASIS,
        primaryBasis,
        CUEST_AOBASIS_NUM_AO,
        &nao,
        sizeof(uint64_t)));

    /* Query the auxiliary basis for the number of basis functions */
    uint64_t naux = 0;
    checkCuestErrors(cuestQuery(
        handle,
        CUEST_AOBASIS,
        auxiliaryBasis,
        CUEST_AOBASIS_NUM_AO,
        &naux,
        sizeof(uint64_t)));

    /* Get nocc for a neutral molecule */
    uint64_t nocc = 0;
    for (size_t i=0; i<xyzData->numAtoms; i++) {
        nocc += (uint64_t) (-1.0 * xyzData->chargesCPU[i]);
    }
    nocc = nocc / 2;

    /* Define nvir as the remaining orbitals for this example.
     * In practice, this comes from a separate SCF/MO calculation.
     */
    uint64_t nvir = (nao > nocc) ? (nao - nocc) : 0;
    if (nvir == 0) {
        fprintf(stderr, "No virtual orbitals available in this simple example.\n");
        exit(EXIT_FAILURE);
    }

    /* Number of coefficient-matrix pairs:
     * 0 : A_ij (left = occ, right = occ)
     * 1 : A_ia (left = occ, right = vir)
     * 2 : A_ab (left = vir, right = vir)
     */
    uint64_t numCoefficientMatrices = 3;

    /* Host arrays for numLeftOrbitals and numRightOrbitals */
    uint64_t numLeftOrbitals[3];
    uint64_t numRightOrbitals[3];

    numLeftOrbitals[0]  = nocc; /* A_ij : i */
    numRightOrbitals[0] = nocc; /* A_ij : j */

    numLeftOrbitals[1]  = nocc; /* A_ia : i */
    numRightOrbitals[1] = nvir; /* A_ia : a */

    numLeftOrbitals[2]  = nvir; /* A_ab : a */
    numRightOrbitals[2] = nvir; /* A_ab : b */

    /* Total number of left/right rows across all matrices */
    uint64_t totalLeftRows  = numLeftOrbitals[0]  + numLeftOrbitals[1]  + numLeftOrbitals[2];
    uint64_t totalRightRows = numRightOrbitals[0] + numRightOrbitals[1] + numRightOrbitals[2];

    /* Allocate space for concatenated left and right coefficient matrices */
    double *d_Cleft, *d_Cright;
    if (cudaMalloc((void**) &d_Cleft,  totalLeftRows  * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer for left coefficients\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void**) &d_Cright, totalRightRows * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer for right coefficients\n");
        exit(EXIT_FAILURE);
    }

    /*
     * Fill with random numbers. In real use cases, these are occupied
     * and virtual molecular orbital coefficients.
     */
    fill_matrix(d_Cleft,  totalLeftRows,  nao);
    fill_matrix(d_Cright, totalRightRows, nao);

    /*
     * Total output size: naux * sum_i(numLeftOrbitals[i] * numRightOrbitals[i])
     */
    uint64_t totalBlockSize = 0;
    for (uint64_t k = 0; k < numCoefficientMatrices; ++k) {
        totalBlockSize += numLeftOrbitals[k] * numRightOrbitals[k];
    }

    double* d_Tensors;
    if (cudaMalloc((void**) &d_Tensors, naux * totalBlockSize * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer for DF MO tensors\n");
        exit(EXIT_FAILURE);
    }

    /*
     * The DF MO integral transformation can benefit from a large temporary
     * workspace. Here, a workspace descriptor is used to set an upper limit
     * for the size of an internal scratch buffer. As a general rule, 2 GB
     * is a reasonable default value for this space.
     */

    cuestDFMOIntegralsComputeParameters_t dfmo_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS,
        &dfmo_parameters));

    cuestWorkspaceDescriptor_t* variableBufferSize = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    variableBufferSize->hostBufferSizeInBytes = 0;
    variableBufferSize->deviceBufferSizeInBytes = 2000000000;

    /* Workspace query to find the temporary space needed to form the MO DF integrals. */
    checkCuestErrors(cuestDFMOIntegralsComputeWorkspaceQuery(
        handle,
        dfint_plan,
        dfmo_parameters,
        variableBufferSize,
        temporaryWorkspaceDescriptor,
        numCoefficientMatrices,
        numLeftOrbitals,
        numRightOrbitals,
        d_Cleft,
        d_Cright,
        d_Tensors));

    cuestWorkspace_t* temporaryDFMOWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Compute DF MO integrals (A_ij, A_ia, A_ab) */
    checkCuestErrors(cuestDFMOIntegralsCompute(
        handle,
        dfint_plan,
        dfmo_parameters,
        variableBufferSize,
        temporaryDFMOWorkspace,
        numCoefficientMatrices,
        numLeftOrbitals,
        numRightOrbitals,
        d_Cleft,
        d_Cright,
        d_Tensors));

    free(variableBufferSize);
    freeWorkspace(temporaryDFMOWorkspace);

    /* The DF MO integrals compute parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS,
        dfmo_parameters));

    /* Free coefficient and tensor buffers */
    if (cudaFree((void*) d_Cleft) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_Cright) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_Tensors) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }

    /*****************************************/
    /* Destroy cuEST handles and free memory */
    /*****************************************/

    /* The workspace descriptors are no longer needed. */
    free(persistentWorkspaceDescriptor);
    free(temporaryWorkspaceDescriptor);

    /* Destroy the DF integral plan handle. */
    checkCuestErrors(cuestDFIntPlanDestroy(dfint_plan));
    freeWorkspace(persistentDFIntPlanWorkspace);

    /* Destroy the AO pair list handle. */
    checkCuestErrors(cuestAOPairListDestroy(pair_list));
    freeWorkspace(persistentAOPairListWorkspace);

    /* Destroy the AO basis handle (auxiliary basis). */
    checkCuestErrors(cuestAOBasisDestroy(auxiliaryBasis));
    freeWorkspace(persistentAuxiliaryBasisWorkspace);

    /* Destroy the AO basis handle (primary basis). */
    checkCuestErrors(cuestAOBasisDestroy(primaryBasis));
    freeWorkspace(persistentPrimaryBasisWorkspace);

    /* Destroy the cuEST handle. */
    checkCuestErrors(cuestDestroy(handle));

    /* Free the XYZ data */
    freeParsedXYZFile(xyzData);

    return 0;
}

