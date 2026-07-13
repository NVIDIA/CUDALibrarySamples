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
 * This sample shows how to use the "Core" density-fitting algorithm
 * to evaluate exchange matrices. The use of density-fitting
 * requires an additional basis set file be provided -- one for a fitting
 * or auxiliary basis. If a customized fitting set that has been optimized
 * for use with the primary basis is not available, the def2-universal-JKFIT
 * basis is recommended.
 *
 * The nonsymmetric exchange matrix takes a left coefficient matrix and a
 * set of right coefficient matrices as input.
 *
 * Since molecular orbital coefficients are not available, synthetic data
 * is generated to populate these matrices. The number of occupied orbitals
 * is calculated from the sum of atomic charges.
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

    /************************/
    /* Compute K Matrices */
    /************************/

    /* Query the AO basis for the number of basis functions */
    uint64_t nao = 0;
    checkCuestErrors(cuestQuery(
        handle,
        CUEST_AOBASIS,
        primaryBasis,
        CUEST_AOBASIS_NUM_AO,
        &nao,
        sizeof(uint64_t)));

    /* Get nocc for a neutral molecule */
    uint64_t nocc = 0;
    for (size_t i=0; i<xyzData->numAtoms; i++) {
        nocc += (uint64_t) (-1.0 * xyzData->chargesCPU[i]);
    }
    nocc = nocc / 2;

    /* Number of right coefficient matrices */
    uint64_t numCoefficientMatrices = 4;

    /* Allocate space for K matrices */
    double *d_K;
    if (cudaMalloc((void**) &d_K, numCoefficientMatrices * nao * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate space for Cleft and Cright matrices */
    double *d_Cleft, *d_Cright;
    if (cudaMalloc((void**) &d_Cleft, nocc * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void**) &d_Cright, numCoefficientMatrices * nocc * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }

    /*
     * Fill with random numbers. In real use cases, these are occupied
     * molecular orbital coefficients.
     */
    fill_matrix(d_Cleft, nocc, nao);
    fill_matrix(d_Cright, numCoefficientMatrices * nocc, nao);

    /*
     * The DF-K algorithm implemented in cuEST can benefit from very large temporary spaces. However, the
     * calculation can be performed in smaller, more managable workspaces. Here, a workspace descriptor
     * is used to set an upper limit for the size of certain intermediates used to form the K matrix. As
     * a general rule, 2 GB is a reasonable, default value for the size of this space. However, larger
     * spaces may improve performance for larger molecules.
     */

    /* Declare the DF exchange compute parameter handle. */
    cuestDFNonsymmetricExchangeComputeParameters_t dfk_compute_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_DFNONSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
        &dfk_compute_parameters));

    cuestWorkspaceDescriptor_t* variableBufferSize = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    variableBufferSize->hostBufferSizeInBytes = 0;
    variableBufferSize->deviceBufferSizeInBytes = 2000000000;

    /* Workspace query to find the temporary space needed to form the K matrices. */
    checkCuestErrors(cuestDFNonsymmetricExchangeComputeWorkspaceQuery(
        handle,
        dfint_plan,
        dfk_compute_parameters,
        variableBufferSize,
        temporaryWorkspaceDescriptor,
        numCoefficientMatrices,
        nocc,
        d_Cleft,
        d_Cright,
        d_K));

    cuestWorkspace_t* temporaryDFKWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Compute K Matrices */
    checkCuestErrors(cuestDFNonsymmetricExchangeCompute(
        handle,
        dfint_plan,
        dfk_compute_parameters,
        variableBufferSize,
        temporaryDFKWorkspace,
        numCoefficientMatrices,
        nocc,
        d_Cleft,
        d_Cright,
        d_K));

    free(variableBufferSize);
    freeWorkspace(temporaryDFKWorkspace);

    /* The DF exchange compute parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_DFNONSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
        dfk_compute_parameters));

    /* Free K, Cleft, and Cright matrices */
    if (cudaFree((void*) d_K) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_Cleft) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_Cright) != cudaSuccess) {
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

