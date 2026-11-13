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

#include <cuest.h>

#include <helper_status.h>
#include <helper_workspaces.h>
#include <helper_xyz_parser.h>
#include <helper_gbs_parser.h>
#include <helper_ao_shells.h>

/*
 * This sample shows how to use the direct (integral-direct, 4-center)
 * algorithm to evaluate nuclear gradients of the Coulomb and exchange
 * energies. In contrast to density fitting, no auxiliary basis set is
 * required -- the 4-center integral derivatives are evaluated on the fly
 * and contracted with the density.
 *
 * For each density matrix D in the batch, the gradient routines compute
 * the derivative of the Coulomb energy E_J = sum_pq D_pq J_pq or of the
 * exchange energy E_K = sum_pq D_pq K_pq at fixed density coefficients
 * (pure integral-derivative contractions; the density-response and
 * overlap terms of an SCF gradient are assembled elsewhere). A batch of
 * density matrices can be processed in a single call; the matrices are
 * stacked contiguously in memory and the number of matrices is passed as
 * numDensityMatrices. One gradient (numAtoms x 3) is produced per density
 * matrix; the output buffers are overwritten, not accumulated. The
 * symmetric exchange routine used here requires symmetric density
 * matrices.
 *
 * The exchange operator is a property of the direct integral plan. The
 * fractions of full-range (1/r) and long-range-corrected (erf(wr)/r)
 * exchange can be configured through the plan parameters; the defaults
 * used here request full-range exact exchange.
 *
 * Since density matrices are not available, synthetic data is generated
 * to populate these matrices.
 */

/*
 * This is a helper function to generate synthetic data to populate
 * a density matrix.
 */
void fill_symmetric_matrix(double *A, uint64_t N)
{
    double *Atmp = (double*) malloc(N * N * sizeof(double));
    if (!Atmp) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    /* Force this function to return the same values on mulitple calls */
    srand(0);
    for (uint64_t i=0; i<N; i++) {
        for (uint64_t j=0; j<=i; j++) {
            /* Box-Muller transform with mu=0 and stddev=1.0 */
            double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
            double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
            Atmp[i*N+j] = Atmp[j*N+i] = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        }
    }
    cudaError_t err = cudaMemcpy(A, Atmp,  N * N * sizeof(double), cudaMemcpyHostToDevice);
    free(Atmp);
    if (err != cudaSuccess) {
        fprintf(stderr, "Host to device copy failed\n");
        exit(EXIT_FAILURE);
    }
}

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
    AtomShellData_t* shellData = formAOShells(handle, xyzData, gbsFilePath, 1);

    /* Unpack the AtomShellData_t struct */
    uint64_t numAtoms = shellData->numAtoms;
    uint64_t numShellsTotal = shellData->numShellsTotal;
    uint64_t* numShellsPerAtom = shellData->numShellsPerAtom;
    cuestAOShell_t* shells = shellData->shells;

    /************************/
    /* cuEST AO Basis Setup */
    /************************/

    /* Declare the AO basis handle. */
    cuestAOBasis_t basis;

    /* Declare the AO basis parameter handle. */
    cuestAOBasisParameters_t basisParameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_AOBASIS_PARAMETERS,
        &basisParameters));

    /* Allocate space for workspace descriptors. Will be used to determine how large the workspace needs to be. */
    cuestWorkspaceDescriptor_t* persistentWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    cuestWorkspaceDescriptor_t* temporaryWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));

    /* Make the workspace query to populate the workspace descriptors. */
    checkCuestErrors(cuestAOBasisCreateWorkspaceQuery(
        handle,
        numAtoms,
        numShellsPerAtom,
        (const cuestAOShell_t*) shells,
        basisParameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &basis));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentBasisWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryBasisWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Create the AO basis handle. */
    checkCuestErrors(cuestAOBasisCreate(
        handle,
        numAtoms,
        numShellsPerAtom,
        (const cuestAOShell_t*) shells,
        basisParameters,
        persistentBasisWorkspace,
        temporaryBasisWorkspace,
        &basis));

    /* The temporary workspace is no longer needed. */
    freeWorkspace(temporaryBasisWorkspace);

    /* The AO basis parameters are no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_AOBASIS_PARAMETERS,
        basisParameters));

    /* Free all the shell data */
    freeAOShellData(shellData);

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
        basis,
        numAtoms,
        xyzData->xyzCPU,
        1.0e-14,
        pair_list_parameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &pair_list));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentAOPairListWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryAOPairListWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

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

    /************************************/
    /* cuEST Direct Integral Plan Setup */
    /************************************/

    /* Declare the direct integral plan handle. */
    cuestDirectIntPlan_t directint_plan;

    /*
     * Declare the direct integral plan parameter handle. The exchange operator is
     * defined when the plan is created. The defaults
     * (CUEST_DIRECTINTPLAN_PARAMETERS_EXCHANGE_FRACTION = 1.0,
     * CUEST_DIRECTINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION = 0.0) request
     * full-range exact exchange. Range-separated exchange operators can be
     * requested by configuring these parameters together with
     * CUEST_DIRECTINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA.
     */
    cuestDirectIntPlanParameters_t directint_plan_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_DIRECTINTPLAN_PARAMETERS,
        &directint_plan_parameters));

    /* Determine how large the workspace must be to form and store the direct integral plan. */
    checkCuestErrors(cuestDirectIntPlanCreateWorkspaceQuery(
        handle,
        basis,
        pair_list,
        directint_plan_parameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &directint_plan));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentDirectIntPlanWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryDirectIntPlanWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Create the direct integral plan. */
    checkCuestErrors(cuestDirectIntPlanCreate(
        handle,
        basis,
        pair_list,
        directint_plan_parameters,
        persistentDirectIntPlanWorkspace,
        temporaryDirectIntPlanWorkspace,
        &directint_plan));

    /* The direct integral plan parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_DIRECTINTPLAN_PARAMETERS,
        directint_plan_parameters));

    /* The temporary workspace can be freed. */
    freeWorkspace(temporaryDirectIntPlanWorkspace);

    /*************************/
    /* Compute J/K Gradients */
    /*************************/

    /* Query the AO basis for the number of basis functions */
    uint64_t nao = 0;
    checkCuestErrors(cuestQuery(
        handle,
        CUEST_AOBASIS,
        basis,
        CUEST_AOBASIS_NUM_AO,
        &nao,
        sizeof(uint64_t)));

    /*
     * A single density matrix is used here (as in RHF/RKS). To process a
     * batch, stack the density matrices contiguously and set
     * numDensityMatrices accordingly; one gradient is produced per density.
     */
    uint64_t numDensityMatrices = 1;

    /* Allocate space for the J and K gradients */
    double *d_J_grad, *d_K_grad;
    if (cudaMalloc((void**) &d_J_grad, numDensityMatrices * numAtoms * 3 * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void**) &d_K_grad, numDensityMatrices * numAtoms * 3 * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate space for the density matrix */
    double *d_D;
    if (cudaMalloc((void**) &d_D, numDensityMatrices * nao * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }

    /*
     * Fill with random numbers. In real use cases, this is a density
     * matrix obtained during an SCF procedure.
     */
    fill_symmetric_matrix(d_D, nao);

    /*
     * Declare the direct Coulomb derivative compute parameter handle. The
     * density-weighted Cauchy-Schwarz screening threshold can be adjusted through
     * CUEST_DIRECTCOULOMBDERIVATIVECOMPUTE_PARAMETERS_DENSITY_SCHWARZ_THRESHOLD (default: 1.0e-13).
     */
    cuestDirectCoulombDerivativeComputeParameters_t directj_grad_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_DIRECTCOULOMBDERIVATIVECOMPUTE_PARAMETERS,
        &directj_grad_parameters));

    /* Workspace query to find the temporary space needed to form the J gradient. */
    checkCuestErrors(cuestDirectCoulombDerivativeComputeWorkspaceQuery(
        handle,
        directint_plan,
        directj_grad_parameters,
        temporaryWorkspaceDescriptor,
        numDensityMatrices,
        d_D,
        d_J_grad));

    cuestWorkspace_t* temporaryDirectJgradWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Compute J gradient */
    checkCuestErrors(cuestDirectCoulombDerivativeCompute(
        handle,
        directint_plan,
        directj_grad_parameters,
        temporaryDirectJgradWorkspace,
        numDensityMatrices,
        d_D,
        d_J_grad));

    freeWorkspace(temporaryDirectJgradWorkspace);

    /* The direct Coulomb derivative compute parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_DIRECTCOULOMBDERIVATIVECOMPUTE_PARAMETERS,
        directj_grad_parameters));

    /*
     * Declare the direct exchange derivative compute parameter handle. As for
     * the Coulomb gradient, the screening threshold can be adjusted through
     * CUEST_DIRECTSYMMETRICEXCHANGEDERIVATIVECOMPUTE_PARAMETERS_DENSITY_SCHWARZ_THRESHOLD.
     */
    cuestDirectSymmetricExchangeDerivativeComputeParameters_t directk_grad_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_DIRECTSYMMETRICEXCHANGEDERIVATIVECOMPUTE_PARAMETERS,
        &directk_grad_parameters));

    /* Workspace query to find the temporary space needed to form the K gradient. */
    checkCuestErrors(cuestDirectSymmetricExchangeDerivativeComputeWorkspaceQuery(
        handle,
        directint_plan,
        directk_grad_parameters,
        temporaryWorkspaceDescriptor,
        numDensityMatrices,
        d_D,
        d_K_grad));

    cuestWorkspace_t* temporaryDirectKgradWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Compute K gradient */
    checkCuestErrors(cuestDirectSymmetricExchangeDerivativeCompute(
        handle,
        directint_plan,
        directk_grad_parameters,
        temporaryDirectKgradWorkspace,
        numDensityMatrices,
        d_D,
        d_K_grad));

    freeWorkspace(temporaryDirectKgradWorkspace);

    /* The direct exchange derivative compute parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_DIRECTSYMMETRICEXCHANGEDERIVATIVECOMPUTE_PARAMETERS,
        directk_grad_parameters));

    /* Free the J/K gradients and the D matrix */
    if (cudaFree((void*) d_J_grad) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_K_grad) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_D) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }

    /*****************************************/
    /* Destroy cuEST handles and free memory */
    /*****************************************/

    /* The workspace descriptors are no longer needed. */
    free(persistentWorkspaceDescriptor);
    free(temporaryWorkspaceDescriptor);

    /* Destroy the direct integral plan handle. */
    checkCuestErrors(cuestDirectIntPlanDestroy(directint_plan));
    freeWorkspace(persistentDirectIntPlanWorkspace);

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
