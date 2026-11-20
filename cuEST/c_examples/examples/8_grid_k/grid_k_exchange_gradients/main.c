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
#include <helper_grid.h>
#include <helper_gbs_parser.h>
#include <helper_ao_shells.h>

/*
 * This sample shows how to use the grid-based exchange (Grid-K) algorithm
 * to evaluate nuclear gradients of the exchange energy. In this
 * pseudospectral approach, the exchange matrix and its energy are
 * assembled by numerical quadrature on a molecular grid (here, an
 * unpruned (75, 302) grid) instead of from 4-center integrals, so a
 * molecular grid must be provided in addition to the AO basis and pair
 * list.
 *
 * For each symmetric density matrix D in the batch, the gradient routine
 * computes the derivative of the Grid-K exchange energy
 * E_K = sum_pq D_pq K_pq at fixed density coefficients. A batch of
 * density matrices can be processed in a single call; the matrices are
 * stacked contiguously in memory and the number of matrices is passed as
 * numDensityMatrices. One gradient (numAtoms x 3) is produced per density
 * matrix; the output buffer is overwritten, not accumulated.
 *
 * The exchange operator is a property of the Grid-K integral plan. The
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

    /* Create the one-electron integral plan required by the Grid-K plan. */
    cuestOEIntPlan_t oeint_plan;

    cuestOEIntPlanParameters_t oeint_plan_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_OEINTPLAN_PARAMETERS,
        &oeint_plan_parameters));

    checkCuestErrors(cuestOEIntPlanCreateWorkspaceQuery(
        handle,
        basis,
        pair_list,
        oeint_plan_parameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &oeint_plan));

    cuestWorkspace_t* persistentOEIntPlanWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryOEIntPlanWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestOEIntPlanCreate(
        handle,
        basis,
        pair_list,
        oeint_plan_parameters,
        persistentOEIntPlanWorkspace,
        temporaryOEIntPlanWorkspace,
        &oeint_plan));

    checkCuestErrors(cuestParametersDestroy(
        CUEST_OEINTPLAN_PARAMETERS,
        oeint_plan_parameters));

    freeWorkspace(temporaryOEIntPlanWorkspace);

    /***************************/
    /* cuEST Atomic Grid Setup */
    /***************************/

    /* This forms an unpruned (75, 302) grid with Ahlrichs radial quadrature. */
    cuestAtomGrid_t* atomGrid = formDirectProductAtomGrid(
        handle,
        xyzData,
        75,
        302);

    /******************************/
    /* cuEST Molecular Grid Setup */
    /******************************/

    /* Declare the molecular grid handle. */
    cuestMolecularGrid_t molecularGrid;

    /* Declare and create the molecular grid parameter handle. */
    cuestMolecularGridParameters_t molecularGridParameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_MOLECULARGRID_PARAMETERS,
        &molecularGridParameters));

    /* Determine the workspaces required to construct the molecular grid. */
    checkCuestErrors(cuestMolecularGridCreateWorkspaceQuery(
        handle,
        xyzData->numAtoms,
        atomGrid,
        xyzData->xyzCPU,
        molecularGridParameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &molecularGrid));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentGridWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryGridWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Create the molecular grid. */
    checkCuestErrors(cuestMolecularGridCreate(
        handle,
        xyzData->numAtoms,
        atomGrid,
        xyzData->xyzCPU,
        molecularGridParameters,
        persistentGridWorkspace,
        temporaryGridWorkspace,
        &molecularGrid));

    /* The molecular grid parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_MOLECULARGRID_PARAMETERS,
        molecularGridParameters));

    /* The temporary workspace can be freed. */
    freeWorkspace(temporaryGridWorkspace);

    /* The atom grid is not needed once the molecular grid is created */
    for (int n=0; n<xyzData->numAtoms; n++) {
        checkCuestErrors(cuestAtomGridDestroy(atomGrid[n]));
    }
    free(atomGrid);

    /************************************/
    /* cuEST Grid-K Integral Plan Setup */
    /************************************/

    /* Declare the Grid-K integral plan handle. */
    cuestGridKIntPlan_t gridkint_plan;

    /*
     * Declare the Grid-K integral plan parameter handle. The exchange operator is
     * defined when the plan is created. The defaults
     * (CUEST_GRIDKINTPLAN_PARAMETERS_EXCHANGE_FRACTION = 1.0,
     * CUEST_GRIDKINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION = 0.0) request
     * full-range exact exchange. Range-separated exchange operators can be
     * requested by configuring these parameters together with
     * CUEST_GRIDKINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA.
     */
    cuestGridKIntPlanParameters_t gridkint_plan_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_GRIDKINTPLAN_PARAMETERS,
        &gridkint_plan_parameters));

    /*
     * Request the symmetric overlap fit of the grid collocation. This is the
     * overlap-fitting step that is part of the Compensated Grid-K procedure.
     * The exchange derivatives computed from this plan will account for the
     * response of the overlap fit.
     */
    cuestGridKIntPlanParametersFittingAlgorithm_t fitting_algorithm =
        CUEST_GRIDKINTPLAN_PARAMETERS_FITTING_ALGORITHM_OVERLAP_SYMMETRIC;
    checkCuestErrors(cuestParametersConfigure(
        CUEST_GRIDKINTPLAN_PARAMETERS,
        gridkint_plan_parameters,
        CUEST_GRIDKINTPLAN_PARAMETERS_FITTING_ALGORITHM,
        &fitting_algorithm,
        sizeof(cuestGridKIntPlanParametersFittingAlgorithm_t)));

    /*
     * The Grid-K gradient algorithm implemented in cuEST can benefit from very large temporary spaces.
     * However, the calculation can be performed in smaller, more managable workspaces. Here, a workspace
     * descriptor is used to set an upper limit for the size of certain intermediates used to form the K
     * gradient. As a general rule, 4 GB is a reasonable, default value for the size of this space. However,
     * larger spaces may improve performance for larger molecules. Note that the variableBufferSize does not
     * limit total memory usage and must remain unchanged between the workspace query and the compute call.
     */
    cuestWorkspaceDescriptor_t* variableBufferSize = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    variableBufferSize->hostBufferSizeInBytes = 0;
    variableBufferSize->deviceBufferSizeInBytes = 4000000000;

    /* Determine how large the workspace must be to form and store the Grid-K integral plan. */
    checkCuestErrors(cuestGridKIntPlanCreateWorkspaceQuery(
        handle,
        molecularGrid,
        oeint_plan,
        gridkint_plan_parameters,
        variableBufferSize,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &gridkint_plan));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentGridKIntPlanWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryGridKIntPlanWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Create the Grid-K integral plan. */
    checkCuestErrors(cuestGridKIntPlanCreate(
        handle,
        molecularGrid,
        oeint_plan,
        gridkint_plan_parameters,
        variableBufferSize,
        persistentGridKIntPlanWorkspace,
        temporaryGridKIntPlanWorkspace,
        &gridkint_plan));

    /* The Grid-K integral plan parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_GRIDKINTPLAN_PARAMETERS,
        gridkint_plan_parameters));

    /* The temporary workspace can be freed. */
    freeWorkspace(temporaryGridKIntPlanWorkspace);

    /***********************/
    /* Compute K Gradients */
    /***********************/

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

    /* Allocate space for the K gradient */
    double *d_K_grad;
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

    /* Declare the Grid-K exchange derivative compute parameter handle. */
    cuestGridKSymmetricExchangeDerivativeComputeParameters_t gridk_grad_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_GRIDKSYMMETRICEXCHANGEDERIVATIVECOMPUTE_PARAMETERS,
        &gridk_grad_parameters));

    /* Workspace query to find the temporary space needed to form the K gradient. */
    checkCuestErrors(cuestGridKSymmetricExchangeDerivativeComputeWorkspaceQuery(
        handle,
        gridkint_plan,
        gridk_grad_parameters,
        variableBufferSize,
        temporaryWorkspaceDescriptor,
        numDensityMatrices,
        d_D,
        d_K_grad));

    cuestWorkspace_t* temporaryGridKgradWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Compute K gradient */
    checkCuestErrors(cuestGridKSymmetricExchangeDerivativeCompute(
        handle,
        gridkint_plan,
        gridk_grad_parameters,
        variableBufferSize,
        temporaryGridKgradWorkspace,
        numDensityMatrices,
        d_D,
        d_K_grad));

    free(variableBufferSize);
    freeWorkspace(temporaryGridKgradWorkspace);

    /* The Grid-K exchange derivative compute parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_GRIDKSYMMETRICEXCHANGEDERIVATIVECOMPUTE_PARAMETERS,
        gridk_grad_parameters));

    /* Free the K gradient and the D matrix */
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

    /* Destroy the Grid-K integral plan handle. */
    checkCuestErrors(cuestGridKIntPlanDestroy(gridkint_plan));
    freeWorkspace(persistentGridKIntPlanWorkspace);

    /* Destroy the molecular grid handle. */
    checkCuestErrors(cuestMolecularGridDestroy(molecularGrid));
    freeWorkspace(persistentGridWorkspace);

    /* Destroy the one-electron integral plan handle. */
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
