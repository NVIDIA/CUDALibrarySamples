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
#include <helper_grid.h>
#include <helper_gbs_parser.h>
#include <helper_ao_shells.h>

/*
 * This example shows how to evaluate the derivative of the exchange 
 * correlation energy. This depends on an AO basis set, a molecular 
 * grid (here, a 75, 302 grid), and coefficient/density matrices. In 
 * this sample, synthetic data is used for those matrices. In real 
 * applications of cuEST, these matrices should be obtained during an 
 * SCF procedure.
 *
 * These derivatives account for motion of the basis function centers,
 * grid centers and Becke weights.
 *
 * Examples of both RKS and UKS usage are provided. In the UKS case, both
 * alpha and beta densities/coefficients must be provided. 
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
    /* Check that an xyz file and basis (gbs file) has been provided */
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
    cuestWorkspace_t* persistentBasisWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryBasisWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

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
    cuestWorkspace_t* persistentGridWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryGridWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

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

    /********************************/
    /* cuEST XC Integral Plan Setup */
    /********************************/

    /* Declare the XC integral plan handle. */
    cuestXCIntPlan_t xcIntPlan;

    /* Declare and create the XC integral plan parameters handle. */
    cuestXCIntPlanParameters_t xcIntPlanParameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_XCINTPLAN_PARAMETERS, 
        &xcIntPlanParameters));

    /* Determine the workspaces required to construct the XC integral plan. */
    checkCuestErrors(cuestXCIntPlanCreateWorkspaceQuery(
        handle, 
        basis, 
        molecularGrid,
        CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE,
        xcIntPlanParameters, 
        persistentWorkspaceDescriptor, 
        temporaryWorkspaceDescriptor, 
        &xcIntPlan));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentXCIntPlanWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryXCIntPlanWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

    /* Create the XC integral plan for a PBE functional. */
    checkCuestErrors(cuestXCIntPlanCreate(
        handle, 
        basis, 
        molecularGrid,
        CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE,
        xcIntPlanParameters, 
        persistentXCIntPlanWorkspace, 
        temporaryXCIntPlanWorkspace, 
        &xcIntPlan));

    /* The XC integral plan paramaters can be destroyed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_XCINTPLAN_PARAMETERS, 
        xcIntPlanParameters));

    /* The temporary workspace is no longer needed. */
    freeWorkspace(temporaryXCIntPlanWorkspace);

    /********************************************/
    /* cuEST Compute the RKS XC Potential (Vxc) */
    /********************************************/

    /* Query the AO basis for the number of basis functions */
    uint64_t nao = 0;
    checkCuestErrors(cuestQuery(
        handle, 
        CUEST_AOBASIS, 
        basis, 
        CUEST_AOBASIS_NUM_AO,        
        &nao,        
        sizeof(uint64_t)));

    /* Get nocc for a neutral molecule */
    uint64_t nocc = 0;
    for (size_t i=0; i<xyzData->numAtoms; i++) {
        nocc += (uint64_t) (-1.0 * xyzData->chargesCPU[i]);
    }
    nocc = nocc / 2;

    /* Exchange correlation energy */
    double Exc = 0.0;

    /* Allocate space for Vxc matrix */
    double *d_grad;
    if (cudaMalloc((void**) &d_grad, numAtoms * 3 * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate space for Cocc matrix */
    double *d_Cocc;
    if (cudaMalloc((void**) &d_Cocc, nocc * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    
    /* 
     * Fill with random numbers. In real use cases, these are density
     * matrices and occupied molecular orbital coefficients.
     */
    fill_matrix(d_Cocc, nocc, nao);

    /* 
     * The memory usage during Vxc evaluation can be controlled be specifying a maximum workspace size. 2 GB is a reasonable workspace size,
     * when possible, larger workspaces might provide additional performance benefits.
     */

    /* Declare and create the RKS potential parameters handle. */
    cuestXCDerivativeRKSComputeParameters_t rks_derivative_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS, 
        &rks_derivative_parameters));

    cuestWorkspaceDescriptor_t* variableBufferSize = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    variableBufferSize->hostBufferSizeInBytes = 0;
    variableBufferSize->deviceBufferSizeInBytes = 2000000000;

    /* Workspace query to find the temporary space needed to form Exc and Vxc. */
    checkCuestErrors(cuestXCDerivativeRKSComputeWorkspaceQuery(
        handle,
        xcIntPlan, 
        rks_derivative_parameters,
        variableBufferSize, 
        temporaryWorkspaceDescriptor, 
        nocc,
        d_Cocc,
        d_grad));

    cuestWorkspace_t* temporaryVxcWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Compute Exc and Vxc */
    checkCuestErrors(cuestXCDerivativeRKSCompute(
        handle,
        xcIntPlan, 
        rks_derivative_parameters,
        variableBufferSize, 
        temporaryVxcWorkspace, 
        nocc,
        d_Cocc,
        d_grad));

    free(variableBufferSize);
    freeWorkspace(temporaryVxcWorkspace);

    /* The RKS potential parameters can be destroyed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS,
        rks_derivative_parameters));

    /* Free gradient and Cocc matrices */
    if (cudaFree((void*) d_grad) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_Cocc) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }

    /********************************************/
    /* cuEST Compute the UKS XC Potential (Vxc) */
    /********************************************/

    /* Get noccA/noccB for a doublet cation. */
    uint64_t noccA = nocc;
    uint64_t noccB = nocc-1;

    /* Exchange correlation energy */
    double Exc_UKS = 0.0;

    /* Allocate space for Vxc matrix */
    double *d_grad_UKS;
    if (cudaMalloc((void**) &d_grad_UKS, numAtoms * 3 * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate space for Cocc matrices */
    double *d_CoccA, *d_CoccB;
    if (cudaMalloc((void**) &d_CoccA, noccA * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void**) &d_CoccB, noccB * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    
    /* 
     * Fill with random numbers. In real use cases, these are 
     * occupied molecular orbital coefficients.
     */
    fill_matrix(d_CoccA, noccA, nao);
    fill_matrix(d_CoccB, noccB, nao);

    /* 
     * The memory usage during Vxc evaluation can be controlled be specifying a maximum workspace size. 2 GB is a reasonable workspace size,
     * when possible, larger workspaces might provide additional performance benefits.
     */

    /* Declare and create the UKS potential parameters handle. */
    cuestXCDerivativeUKSComputeParameters_t uks_derivative_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_XCDERIVATIVEUKSCOMPUTE_PARAMETERS, 
        &uks_derivative_parameters));

    variableBufferSize = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    variableBufferSize->hostBufferSizeInBytes = 0;
    variableBufferSize->deviceBufferSizeInBytes = 2000000000;

    /* Workspace query to find the temporary space needed to form Exc and Vxc. */
    checkCuestErrors(cuestXCDerivativeUKSComputeWorkspaceQuery(
        handle,
        xcIntPlan, 
        uks_derivative_parameters,
        variableBufferSize, 
        temporaryWorkspaceDescriptor, 
        noccA,
        noccB,
        d_CoccA,
        d_CoccB,
        d_grad_UKS));

    cuestWorkspace_t* temporaryVxcUKSWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Compute Exc and Vxc */
    checkCuestErrors(cuestXCDerivativeUKSCompute(
        handle,
        xcIntPlan, 
        uks_derivative_parameters,
        variableBufferSize, 
        temporaryVxcUKSWorkspace, 
        noccA,
        noccB,
        d_CoccA,
        d_CoccB,
        d_grad_UKS));

    free(variableBufferSize);
    freeWorkspace(temporaryVxcUKSWorkspace);

    /* The UKS potential parameters can be destroyed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_XCDERIVATIVEUKSCOMPUTE_PARAMETERS,
        uks_derivative_parameters));

    /* Free gradient and Cocc matrices */
    if (cudaFree((void*) d_grad_UKS) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_CoccA) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_CoccB) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }

    /*****************************************/
    /* Destroy cuEST handles and free memory */
    /*****************************************/

    /* The workspace descriptors are no longer needed. */
    free(persistentWorkspaceDescriptor);
    free(temporaryWorkspaceDescriptor);

    /* Destroy the XC integral plan handle. */
    checkCuestErrors(cuestXCIntPlanDestroy(xcIntPlan));
    freeWorkspace(persistentXCIntPlanWorkspace);

    /* Destroy the molecular grid handle. */
    checkCuestErrors(cuestMolecularGridDestroy(molecularGrid));
    freeWorkspace(persistentGridWorkspace);

    /* Destroy the AO basis handle. */
    checkCuestErrors(cuestAOBasisDestroy(basis));
    freeWorkspace(persistentBasisWorkspace);

    /* Destroy the cuEST handle. */
    checkCuestErrors(cuestDestroy(handle));

    /* Free the XYZ data */
    freeParsedXYZFile(xyzData);

    return 0;
}
