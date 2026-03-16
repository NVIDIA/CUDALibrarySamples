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
#include <errno.h>

#include <cuest.h>

#include <helper_status.h>
#include <helper_workspaces.h>
#include <helper_xyz_parser.h>
#include <helper_gbs_parser.h>
#include <helper_ao_shells.h>
#include <helper_ecp_parser.h>

/*
 * This sample demonstrates how to compute the derivatives of the 
 * effective core potential (ECP) integrals contracted with a
 * density matrix. In cuEST, the derivative integrals cannot be
 * computed individually, but are always contracted with a density
 * (or pseudo-density) matrix. The result of this contraction is
 * stored in a number of atoms by 3 array.
 *
 * In this sample, a "real" density matrix is not availble, so a symmetric 
 * random matrix is substituted.
 *
 * The use of the effective core potential integral derivative
 * routines follows very closely with the underlying ECP integrals.
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
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <xyz_file_path> <gbs_file_path> <ecp_file_path>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const char* xyzFilePath = argv[1];
    const char* gbsFilePath = argv[2];
    const char* ecpFilePath = argv[3];

    int is_pure = 1;

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
    AtomShellData_t* shell_data = formAOShells(handle, xyzData, gbsFilePath, is_pure);

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

    /****************************************************************/
    /* cuEST Effective Core Potential Integral Shell and Atom Setup */
    /****************************************************************/

    /* Find the number of unique atoms */
    uint64_t maxUnique = (xyzData->numAtoms > 128) ? 128 : xyzData->numAtoms;
    uint64_t numUnique = 0;

    char **uniqueSymbols = (char**) malloc(maxUnique * sizeof(char*));
    if (!uniqueSymbols) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    for (uint64_t i = 0; i < xyzData->numAtoms; i++) {
        int isUnique = 1;
        for (uint64_t j = 0; j < numUnique; j++) {
            if (strcmp(xyzData->symbols[i], uniqueSymbols[j]) == 0) {
                isUnique = 0;
                break;
            }
        }
        if (isUnique) {
            uniqueSymbols[numUnique] = xyzData->symbols[i];
            numUnique++;
        }
    }

    /* Read the shells in for the unique atoms from the ECP basis set using the helper */
    ECPShellSet_t** shellList = (ECPShellSet_t**) malloc(numUnique * sizeof(ECPShellSet_t*));
    if (!shellList) {
        free(uniqueSymbols);
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }

    for (uint64_t i = 0; i < numUnique; i++) {
        shellList[i] = parseECPFileForElement(ecpFilePath, uniqueSymbols[i]);
    }

    /* Compute the total number of active ECP atoms */
    uint64_t numActiveECP = 0;
    for (uint64_t i = 0; i < xyzData->numAtoms; i++) {
        uint64_t basisIndex = 0;
        for (uint64_t j = 0; j < numUnique; j++) {
            if (strcmp(xyzData->symbols[i], uniqueSymbols[j]) == 0) {
                basisIndex = j;
                if (shellList[j]) numActiveECP++;
                break;
            }
        }
    }

    /* Generate a map from active ECP atoms to the full atom set */
    uint64_t *ecpIndices = (uint64_t *) malloc(numActiveECP * sizeof(uint64_t));

    /* Generate a map from active ECP atoms to ECP shells */
    uint64_t *ecpMap = (uint64_t *) malloc(numActiveECP * sizeof(uint64_t));

    for (uint64_t i = 0, ecp = 0; i < xyzData->numAtoms; i++) {
        uint64_t basisIndex = 0;
        for (uint64_t j = 0; j < numUnique; j++) {
            if (strcmp(xyzData->symbols[i], uniqueSymbols[j]) == 0) {
                basisIndex = j;
                if (shellList[j]) {
                    ecpIndices[ecp] = i;
                    ecpMap[ecp] = j;
                    ecp++;
                }
                break;
            }
        }
    }

    /* Free the uniqueSymbols */
    free(uniqueSymbols);

    cuestECPShell_t** ecp_shells_pack = (cuestECPShell_t**) malloc(numUnique * sizeof(cuestECPShell_t*));
    cuestECPShell_t** ecp_top_shell_pack = (cuestECPShell_t**) malloc(numUnique * sizeof(cuestECPShell_t*));

    /* Declare the ECP shell parameter handle. */
    cuestECPShellParameters_t ecpshell_parameters;

    /* Create the ECP shell parameters. */
    checkCuestErrors(cuestParametersCreate(
        CUEST_ECPSHELL_PARAMETERS, 
        &ecpshell_parameters));

    for (uint64_t i = 0; i < numUnique; i++) {
        if (!shellList[i]) continue;

        size_t numTotalShells = shellList[i]->n_shells;
        size_t numShells = numTotalShells - 1;

        /* Allocate space to store all the ECP shell handles. */
        cuestECPShell_t* ecp_shells = (cuestECPShell_t*) malloc(numShells * sizeof(cuestECPShell_t));
        if (!ecp_shells) {
            fprintf(stderr, "Failed to allocate ECP shell array\n");
            checkCuestErrors(cuestDestroy(handle));
            exit(EXIT_FAILURE);
        }
    
        /* Allocate space to store the top shell handle. */
        cuestECPShell_t* top_shell = (cuestECPShell_t*) malloc(sizeof(cuestECPShell_t));
        if (!top_shell) {
            fprintf(stderr, "Failed to allocate ECP top shell\n");
            checkCuestErrors(cuestDestroy(handle));
            exit(EXIT_FAILURE);
        }
    
        /* Create the ECP top shell. */
        checkCuestErrors(cuestECPShellCreate(
            handle,                          ///< cuEST handle 
            shellList[i]->shell_types[0],    ///< L=3 for an f shell
            shellList[i]->num_primitives[0], ///< The shell has 4 primitives
            shellList[i]->Ns,                ///< size_t* containing the atomic numbers
            shellList[i]->coefficients,      ///< double* containing the 
            shellList[i]->exponents,         ///< double* containing the 
            ecpshell_parameters,             ///< cuestECPShellParameters_t with default parameters
            &top_shell[0]));                 ///< The output cuestECPShell_t

        /* Create the remaining ECP shells. */
        for (uint64_t k=0; k<numShells; ++k) {
            uint64_t offset = shellList[i]->primitive_offsets[k + 1];
            checkCuestErrors(cuestECPShellCreate(
                handle,                                ///< cuEST handle 
                shellList[i]->shell_types[k+1],        ///< L=3 for an f shell
                shellList[i]->num_primitives[k+1],     ///< The shell has 4 primitives
                &(shellList[i]->Ns)[offset],           ///< size_t* containing the atomic numbers
                &(shellList[i]->coefficients)[offset], ///< double* containing the 
                &(shellList[i]->exponents)[offset],    ///< double* containing the 
                ecpshell_parameters,                   ///< cuestECPShellParameters_t with default parameters
                &ecp_shells[k]));                      ///< The output cuestECPShell_t
        }

        ecp_top_shell_pack[i] = top_shell;
        ecp_shells_pack[i] = ecp_shells;
    }

    /* Once all the shells are created, the ECP shell parameters can be freed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_ECPSHELL_PARAMETERS, 
        ecpshell_parameters));

    /* Allocate space to store the ECP atom handles. */
    cuestECPAtom_t* ecpAtoms = (cuestECPAtom_t*) malloc(numActiveECP * sizeof(cuestECPAtom_t));
    if (!ecpAtoms) {
        fprintf(stderr, "Failed to allocate ECP atom array\n");
        checkCuestErrors(cuestDestroy(handle));
        exit(EXIT_FAILURE);
    }

    /* Declare the ECP atom parameter handle. */
    cuestECPAtomParameters_t ecpatom_parameters;

    /* Create the ECP atom parameters. */
    checkCuestErrors(cuestParametersCreate(
        CUEST_ECPATOM_PARAMETERS, 
        &ecpatom_parameters));

    /* Create the ECP Atoms  */
    for (uint64_t i=0; i<numActiveECP; ++i) {
        uint64_t listId = ecpMap[i];
        uint64_t nelec = shellList[listId]->n_elec;
        uint64_t nshell = shellList[listId]->n_shells - 1;
        checkCuestErrors(cuestECPAtomCreate(
            handle,                        ///< cuEST handle 
            nelec,                         ///< numElectrons
            nshell,                        ///< numShells
            ecp_shells_pack[listId],       ///< ECPShell* containing the shells
            ecp_top_shell_pack[listId][0], ///< ECPShell containing the top shell
            ecpatom_parameters,            ///< cuestECPAtomParameters_t with default parameters
            &ecpAtoms[i]));                ///< The output cuestECPAtom_t
    }

    /* Free shell and atom data */
    for (uint64_t i = 0; i < numUnique; i++) {
        if (!shellList[i]) continue;

        size_t numTotalShells = shellList[i]->n_shells;
        size_t numShells = numTotalShells - 1;

        checkCuestErrors(cuestECPShellDestroy(ecp_top_shell_pack[i][0]));
        free(ecp_top_shell_pack[i]);

        for (uint64_t k=0; k<numShells; ++k) {
            checkCuestErrors(cuestECPShellDestroy(ecp_shells_pack[i][k]));
        }
        free(ecp_shells_pack[i]);
    }
    free(ecp_top_shell_pack);
    free(ecp_shells_pack);

    /* Free shellList and map */
    for (uint64_t i = 0; i < numUnique; i++) {
        freeParsedECPFile(shellList[i]);
    }
    free(shellList);
    free(ecpMap);

    /* Once all the atoms are created, the ECP atom parameters can be freed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_ECPATOM_PARAMETERS, 
        ecpatom_parameters));

    /******************************************************/
    /* cuEST Effective Core Potential Integral Plan Setup */
    /******************************************************/

    /* Declare the ECP integral plan handle. */
    cuestECPIntPlan_t ecpint_plan;

    /* Declare and create the ECP integral plan parameter handle. */
    cuestECPIntPlanParameters_t ecpint_plan_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_ECPINTPLAN_PARAMETERS,
        &ecpint_plan_parameters));

    /* Determine how large the workspace must be to form and store the ECP integral plan. */
    checkCuestErrors(cuestECPIntPlanCreateWorkspaceQuery(
        handle,
        basis,
        xyzData->xyzCPU,
        numActiveECP,
        ecpIndices,
        ecpAtoms,
        ecpint_plan_parameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &ecpint_plan));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentECPIntPlanWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryECPIntPlanWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Create the ECP integral plan. */
    checkCuestErrors(cuestECPIntPlanCreate(
        handle,
        basis,
        xyzData->xyzCPU,
        numActiveECP,
        ecpIndices,
        ecpAtoms,
        ecpint_plan_parameters,
        persistentECPIntPlanWorkspace,
        temporaryECPIntPlanWorkspace,
        &ecpint_plan));

    free(ecpIndices);

    for (uint64_t i=0; i<numActiveECP; ++i) {
        checkCuestErrors(cuestECPAtomDestroy(ecpAtoms[i]));
    }
    free(ecpAtoms);

    /* The ECP integral plan parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_ECPINTPLAN_PARAMETERS,
        ecpint_plan_parameters));

    /* The temporary workspace can be freed. */
    freeWorkspace(temporaryECPIntPlanWorkspace);

    /**********************************/
    /*      Compute ECP Gradient      */
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

    /* Allocate space for dECP/dR and density matrices */
    double *d_dECPdR, *d_D;
    if (cudaMalloc((void**) &d_dECPdR, numAtoms * 3 * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void**) &d_D, nao * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }

    /* Read in density from file */
    fill_symmetric_matrix(d_D, nao);

    /* Declare the ECP compute parameter handle. */
    cuestECPDerivativeComputeParameters_t ecpcompute_parameters;

    /* Create the ECP compute parameters. */
    checkCuestErrors(cuestParametersCreate(
        CUEST_ECPDERIVATIVECOMPUTE_PARAMETERS, 
        &ecpcompute_parameters));

    cuestWorkspaceDescriptor_t* variableBufferSize = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    variableBufferSize->hostBufferSizeInBytes = 0;
    variableBufferSize->deviceBufferSizeInBytes = 2000000000;

    /* Workspace query to find the temporary space needed to form the ECP integrals. */
    checkCuestErrors(cuestECPDerivativeComputeWorkspaceQuery(
        handle,
        ecpint_plan,
        ecpcompute_parameters,
        variableBufferSize,
        temporaryWorkspaceDescriptor,
        d_D,
        d_dECPdR));

    cuestWorkspace_t* temporaryECPWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    /* Compute ECP Integrals */
    checkCuestErrors(cuestECPDerivativeCompute(
        handle,
        ecpint_plan,
        ecpcompute_parameters,
        variableBufferSize,
        temporaryECPWorkspace,
        d_D,
        d_dECPdR));

    /* Destroy the ECP compute parameters */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_ECPDERIVATIVECOMPUTE_PARAMETERS, 
        ecpcompute_parameters));

    // Copy calculated results back to host
    double *h_dECPdR = (double*) malloc(numAtoms * 3 * sizeof(double));
    memset(h_dECPdR, 0, numAtoms * 3 * sizeof(double));
    if(cudaMemcpy(h_dECPdR, d_dECPdR, numAtoms * 3 * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Device to Host copy failed\n");
        exit(EXIT_FAILURE);
    }

    int print_output = 0;
    if (print_output) {
        for (int i=0; i<numAtoms; ++i) {
            fprintf(stderr, "%lf ", h_dECPdR[i * 3 + 0]);
            fprintf(stderr, "%lf ", h_dECPdR[i * 3 + 1]);
            fprintf(stderr, "%lf ", h_dECPdR[i * 3 + 2]);
            fprintf(stderr, "\n");
        }
    }

    free(h_dECPdR);

    free(variableBufferSize);
    freeWorkspace(temporaryECPWorkspace);

    /* Free ECP gradient */
    if (cudaFree((void*) d_dECPdR) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }

    /* Free density */
    if (cudaFree((void*) d_D) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }

    /*****************************************/
    /* Destroy cuEST handles and free memory */
    /*****************************************/

    /* Destroy the ECP integral plan handle. */
    checkCuestErrors(cuestECPIntPlanDestroy(ecpint_plan));
    freeWorkspace(persistentECPIntPlanWorkspace);

    /* The workspace descriptors are no longer needed. */
    free(persistentWorkspaceDescriptor);
    free(temporaryWorkspaceDescriptor);

    /* Destroy the AO basis handle. */
    checkCuestErrors(cuestAOBasisDestroy(basis));
    freeWorkspace(persistentBasisWorkspace);

    /* Destroy the cuEST handle. */
    checkCuestErrors(cuestDestroy(handle));

    /* Free the XYZ data */
    freeParsedXYZFile(xyzData);

    return 0;
}
