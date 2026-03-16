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
#include <helper_xyz_parser.h>
#include <helper_ecp_parser.h>

/*
 * This sample shows how to create an ECP basis for an arbitrary molecule
 * from an array of ECP shells. The def2-SVP-ecp basis set is used as an example.
 * The parser in helper_ecp_parser.h was tested only for the Gaussian 94 format.
 * The radial powers, coefficients, and exponents can be queried directly from the basis file.
 */
int main(int argc, char **argv)
{
    /* Check that an xyz file and an ecp basis has been provided */
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <xyz_file_path> <ecp_file_path>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const char* xyzFilePath = argv[1];
    const char* ecpFilePath = argv[2];

    /* Parse the XYZ file */
    parsedXYZFile_t* xyzData = parseXYZFile(xyzFilePath, 1.0 / 0.52917720859);
    if (!xyzData) {
        fprintf(stderr, "Error: failed to parse file '%s'\n", xyzFilePath);
        exit(EXIT_FAILURE);
    }

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
    uint64_t num_active_ecp = 0;
    for (uint64_t i = 0; i < xyzData->numAtoms; i++) {
        uint64_t basisIndex = 0;
        for (uint64_t j = 0; j < numUnique; j++) {
            if (strcmp(xyzData->symbols[i], uniqueSymbols[j]) == 0) {
                basisIndex = j;
                if (shellList[j]) num_active_ecp++;
                break;
            }
        }
    }

    /* Generate a map from active ECP atoms to ECP shells */
    uint64_t *ecpMap = (uint64_t *) malloc(num_active_ecp * sizeof(uint64_t));

    for (uint64_t i = 0, ecp = 0; i < xyzData->numAtoms; i++) {
        uint64_t basisIndex = 0;
        for (uint64_t j = 0; j < numUnique; j++) {
            if (strcmp(xyzData->symbols[i], uniqueSymbols[j]) == 0) {
                basisIndex = j;
                if (shellList[j]) {
                    ecpMap[ecp] = j;
                    ecp++;
                }
                break;
            }
        }
    }

    /* free the xyzData */
    freeParsedXYZFile(xyzData);

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
            handle,                                    ///< cuEST handle 
            shellList[i]->shell_types[0],              ///< Angular momentum
            shellList[i]->num_primitives[0],           ///< Number of primitives
            shellList[i]->Ns,                          ///< size_t* containing the radial powers
            shellList[i]->coefficients,                ///< double* containing the coefficients
            shellList[i]->exponents,                   ///< double* containing the exponents
            ecpshell_parameters,                       ///< cuestECPShellParameters_t with default parameters
            &top_shell[0]));                           ///< The output cuestECPShell_t

        /* Create the remaining ECP shells. */
        for (uint64_t k=0; k<numShells; ++k) {
            uint64_t offset = shellList[i]->primitive_offsets[k + 1];
            checkCuestErrors(cuestECPShellCreate(
                handle,                                ///< cuEST handle 
                shellList[i]->shell_types[k+1],        ///< Angular momentum
                shellList[i]->num_primitives[k+1],     ///< Number of primitives
                &(shellList[i]->Ns)[offset],           ///< size_t* containing the radial powers
                &(shellList[i]->coefficients)[offset], ///< double* containing the coefficients
                &(shellList[i]->exponents)[offset],    ///< double* containing the exponents
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
    cuestECPAtom_t* ecp_atoms = (cuestECPAtom_t*) malloc(num_active_ecp * sizeof(cuestECPAtom_t));
    if (!ecp_atoms) {
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
    for (uint64_t i=0; i<num_active_ecp; ++i) {
        uint64_t listId = ecpMap[i];
        uint64_t nelec = shellList[listId]->n_elec;
        uint64_t nshell = shellList[listId]->n_shells - 1;
        checkCuestErrors(cuestECPAtomCreate(
            handle,                        ///< cuEST handle 
            nelec,                         ///< numElectrons
            nshell,                        ///< numShells (not including the top shell)
            ecp_shells_pack[listId],       ///< ECPShell* containing the shells
            ecp_top_shell_pack[listId][0], ///< ECPShell containing the top shell
            ecpatom_parameters,            ///< cuestECPAtomParameters_t with default parameters
            &ecp_atoms[i]));               ///< The output cuestECPAtom_t
    }

    /* Once all the atoms are created, the ECP atom parameters can be freed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_ECPATOM_PARAMETERS, 
        ecpatom_parameters));

    /* Free shell data */
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

    uint64_t max_L = 0;
    uint64_t nelec = 0;

    /* Query and destroy the Atoms */
    for (uint64_t i=0; i<num_active_ecp; ++i) {
        checkCuestErrors(cuestQuery(handle, CUEST_ECPATOM, ecp_atoms[i], CUEST_ECPATOM_MAX_L,        &max_L,      sizeof(uint64_t)));
        checkCuestErrors(cuestQuery(handle, CUEST_ECPATOM, ecp_atoms[i], CUEST_ECPATOM_NUM_ELECTRON, &nelec,      sizeof(uint64_t)));

        fprintf(stdout,"ECP Atom from handle:\n");
        fprintf(stdout,"%-10s = %6zu\n", "max_L",      max_L);
        fprintf(stdout,"%-10s = %6zu\n", "nelec",      nelec);
        fprintf(stdout, "\n");

        checkCuestErrors(cuestECPAtomDestroy(ecp_atoms[i]));
    }

    free(ecp_atoms);

    /* Destroy the cuEST handle. */
    checkCuestErrors(cuestDestroy(handle));

    return 0;
}
