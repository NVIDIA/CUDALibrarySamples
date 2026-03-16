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

#ifndef COMMON_HELPER_AO_SHELLS
#define COMMON_HELPER_AO_SHELLS

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#include <cuest.h>

#include "helper_xyz_parser.h"
#include "helper_gbs_parser.h"
#include "helper_shell_normalization.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * This helper provides a formAOShells function that, given the contents of an XYZ file
 * (parsed using parseXYZFile helper), will return an array of shells suitable for use
 * constructing a cuEST AO basis handle. The array of shells (and important metadata)
 * is returned via the AtomShellData_t struct.
 *
 * This helper uses the computeNormalizedCoefficients helper to return a basis set
 * definition with normalized coefficients.
 */

typedef struct {
    uint64_t        numAtoms;           ///< Total number of atoms
    uint64_t        numShellsTotal;     ///< Total number of shells
    uint64_t*       numShellsPerAtom;   ///< Number of shells per atom
    cuestAOShell_t* shells;             ///< 
} AtomShellData_t;

static uint64_t get_L(const AtomBasisSet_t* basis, uint64_t shell) {
    return basis->shell_types[shell];
} 

static uint64_t get_number_of_primitives(const AtomBasisSet_t* basis, uint64_t shell) {
    return basis->num_primitives[shell];
} 

static const double* get_exponents(const AtomBasisSet_t* basis, uint64_t shell) {
    return &(basis->exponents[basis->primitive_offsets[shell]]);
}

static const double* get_coefficients(const AtomBasisSet_t* basis, uint64_t shell) {
    return &(basis->coefficients[basis->primitive_offsets[shell]]);
}

static AtomShellData_t* formAOShells(cuestHandle_t handle, parsedXYZFile_t* xyzData, const char* gbsFile, int32_t isPure)
{
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

    AtomBasisSet_t** basisList = (AtomBasisSet_t**) malloc(numUnique * sizeof(AtomBasisSet_t*));
    if (!basisList) {
        free(uniqueSymbols);
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    for (uint64_t i = 0; i < numUnique; i++) {
        basisList[i] = parseGBSFileForElement(gbsFile, uniqueSymbols[i]);
    }

    uint64_t numAtoms = xyzData->numAtoms;
    uint64_t *numShellsPerAtom = (uint64_t*) malloc(numAtoms * sizeof(uint64_t));
    uint64_t numShellsTotal = 0;
  
    if (!numShellsPerAtom) {
        free(uniqueSymbols);
        for (uint64_t j = 0; j < numUnique; j++) {
            freeParsedGBSFile(basisList[j]);
        }
        free(basisList);
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
 
    for (uint64_t i = 0; i < xyzData->numAtoms; i++) {
        uint64_t basisIndex = 0;
        for (uint64_t j = 0; j < numUnique; j++) {
            if (strcmp(xyzData->symbols[i], uniqueSymbols[j]) == 0) {
                basisIndex = j;
                break;
            }
        }
        const AtomBasisSet_t* basis = basisList[basisIndex];
        numShellsPerAtom[i] = basis->n_shells;
        numShellsTotal += basis->n_shells;
    }

    /* Allocate space for the array of AO shells */
    cuestAOShell_t* shells = (cuestAOShell_t*) malloc(numShellsTotal * sizeof(cuestAOShell_t));

    if (!shells) {
        free(uniqueSymbols);
        for (uint64_t j = 0; j < numUnique; j++) {
            freeParsedGBSFile(basisList[j]);
        }
        free(basisList);
        free(numShellsPerAtom);
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }

    /* Declare the AO shell parameter handle. */
    cuestAOShellParameters_t aoshell_parameters;

    /* Create the AO shell parameters. */
    checkCuestErrors(cuestParametersCreate(
        CUEST_AOSHELL_PARAMETERS, 
        &aoshell_parameters));

    /* Outer loop over the number of atoms. */
    int fail = 0;
    for (uint64_t n=0, count=0; n<numAtoms; n++) {
        /* Grab the basis definition for the current atom */
        uint64_t basisIndex = 0;
        for (uint64_t j = 0; j < numUnique; j++) {
            if (strcmp(xyzData->symbols[n], uniqueSymbols[j]) == 0) {
                basisIndex = j;
                break;
            }
        }
        const AtomBasisSet_t* basis = basisList[basisIndex];

        /* Inner loop over the number of shells on atom n. */
        for (uint64_t i=0; i<numShellsPerAtom[n]; i++, count++) {
            double *normalized_coefficients = (double*) malloc(get_number_of_primitives(basis, i) * sizeof(double));
            if (!normalized_coefficients) {
                /* Clean up previously created shells */
                for (uint64_t cleanup = 0; cleanup < count; cleanup++) {
                    cuestAOShellDestroy(shells[cleanup]);
                }
                fail = 1;
                break;
            }
            checkCuestErrors(computeNormalizedCoefficients(
                get_L(basis, i),                         ///< L for a the shell
                get_number_of_primitives(basis, i),      ///< Number of primitives for the shell
                get_exponents(basis, i),                 ///< double* containing the exponents
                get_coefficients(basis, i),              ///< double* containing the coefficients
                1.0,
                normalized_coefficients));
            checkCuestErrors(cuestAOShellCreate(
                handle,                                  ///< cuEST handle 
                isPure,                                  ///< isPure == 1 if pure, otherwise, 0 for cartesian
                get_L(basis, i),                         ///< L for a the shell
                get_number_of_primitives(basis, i),      ///< Number of primitives for the shell
                get_exponents(basis, i),                 ///< double* containing the exponents
                normalized_coefficients,                 ///< double* containing the coefficients
                aoshell_parameters,                      ///< cuestAOShellParameters_t with default parameters
                &shells[count]));                        ///< The output cuestAOShell_t in the array of shells
            free(normalized_coefficients);
        }
    }

    /* Once all the shells are created, the AO shell parameters can be freed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_AOSHELL_PARAMETERS, 
        aoshell_parameters));

    /* Clear out the basis set helpers */
    for (uint64_t j = 0; j < numUnique; j++) {
        freeParsedGBSFile(basisList[j]);
    }
    free(basisList);
    free(uniqueSymbols);

    AtomShellData_t* shellData = (AtomShellData_t*) malloc(sizeof(AtomShellData_t));

    if (!shellData || fail) {
        if (shellData) free(shellData);
        for (uint64_t n=0, count=0; n<numAtoms; n++) {
            for (uint64_t i=0; i<numShellsPerAtom[n]; i++, count++) {
                checkCuestErrors(cuestAOShellDestroy(shells[count]));
            }
        }
        free(numShellsPerAtom);
        free(shells);
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }

    shellData->numAtoms = numAtoms;
    shellData->numShellsTotal = numShellsTotal;
    shellData->numShellsPerAtom = numShellsPerAtom;
    shellData->shells = shells;

    return shellData;
}

static void freeAOShellData(AtomShellData_t *data)
{
    for (uint64_t n=0, count=0; n<data->numAtoms; n++) {
        for (uint64_t i=0; i<data->numShellsPerAtom[n]; i++, count++) {
            checkCuestErrors(cuestAOShellDestroy(data->shells[count]));
        }
    }
    free(data->numShellsPerAtom);
    free(data->shells);
    free(data);
}

#ifdef __cplusplus
} 
#endif

#endif /* COMMON_HELPER_AO_SHELLS */
