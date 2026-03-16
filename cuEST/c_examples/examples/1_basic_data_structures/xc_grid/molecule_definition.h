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

#ifndef MOLECULE_DEFINITION_H
#define MOLECULE_DEFINITION_H

#include <helper_xyz_parser.h>

/* This is a function that returns hardcoded XYZ coordinates of a
 * water molecule in a parsedXYZFile_t structure.
 */
static parsedXYZFile_t* h2oXYZFile()
{
    size_t numAtoms = 3;
    double* xyzCPU = (double*) malloc(3 * numAtoms * sizeof(double));
    double* chargesCPU = (double*) malloc(numAtoms * sizeof(double));
    char **symbols = (char**) malloc(numAtoms * sizeof(char*));
    if (!xyzCPU || !chargesCPU || !symbols) { 
        if (xyzCPU) free(xyzCPU); 
        if (chargesCPU) free(chargesCPU); 
        if (symbols) free(symbols); 
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    chargesCPU[0] = -8.0;
    chargesCPU[1] = -1.0;
    chargesCPU[2] = -1.0;
  
    xyzCPU[0*3 + 0] =  0.000000;
    xyzCPU[0*3 + 1] = -0.224906;
    xyzCPU[0*3 + 2] =  0.000000;

    xyzCPU[1*3 + 0] =  1.452350;
    xyzCPU[1*3 + 1] =  0.899624;
    xyzCPU[1*3 + 2] =  0.000000;
                               
    xyzCPU[2*3 + 0] = -1.452350;
    xyzCPU[2*3 + 1] =  0.899624;
    xyzCPU[2*3 + 2] =  0.000000;

    int fail = 0;
    const char *temp[] = {"O", "H", "H"};
    for (int i=0; i<numAtoms; i++) {
        symbols[i] = malloc(strlen(temp[i]) + 1);
        if (!symbols[i]) {
            fail = 1;
            break;
        }
        strcpy(symbols[i], temp[i]);
    }

    if (fail) {
        free(xyzCPU); 
        free(chargesCPU); 
        for (size_t i=0; i<numAtoms; i++) {
            if (symbols[i]) {
                free(symbols[i]);
            }
        }
        free(symbols);
        fprintf(stderr, "Failed to allocate atomic symbols\n");
        exit(EXIT_FAILURE);
    }

    double* xyzGPU = NULL;
    double* chargesGPU = NULL;
    parsedXYZFile_t* result = NULL;
    
    do {
        if (cudaMalloc((void**) &xyzGPU, 3 * numAtoms * sizeof(double)) != cudaSuccess) {
            fail = 1;
            break;
        }
        if (cudaMalloc((void**) &chargesGPU, numAtoms * sizeof(double)) != cudaSuccess) {
            fail = 1;
            break;
        }
        if (cudaMemcpy(xyzGPU, xyzCPU, 3 * numAtoms * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
            fail = 1;
            break;
        }
        if (cudaMemcpy(chargesGPU, chargesCPU,  numAtoms * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
            fail = 1;
            break;
        }
    }
    while(0);

    result = (parsedXYZFile_t*) malloc(sizeof(parsedXYZFile_t));
    if (!result || fail) {
        if (result) free(result);
        if (xyzGPU) cudaFree(xyzGPU);
        if (chargesGPU) cudaFree(chargesGPU); 
        free(xyzCPU); 
        free(chargesCPU); 
        for (size_t i=0; i<numAtoms; i++) {
            if (symbols[i]) {
                free(symbols[i]);
            }
        }
        free(symbols);
        fprintf(stderr, "Memory allocation/copy failed\n");
        exit(EXIT_FAILURE);
    }

    result->numAtoms = numAtoms;
    result->xyzCPU = xyzCPU;
    result->xyzGPU = xyzGPU;
    result->chargesCPU = chargesCPU;
    result->chargesGPU = chargesGPU;
    result->symbols = symbols;

    return result;
}


#endif /* MOLECULE_DEFINITION_H */
