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

#ifndef COMMON_HELPER_XYZ_PARSER
#define COMMON_HELPER_XYZ_PARSER

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <string.h>
#include <cuda_runtime.h>

/*
 * This helper provides a rudimentary parser for XYZ files.
 * It will parse the XYZ file and return a parsedXYZFile_t
 * with some useful arrays populated. 
 *
 * Expected formatting for XYZ files is:
 *
 * [Number of atoms]
 * Comment line
 * Symbol_1 X_1 Y_1 Z_1
 * Symbol_2 X_2 Y_2 Z_2
 * ...
 * ...
 * Symbol_N X_N Y_N Z_N
 *
 * This parser only addresses the first number of atoms + 2
 * lines of the file. Additional lines are ignored.
 */ 

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t numAtoms;
    double *xyzCPU;
    double *xyzGPU;
    double *chargesCPU;
    double *chargesGPU;
    char **symbols;
} parsedXYZFile_t;

static void xyz_to_upper(char *s) 
{
    while (*s) {
        *s = (char)toupper((unsigned char)*s);
        s++;
    }
}

static int symbol_to_atomic_number(const char *symbol) 
{
    static const char *elements[] = {
        "X",  "H",  "HE", "LI", "BE", "B",  "C",  "N",  "O",  "F",  "NE",
        "NA", "MG", "AL", "SI", "P",  "S",  "CL", "AR", "K",  "CA",
        "SC", "TI", "V",  "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
        "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y",  "ZR",
        "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
        "SB", "TE", "I",  "XE", "CS", "BA", "LA", "CE", "PR", "ND",
        "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
        "LU", "HF", "TA", "W",  "RE", "OS", "IR", "PT", "AU", "HG",
        "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
        "PA", "U",  "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM",
        "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT", "DS",
        "RG", "CN", "NH", "FL", "MC", "LV", "TS", "OG"
    };
    for (int i = 0; i < 119; ++i) {
        if (strcmp(symbol, elements[i]) == 0) return i;
    }
    fprintf(stderr, "Unknown atomic symbol\n");
    exit(EXIT_FAILURE);
}

static parsedXYZFile_t* parseXYZFile(
    const char* xyzFilePath,
    double toBohrScaleFactor)
{
    FILE* fin = fopen(xyzFilePath, "r");
    if (!fin) {
        fprintf(stderr, "Unable to open XYZ file\n");
        exit(EXIT_FAILURE);
    }
    char line[1024];
    if (!fgets(line, sizeof(line), fin)) { 
        fclose(fin); 
        fprintf(stderr, "Failed to read number of atoms\n");
        exit(EXIT_FAILURE);
    }
    size_t numAtoms = (size_t) strtoul(line, NULL, 10);
    if (!fgets(line, sizeof(line), fin)) { 
        fclose(fin);
        fprintf(stderr, "Failed to read comment line\n");
        exit(EXIT_FAILURE);
    }

    double* xyzCPU = (double*) malloc(3 * numAtoms * sizeof(double));
    double* chargesCPU = (double*) malloc(numAtoms * sizeof(double));
    char **symbols = (char**) malloc(numAtoms * sizeof(char*));
    if (!xyzCPU || !chargesCPU || !symbols) { 
        if (xyzCPU) free(xyzCPU); 
        if (chargesCPU) free(chargesCPU); 
        if (symbols) free(symbols); 
        fclose(fin);
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    int fail = 0;
    for (size_t i = 0; i < numAtoms; ++i) {
        if (!fgets(line, sizeof(line), fin)) { 
            fail = 1;
            break;
        }
        char symbol[16];
        double x, y, z;
        if (sscanf(line, "%15s%lf%lf%lf", symbol, &x, &y, &z) != 4) { 
            fail = 1;
            break;
        }
        xyz_to_upper(symbol);
        xyzCPU[3*i + 0] = x * toBohrScaleFactor;
        xyzCPU[3*i + 1] = y * toBohrScaleFactor;
        xyzCPU[3*i + 2] = z * toBohrScaleFactor;
        int atomic_number = symbol_to_atomic_number(symbol);
        /* 
         * cuEST uses a convention for potential evaluation that the charge of the electron
         * is not assumed to be included. Here, the charge of the electron (-1) is included
         * along with the nuclear chage (+Z).
         */ 
        chargesCPU[i] = -1.0 * (double) atomic_number;

        symbols[i] = (char*) malloc(strlen(symbol) + 1);
        if (!symbols[i]) {
            fail = 1;
            break;
        }
        strcpy(symbols[i], symbol);
    }
    fclose(fin);

    if (fail) {
        free(xyzCPU); 
        free(chargesCPU); 
        for (size_t i=0; i<numAtoms; i++) {
            if (symbols[i]) {
                free(symbols[i]);
            }
        }
        free(symbols);
        fprintf(stderr, "Failed to parse XYZ file\n");
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

static void freeParsedXYZFile(parsedXYZFile_t *data) 
{
    if (data == NULL) {
        return;
    }

    int fail = 0;
    if (data->xyzCPU) {
        free(data->xyzCPU);
    }
    if (data->chargesCPU) {
        free(data->chargesCPU);
    }
    if (data->xyzGPU) {
        if (cudaFree(data->xyzGPU) != cudaSuccess) {
            fail++;
        }
    }
    if (data->chargesGPU) {
        if (cudaFree(data->chargesGPU) != cudaSuccess) {
            fail++;
        }
    }
    if (data->symbols) {
        for (size_t i=0; i<data->numAtoms; i++) {
            if (data->symbols[i]) {
                free(data->symbols[i]); 
            }
        }
        free(data->symbols); 
    }
    free(data);

    if (fail) {
        fprintf(stderr, "Failed to free device data\n");
        exit(EXIT_FAILURE);
    }
}

#ifdef __cplusplus
} 
#endif

#endif /* COMMON_HELPER_XYZ_PARSER */
