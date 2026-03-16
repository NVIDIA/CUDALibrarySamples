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

#ifndef COMMON_HELPER_GRID
#define COMMON_HELPER_GRID

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include <cuest.h>

#include "helper_xyz_parser.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * This helper provides a formDirectProductAtomGrid function that, given the contents 
 * of an XYZ file (parsed using parseXYZFile helper), will return an array of atom grids
 * cothat can be used to create a cuEST molecular grid handle. This function constructs
 * unpruned grids with a user specified number of radial and angular points. An Ahlrichs
 * type radial quadrature is used.
 */

static double symbol_to_ahlrichs_radius(const char *symbol) 
{
    static const char *elements[] = {
        "X",  
        "H",  "HE", "LI", "BE", "B",  "C",  "N",  "O",  "F",  "NE",
        "NA", "MG", "AL", "SI", "P",  "S",  "CL", "AR", "K",  "CA",
        "SC", "TI", "V",  "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
        "GA", "GE", "AS", "SE", "BR", "KR"
    };
    static const double ahlrichs_radii_[] = {
    1.00,
    0.80,                                                                                0.90,
    1.80,1.40,                                                  1.30,1.10,0.90,0.90,0.90,0.90,
    1.40,1.30,                                                  1.30,1.20,1.10,1.00,1.00,1.00,
    1.50,1.40,1.30,1.20,1.20,1.20,1.20,1.20,1.20,1.10,1.10,1.10,1.10,1.00,0.90,0.90,0.90,0.90
    };
    for (int i = 0; i <= 36; ++i) {
        if (strcmp(symbol, elements[i]) == 0) return ahlrichs_radii_[i];
    }
    return 1.0;
}

void build_ahlrichs_radial_quadrature(
    size_t npoint,
    double R,
    double *radialNodes,
    double *radialWeights)
{
    const double alpha = 0.6;
    for (size_t i = 1; i <= npoint; i++) {
        double z = i * M_PI / (npoint + 1.0);
        double x = cos(z);
        double y = sin(z);
        double u = log((1.0 - x) / 2.0);
        double v = pow(1.0 + x, alpha) / log(2.0);
        double r = - R * v * u;
        double w = M_PI / (npoint + 1.0) * y * R  * v * (-alpha * u / (1.0 + x) + 1.0 / (1.0 - x)) * r * r;
        radialNodes[npoint-i] = r;
        radialWeights[npoint-i] = w;
    }
}

cuestAtomGrid_t* formDirectProductAtomGrid(
    cuestHandle_t handle,
    parsedXYZFile_t* xyzData,
    size_t numRadialPoints,
    size_t numAngularPoints)
{
    uint64_t numAtoms = xyzData->numAtoms;

    cuestAtomGrid_t* atomGrid = (cuestAtomGrid_t*) malloc(numAtoms * sizeof(cuestAtomGrid_t));
    if (!atomGrid) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    
    cuestAtomGridParameters_t atomGridParameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_ATOMGRID_PARAMETERS, 
        &atomGridParameters));

    double *radialNodes = (double*) malloc(numRadialPoints * sizeof(double));
    if (!radialNodes) {
        free(atomGrid);
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    double *radialWeights = (double*) malloc(numRadialPoints * sizeof(double));
    if (!radialWeights) {
        free(atomGrid);
        free(radialNodes);
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    uint64_t *numAngularPointsArray = (uint64_t*) malloc(numRadialPoints * sizeof(uint64_t));
    if (!numAngularPointsArray) {
        free(atomGrid);
        free(radialNodes);
        free(radialWeights);
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    for (uint64_t n=0; n<numRadialPoints; n++) {
        numAngularPointsArray[n] = numAngularPoints;
    }

    for (uint64_t n=0; n<numAtoms; n++) {
        double radius = symbol_to_ahlrichs_radius(xyzData->symbols[n]);
        build_ahlrichs_radial_quadrature(
            numRadialPoints,
            radius,
            radialNodes,
            radialWeights);
        checkCuestErrors(cuestAtomGridCreate(
            handle,
            numRadialPoints,
            radialNodes,
            radialWeights,
            numAngularPointsArray,
            atomGridParameters,
            &atomGrid[n]));
    }

    checkCuestErrors(cuestParametersDestroy(
        CUEST_ATOMGRID_PARAMETERS,
        atomGridParameters));

    free(radialNodes);
    free(radialWeights);
    free(numAngularPointsArray);

    return atomGrid;
}

#ifdef __cplusplus
} 
#endif

#endif /* COMMON_HELPER_GRID */
