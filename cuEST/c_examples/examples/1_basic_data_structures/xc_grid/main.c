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
#include <math.h>

#include <cuest.h>

#include <helper_status.h>
#include <helper_workspaces.h>

#include "molecule_definition.h"

/*
 * This example shows how to construct a pruned integration grid 
 * following:
 *
 * Reference: O. Treutler and R. Ahlrichs,
 *   J. Chem. Phys., 102, 346 (1995)
 *
 * This example builds "GRID1" for a water molecule. 
 */

/*
 * This function produces the Ahlrichs radial quadrature. It takes
 * preallocated arrays to store the radial nodes and weights as input
 * and populates them with the quadrature. Arrays must be of length
 * npoint.
 */
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

int main(int argc, char **argv)
{
    /* Obtain XYZ coordinates of a water molecule. */
    parsedXYZFile_t* xyzData = h2oXYZFile();
    if (!xyzData) {
        fprintf(stderr, "Error: failed to produce H2O xyz data\n");
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

    /***************************/
    /* cuEST Atomic Grid Setup */
    /***************************/

    /* Build an array of cuestAtomGrid_t for each atom in the molecule. */
    uint64_t numAtoms = xyzData->numAtoms;
    cuestAtomGrid_t* atomGrid = (cuestAtomGrid_t*) malloc(numAtoms * sizeof(cuestAtomGrid_t));
    if (!atomGrid) {
        checkCuestErrors(cuestDestroy(handle));
        freeParsedXYZFile(xyzData);
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    
    for (uint64_t n=0; n<numAtoms; n++) {

        /* Define the number of radial points and atomic radius for each atom. */
        size_t numRadialPoints = 0;
        double radius = 0.0;
        if (strcmp("O", xyzData->symbols[n]) == 0) {
            numRadialPoints = 25;
            radius = 0.90;
        }
        else /* (strcmp("H", xyzData->symbols[n]) == 0) */ {
            numRadialPoints = 20;
            radius = 0.80;
        }

        /* Allocate space for the radial and angular quadratures. */
        double *radialNodes = NULL;
        double *radialWeights = NULL;
        uint64_t *numAngularPointsArray = NULL;
 
        int fail = 0;
        do {
            radialNodes = (double*) malloc(numRadialPoints * sizeof(double));
            if (!radialNodes) {
                fail = 1;
                break;
            }
            radialWeights = (double*) malloc(numRadialPoints * sizeof(double));
            if (!radialWeights) {
                fail = 1;
                break;
            }
            numAngularPointsArray = (uint64_t*) malloc(numRadialPoints * sizeof(uint64_t));
            if (!numAngularPointsArray) {
                fail = 1;
                break;
            }
        }
        while(0);
      
        if (fail) {
            free(radialNodes);
            free(radialWeights);
            free(numAngularPointsArray);
            break;
        }

        /* The Treutler-Ahlrichs pruning scheme is encoded in the array of angular points. */
        if (strcmp("O", xyzData->symbols[n]) == 0) {
            for (uint64_t i=0; i<8; i++) {
                numAngularPointsArray[i] = 14;
            }
            for (uint64_t i=8; i<12; i++) {
                numAngularPointsArray[i] = 50;
            }
            for (uint64_t i=12; i<numRadialPoints; i++) {
                numAngularPointsArray[i] = 110;
            }
        }
        else /* (strcmp("H", xyzData->symbols[n]) == 0) */ {
            for (uint64_t i=0; i<6; i++) {
                numAngularPointsArray[i] = 14;
            }
            for (uint64_t i=6; i<10; i++) {
                numAngularPointsArray[i] = 50;
            }
            for (uint64_t i=10; i<numRadialPoints; i++) {
                numAngularPointsArray[i] = 50;
            }
        }

        /* Create the cuestAtomGrid_t for each atom. */
        cuestAtomGridParameters_t atomGridParameters;
        checkCuestErrors(cuestParametersCreate(
            CUEST_ATOMGRID_PARAMETERS, 
            &atomGridParameters));
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
        checkCuestErrors(cuestParametersDestroy(
            CUEST_ATOMGRID_PARAMETERS,
            atomGridParameters));

        /* Free temporary arrays. */
        free(radialNodes);
        free(radialWeights);
        free(numAngularPointsArray);
    }

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

    /* Allocate space for workspace descriptors. Will be used to determine how large the workspace needs to be. */
    cuestWorkspaceDescriptor_t* persistentWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    cuestWorkspaceDescriptor_t* temporaryWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));

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

    /* The workspace descriptors are no longer needed. */
    free(persistentWorkspaceDescriptor);
    free(temporaryWorkspaceDescriptor);

    /*****************************************/
    /* Destroy cuEST handles and free memory */
    /*****************************************/

    /* Destroy the molecular grid handle. */
    checkCuestErrors(cuestMolecularGridDestroy(molecularGrid));
    freeWorkspace(persistentGridWorkspace);

    /* Destroy the atom grid handles. */
    for (int n=0; n<xyzData->numAtoms; n++) {
        checkCuestErrors(cuestAtomGridDestroy(atomGrid[n]));
    }
    free(atomGrid);

    /* Destroy the cuEST handle. */
    checkCuestErrors(cuestDestroy(handle));

    /* Free the XYZ data */
    freeParsedXYZFile(xyzData);

    return 0;
}
