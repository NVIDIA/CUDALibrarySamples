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
#include <helper_ao_shells.h>

#include "basis_definition.h"

/*
 * This sample shows how to create a Gaussian basis for a water molecule
 * from an array of AO shells. The def2-SVP basis set is used as an example.
 * The coefficients and exponents are stored in basis_definition.h. 
 * The shells and basis are queried for their attributes.
 */
int main(int argc, char **argv)
{
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

    /* The example water molecule with def2-SVP has 6 shells on the oxygen and 3 on each hydrogen. */
    uint64_t numAtoms = 3;
    uint64_t numShellsPerAtom[3] = {6, 3, 3};
    uint64_t numShellsTotal = 12;
   
    /* Allocate space to store all the AO shell handles. */
    cuestAOShell_t* shells = (cuestAOShell_t*) malloc(numShellsTotal * sizeof(cuestAOShell_t));
    if (!shells) {
        fprintf(stderr, "Failed to allocate AO shell array\n");
        checkCuestErrors(cuestDestroy(handle));
        exit(EXIT_FAILURE);
    }
    
    /* Declare the AO shell parameter handle. */
    cuestAOShellParameters_t aoshell_parameters;

    /* Create the AO shell parameters. */
    checkCuestErrors(cuestParametersCreate(
        CUEST_AOSHELL_PARAMETERS, 
        &aoshell_parameters));

    /* Outer loop over the number of atoms. */
    for (uint64_t n=0, count=0; n<numAtoms; n++) {
        const AtomBasisSet_t* atom_basis;
        if (n == 0) {
            /* Set the basis to oxygen for the first atom. */
            atom_basis = &(O_basis);
        } else {
            /* Set the basis to hydrogen for the second two atoms. */
            atom_basis = &(H_basis);
        }
        /* Inner loop over the number of shells on atom n. */
        for (uint64_t i=0; i<numShellsPerAtom[n]; i++, count++) {
            checkCuestErrors(cuestAOShellCreate(
                handle,                                  ///< cuEST handle 
                1,                                       ///< 1 implies pure angular momentum (correct for def2-SVP)
                get_L(atom_basis, i),                    ///< L for the shell
                get_number_of_primitives(atom_basis, i), ///< Number of primitives for the shell
                get_exponents(atom_basis, i),            ///< double* containing the exponents
                get_coefficients(atom_basis, i),         ///< double* containing the coefficients
                aoshell_parameters,                      ///< cuestAOShellParameters_t with default parameters
                &shells[count]));                        ///< The output cuestAOShell_t in the array of shells
        }
    }

    /* Once all the shells are created, the AO shell parameters can be freed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_AOSHELL_PARAMETERS, 
        aoshell_parameters));

    /* 
     * Query the AO shells for the attributes of the shells and sum the number of functions to anticipate how large
     * the basis will be. 
     */

    uint64_t max_L = 0, nao_total = 0, nprim_total = 0, npure_total = 0, ncart_total = 0;

    for (uint64_t n=0, count=0; n<numAtoms; n++) {
        for (uint64_t i=0; i<numShellsPerAtom[n]; i++, count++) {
            int32_t isPure = 0;
            uint64_t L = 0, nao = 0, nprim = 0, npure = 0, ncart = 0;

            checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, shells[count], CUEST_AOSHELL_IS_PURE,       &isPure, sizeof(int32_t)));
            checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, shells[count], CUEST_AOSHELL_L,             &L,      sizeof(uint64_t)));
            checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, shells[count], CUEST_AOSHELL_NUM_PRIMITIVE, &nprim,  sizeof(uint64_t)));
            checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, shells[count], CUEST_AOSHELL_NUM_AO,        &nao,    sizeof(uint64_t)));
            checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, shells[count], CUEST_AOSHELL_NUM_PURE,      &npure,  sizeof(uint64_t)));
            checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, shells[count], CUEST_AOSHELL_NUM_CART,      &ncart,  sizeof(uint64_t)));
         
            fprintf(stdout, "%s Shell %zu (def2-SVP):\n",(n==0) ? "Oxygen" : "Hydrogen", i+1);
            fprintf(stdout, "Angular momentum:              %s\n",  isPure ? "spherical" : "cartesian" );
            fprintf(stdout, "L:                             %zu\n", L);
            fprintf(stdout, "Number of primitives:          %zu\n", nprim);
            fprintf(stdout, "Number of basis functions:     %zu\n", nao);
            fprintf(stdout, "Number of pure functions:      %zu\n", npure);
            fprintf(stdout, "Number of cartesian functions: %zu\n", ncart);
            fprintf(stdout, "\n");

            max_L = (max_L > L) ? max_L : L;
            nao_total += nao;
            nprim_total += nprim;
            npure_total += npure;
            ncart_total += ncart;
        }
    }

    /* Declare the AO basis handle. */
    cuestAOBasis_t basis;

    /* Declare the AO basis parameter handle. */
    cuestAOBasisParameters_t basis_parameters;
    checkCuestErrors(cuestParametersCreate(CUEST_AOBASIS_PARAMETERS, &basis_parameters));

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
    cuestWorkspace_t* persistentWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

    /* The workspace descriptors are no longer needed. */
    free(persistentWorkspaceDescriptor);
    free(temporaryWorkspaceDescriptor);

    /* Create the AO basis handle. */
    checkCuestErrors(cuestAOBasisCreate(
        handle, 
        numAtoms, 
        numShellsPerAtom, 
        (const cuestAOShell_t*) shells, 
        basis_parameters, 
        persistentWorkspace, 
        temporaryWorkspace, 
        &basis));

    /* The temporary workspace is no longer needed. */
    freeWorkspace(temporaryWorkspace);

    /* The AO basis parameters are no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_AOBASIS_PARAMETERS, 
        basis_parameters));

    /* The AO shell handles can be freed following creation of the basis. */
    for (uint64_t n=0, count=0; n<numAtoms; n++) {
        for (uint64_t i=0; i<numShellsPerAtom[n]; i++, count++) {
            checkCuestErrors(cuestAOShellDestroy(shells[count]));
        }
    }
    free(shells);

    /*
     * From here, the AO basis handle and the persistentWorkspace must be retained
     * in order the use the handle. The shells, shell parameters, workspace
     * descriptors are no longer needed and have already been freed.
     */

    /* 
     * Query the AO basis handle to determine the size of the basis. If it was
     * constructed correctly, the sizes will match those calculated from the 
     * array of shells.
     */
    uint64_t natom = 0;
    uint64_t nshell = 0;
    uint64_t nao = 0;
    uint64_t ncart = 0;
    uint64_t nprimitive = 0;
    int32_t is_pure = 0;

    checkCuestErrors(cuestQuery(handle, CUEST_AOBASIS, basis, CUEST_AOBASIS_NUM_ATOM,      &natom,      sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOBASIS, basis, CUEST_AOBASIS_NUM_SHELL,     &nshell,     sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOBASIS, basis, CUEST_AOBASIS_NUM_AO,        &nao,        sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOBASIS, basis, CUEST_AOBASIS_NUM_CART,      &ncart,      sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOBASIS, basis, CUEST_AOBASIS_NUM_PRIMITIVE, &nprimitive, sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOBASIS, basis, CUEST_AOBASIS_MAX_L,         &max_L,      sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOBASIS, basis, CUEST_AOBASIS_IS_PURE,       &is_pure,    sizeof(int32_t)));

    fprintf(stdout,"AO Basis from handle:\n");
    fprintf(stdout,"%-10s = %6zu\n", "natom",      natom);
    fprintf(stdout,"%-10s = %6zu\n", "nshell",     nshell);
    fprintf(stdout,"%-10s = %6zu\n", "nao",        nao);
    fprintf(stdout,"%-10s = %6zu\n", "ncart",      ncart);
    fprintf(stdout,"%-10s = %6zu\n", "nprimitive", nprimitive);
    fprintf(stdout,"%-10s = %6zu\n", "max_L",      max_L);
    fprintf(stdout,"%-10s = %6s\n",  "is_pure",    is_pure ? "true" : "false");

    /* Destroy the AO basis handle. */
    checkCuestErrors(cuestAOBasisDestroy(basis));
    freeWorkspace(persistentWorkspace);

    /* Destroy the cuEST handle. */
    checkCuestErrors(cuestDestroy(handle));

    return 0;
}
