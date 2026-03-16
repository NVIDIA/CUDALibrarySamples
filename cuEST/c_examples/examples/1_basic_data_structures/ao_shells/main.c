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

/*
 * This sample shows how to create contracted Gaussian AO shells from the
 * contraction coefficients and exponents of each primitive. The carbon
 * def2-SVP basis set is used as an example. A 1s, 2p and a d shell will be
 * created. These shells are queried for their attributes.
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

    /* Only one AO shell parameter handle is needed. */

    /* Create the AO shell parameter handle. */
    cuestAOShellParameters_t aoshell_parameters;

    /* Create the AO shell parameters. */
    checkCuestErrors(cuestParametersCreate(
        CUEST_AOSHELL_PARAMETERS, 
        &aoshell_parameters));

    /* A carbon 1s shell will be created. */
    double C_1s_exponents[5] = {1238.4016938, 186.29004992, 42.251176346, 11.676557932, 3.5930506482};
    double C_1s_coefficients[5] = {0.0054568832082, 0.040638409211, 0.18025593888, 0.46315121755, 0.44087173314};

    /* Declare the AO shell handle. */
    cuestAOShell_t C_1s_shell;

    /* Create the AO shell. */
    checkCuestErrors(cuestAOShellCreate(
        handle,               ///< cuEST handle 
        1,                    ///< 1 implies pure angular momentum (correct for def2-SVP)
        0,                    ///< L=0 for an s shell
        5,                    ///< The 1s shell has 5 primitives
        C_1s_exponents,       ///< double* containing the exponents
        C_1s_coefficients,    ///< double* containing the coefficients
        aoshell_parameters,   ///< cuestAOShellParameters_t with default parameters
        &C_1s_shell));        ///< The output cuestAOShell_t

    /* A carbon 2p shell will be created. */
    double C_2p_exponents[3] = {9.4680970621, 2.0103545142, 0.54771004707};
    double C_2p_coefficients[3] = {0.038387871728, 0.21117025112, 0.51328172114};

    /* Declare the AO shell handle. */
    cuestAOShell_t C_2p_shell;

    /* Create the AO shell. */
    checkCuestErrors(cuestAOShellCreate(
        handle,               ///< cuEST handle 
        1,                    ///< 1 implies pure angular momentum (correct for def2-SVP)
        1,                    ///< L=1 for a p shell
        3,                    ///< The 2p shell has 3 primitives
        C_2p_exponents,       ///< double* containing the exponents
        C_2p_coefficients,    ///< double* containing the coefficients
        aoshell_parameters,   ///< cuestAOShellParameters_t with default parameters
        &C_2p_shell));        ///< The output cuestAOShell_t

    /* A carbon d polarization function shell will be created. */
    double C_d_exponents[1] = {0.8};
    double C_d_coefficients[1] = {1.0};

    /* Declare the AO shell handle. */
    cuestAOShell_t C_d_shell;

    /* Create the AO shell. */
    checkCuestErrors(cuestAOShellCreate(
        handle,               ///< cuEST handle 
        1,                    ///< 1 implies pure angular momentum (5 d-functions, correct for def2-SVP)
        2,                    ///< L=2 for a d shell
        1,                    ///< The d shell has 1 primitive
        C_d_exponents,        ///< double* containing the exponent
        C_d_coefficients,     ///< double* containing the coefficient
        aoshell_parameters,   ///< cuestAOShellParameters_t with default parameters
        &C_d_shell));         ///< The output cuestAOShell_t

    /* Once all the shells are created, the AO shell parameters can be freed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_AOSHELL_PARAMETERS, 
        aoshell_parameters));

    /* Query the AO shells for the attributes of the shells. */
    int32_t isPure = 0;
    uint64_t L = 0, nao = 0, nprim = 0, npure = 0, ncart = 0;

    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_1s_shell, CUEST_AOSHELL_IS_PURE,       &isPure, sizeof(int32_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_1s_shell, CUEST_AOSHELL_L,             &L,      sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_1s_shell, CUEST_AOSHELL_NUM_PRIMITIVE, &nprim,  sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_1s_shell, CUEST_AOSHELL_NUM_AO,        &nao,    sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_1s_shell, CUEST_AOSHELL_NUM_PURE,      &npure,  sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_1s_shell, CUEST_AOSHELL_NUM_CART,      &ncart,  sizeof(uint64_t)));
 
    fprintf(stdout, "Carbon 1S shell (def2-SVP):\n\n");
    fprintf(stdout, "Angular momentum:              %s\n",  isPure ? "spherical" : "cartesian" );
    fprintf(stdout, "L:                             %zu\n", L);
    fprintf(stdout, "Number of primitives:          %zu\n", nprim);
    fprintf(stdout, "Number of basis functions:     %zu\n", nao);
    fprintf(stdout, "Number of pure functions:      %zu\n", npure);
    fprintf(stdout, "Number of cartesian functions: %zu\n", ncart);
    fprintf(stdout, "\n");

    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_2p_shell, CUEST_AOSHELL_IS_PURE,       &isPure, sizeof(int32_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_2p_shell, CUEST_AOSHELL_L,             &L,      sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_2p_shell, CUEST_AOSHELL_NUM_PRIMITIVE, &nprim,  sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_2p_shell, CUEST_AOSHELL_NUM_AO,        &nao,    sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_2p_shell, CUEST_AOSHELL_NUM_PURE,      &npure,  sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_2p_shell, CUEST_AOSHELL_NUM_CART,      &ncart,  sizeof(uint64_t)));
 
    fprintf(stdout, "Carbon 2p shell (def2-SVP):\n\n");
    fprintf(stdout, "Angular momentum:              %s\n",  isPure ? "spherical" : "cartesian" );
    fprintf(stdout, "L:                             %zu\n", L);
    fprintf(stdout, "Number of primitives:          %zu\n", nprim);
    fprintf(stdout, "Number of basis functions:     %zu\n", nao);
    fprintf(stdout, "Number of pure functions:      %zu\n", npure);
    fprintf(stdout, "Number of cartesian functions: %zu\n", ncart);
    fprintf(stdout, "\n");

    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_d_shell, CUEST_AOSHELL_IS_PURE,       &isPure, sizeof(int32_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_d_shell, CUEST_AOSHELL_L,             &L,      sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_d_shell, CUEST_AOSHELL_NUM_PRIMITIVE, &nprim,  sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_d_shell, CUEST_AOSHELL_NUM_AO,        &nao,    sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_d_shell, CUEST_AOSHELL_NUM_PURE,      &npure,  sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_AOSHELL, C_d_shell, CUEST_AOSHELL_NUM_CART,      &ncart,  sizeof(uint64_t)));
 
    fprintf(stdout, "Carbon d shell (def2-SVP):\n\n");
    fprintf(stdout, "Angular momentum:              %s\n",  isPure ? "spherical" : "cartesian" );
    fprintf(stdout, "L:                             %zu\n", L);
    fprintf(stdout, "Number of primitives:          %zu\n", nprim);
    fprintf(stdout, "Number of basis functions:     %zu\n", nao);
    fprintf(stdout, "Number of pure functions:      %zu\n", npure);
    fprintf(stdout, "Number of cartesian functions: %zu\n", ncart);
    fprintf(stdout, "\n");

    /* Destroy the AO shell handles. */
    checkCuestErrors(cuestAOShellDestroy(C_1s_shell));
    checkCuestErrors(cuestAOShellDestroy(C_2p_shell));
    checkCuestErrors(cuestAOShellDestroy(C_d_shell));

    /* Destroy the cuEST handle. */
    checkCuestErrors(cuestDestroy(handle));

    return 0;
}
