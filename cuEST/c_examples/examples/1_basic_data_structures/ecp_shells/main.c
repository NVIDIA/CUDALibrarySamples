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
 * This sample shows how to create ECP shells from the
 * radial powers, coefficients, and exponents of each primitive. The iodine
 * def2-SVP-ecp basis set is used as an example. The S, P, D, and F shells will be
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

    /* Create ECP Shells for an iodine atom from def2-svp-ecp.gbs */

    /* This will be the "top shell" */

    /* I     0 */
    /* I-ECP     3     28 */
    /* f potential */
    /*   4 */
    /* 2     19.45860900           -21.84204000 */
    /* 2     19.34926000           -28.46819100 */
    /* 2      4.82376700            -0.24371300 */
    /* 2      4.88431500            -0.32080400 */

    /* These are the remaining shells */

    /* s-f potential */
    /*   7 */
    /* 2     40.01583500            49.99429300 */
    /* 2     17.42974700           281.02531700 */
    /* 2      9.00548400            61.57332600 */
    /* 2     19.45860900            21.84204000 */
    /* 2     19.34926000            28.46819100 */
    /* 2      4.82376700             0.24371300 */
    /* 2      4.88431500             0.32080400 */
    /* p-f potential */
    /*   8 */
    /* 2     15.35546600            67.44284100 */
    /* 2     14.97183300           134.88113700 */
    /* 2      8.96016400            14.67505100 */
    /* 2      8.25909600            29.37566600 */
    /* 2     19.45860900            21.84204000 */
    /* 2     19.34926000            28.46819100 */
    /* 2      4.82376700             0.24371300 */
    /* 2      4.88431500             0.32080400 */
    /* d-f potential */
    /*   10 */
    /* 2     15.06890800            35.43952900 */
    /* 2     14.55532200            53.17605700 */
    /* 2      6.71864700             9.06719500 */
    /* 2      6.45639300            13.20693700 */
    /* 2      1.19177900             0.08933500 */
    /* 2      1.29115700             0.05238000 */
    /* 2     19.45860900            21.84204000 */
    /* 2     19.34926000            28.46819100 */
    /* 2      4.82376700             0.24371300 */
    /* 2      4.88431500             0.32080400 */

    /* f potential - top ECP shell. */
    uint64_t radial_powers_f[4] = {2, 2, 2, 2};
    double coefficients_f[4] = {-21.84204000, -28.46819100, -0.24371300, -0.32080400};
    double exponents_f[4] = {19.45860900, 19.34926000, 4.82376700, 4.88431500};

    /* s-f potential ECP shell. */
    uint64_t radial_powers_s[7] = {2, 2, 2, 2, 2, 2, 2};
    double coefficients_s[7] = {49.99429300, 281.02531700, 61.57332600, 21.84204000, 28.46819100, 0.24371300, 0.32080400};
    double exponents_s[7] = {40.01583500, 17.42974700, 9.00548400, 19.45860900, 19.34926000,  4.82376700,  4.88431500};

    /* p-f potential ECP shell. */
    uint64_t radial_powers_p[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    double coefficients_p[8] = {67.44284100, 134.88113700, 14.67505100, 29.37566600, 21.84204000, 28.46819100, 0.24371300, 0.32080400};
    double exponents_p[8] = {15.35546600, 14.97183300, 8.96016400, 8.25909600,  19.45860900, 19.34926000, 4.82376700, 4.88431500};

    /* d-f potential ECP shell. */
    uint64_t radial_powers_d[10] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    double coefficients_d[10] = {35.43952900, 53.17605700, 9.06719500, 13.20693700, 0.08933500, 0.05238000, 21.84204000, 28.46819100, 0.24371300, 0.32080400};
    double exponents_d[10] = {15.06890800, 14.55532200, 6.71864700, 6.45639300, 1.19177900, 1.29115700, 19.45860900, 19.34926000, 4.82376700, 4.88431500};

    /* Allocate space to store the top shell handle, which has L == max_L. */
    cuestECPShell_t* top_shell = (cuestECPShell_t*) malloc(sizeof(cuestECPShell_t));
    if (!top_shell) {
        fprintf(stderr, "Failed to allocate ECP top shell\n");
        checkCuestErrors(cuestDestroy(handle));
        exit(EXIT_FAILURE);
    }

    /* The pointer to cuestECPShell_t with L < max_L has (max_L - 1) elements */
    size_t num_shells = 3;

    /* Allocate space to store all the ECP shell handles with (L < max_L). */
    cuestECPShell_t* ecp_shells = (cuestECPShell_t*) malloc(num_shells * sizeof(cuestECPShell_t));
    if (!ecp_shells) {
        fprintf(stderr, "Failed to allocate ECP shell array\n");
        checkCuestErrors(cuestDestroy(handle));
        exit(EXIT_FAILURE);
    }

    /* Only one ECP shell parameter handle is needed. */
    /* Create the ECP shell parameter handle. */
    cuestECPShellParameters_t ecpshell_parameters;

    /* Create the ECP shell parameters. */
    checkCuestErrors(cuestParametersCreate(
        CUEST_ECPSHELL_PARAMETERS, 
        &ecpshell_parameters));

    /* Create the ECP top shell. */
    checkCuestErrors(cuestECPShellCreate(
        handle,              ///< cuEST handle 
        3,                   ///< L=3 for an f shell
        4,                   ///< This shell has 4 primitives
        radial_powers_f,     ///< uint64_t* containing the radial powers
        coefficients_f,      ///< double* containing the coefficients
        exponents_f,         ///< double* containing the exponents
        ecpshell_parameters, ///< cuestECPShellParameters_t with default parameters
        &top_shell[0]));     ///< The output cuestECPShell_t

    /* Create the ECP shells. */
    checkCuestErrors(cuestECPShellCreate(
        handle,              ///< cuEST handle 
        0,                   ///< L=0 for an s shell
        7,                   ///< The shell has 5 primitives
        radial_powers_s,     ///< uint64_t* containing the radial powers
        coefficients_s,      ///< double* containing the coefficients
        exponents_s,         ///< double* containing the exponents
        ecpshell_parameters, ///< cuestECPShellParameters_t with default parameters
        &ecp_shells[0]));    ///< The output cuestECPShell_t

    checkCuestErrors(cuestECPShellCreate(
        handle,              ///< cuEST handle 
        1,                   ///< L=1 for a p shell
        8,                   ///< The shell has 8 primitives
        radial_powers_p,     ///< uint64_t* containing the radial powers
        coefficients_p,      ///< double* containing the coefficients
        exponents_p,         ///< double* containing the exponents
        ecpshell_parameters, ///< cuestECPShellParameters_t with default parameters
        &ecp_shells[1]));    ///< The output cuestECPShell_t

    checkCuestErrors(cuestECPShellCreate(
        handle,              ///< cuEST handle 
        2,                   ///< L=2 for a d shell
        10,                  ///< The shell has 10 primitives
        radial_powers_d,     ///< uint64_t* containing the radial powers
        coefficients_d,      ///< double* containing the coefficients
        exponents_d,         ///< double* containing the exponents
        ecpshell_parameters, ///< cuestECPShellParameters_t with default parameters
        &ecp_shells[2]));    ///< The output cuestECPShell_t

    /* Once all the shells are created, the ECP shell parameters can be freed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_ECPSHELL_PARAMETERS, 
        ecpshell_parameters));

    /* Query the ECP shells for the attributes of the shells. */
    uint64_t L = 0, nprimitive = 0;

    checkCuestErrors(cuestQuery(handle, CUEST_ECPSHELL, top_shell[0], CUEST_ECPSHELL_L,             &L,          sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_ECPSHELL, top_shell[0], CUEST_ECPSHELL_NUM_PRIMITIVE, &nprimitive, sizeof(uint64_t)));
 
    fprintf(stdout, "ECP Top shell:\n\n");
    fprintf(stdout, "L:                             %zu\n", L);
    fprintf(stdout, "Number of primitives:          %zu\n", nprimitive);
    fprintf(stdout, "\n");

    checkCuestErrors(cuestQuery(handle, CUEST_ECPSHELL, ecp_shells[0], CUEST_ECPSHELL_L,             &L,          sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_ECPSHELL, ecp_shells[0], CUEST_ECPSHELL_NUM_PRIMITIVE, &nprimitive, sizeof(uint64_t)));

    fprintf(stdout, "ECP 1 shell:\n\n");
    fprintf(stdout, "L:                             %zu\n", L);
    fprintf(stdout, "Number of primitives:          %zu\n", nprimitive);
    fprintf(stdout, "\n");

    checkCuestErrors(cuestQuery(handle, CUEST_ECPSHELL, ecp_shells[1], CUEST_ECPSHELL_L,             &L,          sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_ECPSHELL, ecp_shells[1], CUEST_ECPSHELL_NUM_PRIMITIVE, &nprimitive, sizeof(uint64_t)));

    fprintf(stdout, "ECP 2 shell:\n\n");
    fprintf(stdout, "L:                             %zu\n", L);
    fprintf(stdout, "Number of primitives:          %zu\n", nprimitive);
    fprintf(stdout, "\n");

    checkCuestErrors(cuestQuery(handle, CUEST_ECPSHELL, ecp_shells[2], CUEST_ECPSHELL_L,             &L,          sizeof(uint64_t)));
    checkCuestErrors(cuestQuery(handle, CUEST_ECPSHELL, ecp_shells[2], CUEST_ECPSHELL_NUM_PRIMITIVE, &nprimitive, sizeof(uint64_t)));

    fprintf(stdout, "ECP 3 shell:\n\n");
    fprintf(stdout, "L:                             %zu\n", L);
    fprintf(stdout, "Number of primitives:          %zu\n", nprimitive);
    fprintf(stdout, "\n");

    /* Destroy the ECP shell handles. */
    checkCuestErrors(cuestECPShellDestroy(top_shell[0]));
    free(top_shell);
    checkCuestErrors(cuestECPShellDestroy(ecp_shells[0]));
    checkCuestErrors(cuestECPShellDestroy(ecp_shells[1]));
    checkCuestErrors(cuestECPShellDestroy(ecp_shells[2]));
    free(ecp_shells);

    /* Destroy the cuEST handle. */
    checkCuestErrors(cuestDestroy(handle));

    return 0;
}
