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
 * This sample shows the most basic usage of the cuEST handle. A single 
 * threaded application creates the handle using the default paramaters.
 * This will cause the cuEST to create a CUDA stream, cuBLAS handle and 
 * cuSolver handle. The cuEST handle will own these handles and destroy
 * them when it is destroyed.
 */
int main(int argc, char **argv)
{
    /* Declare the cuEST handle. */
    cuestHandle_t handle;

    /* Declare the parameters for cuEST handle creation. */
    cuestHandleParameters_t handle_parameters;

    /* Create the cuEST handle parameters. This sets reasonable defaults. */
    checkCuestErrors(cuestParametersCreate(
        CUEST_HANDLE_PARAMETERS, 
        &handle_parameters));

    /*
     * Here, it is demonstrated how to check the default value of some 
     * attributes of the cuestHandleParameters_t handle that were set
     * by the call to cuestParametersCreate.
     */
    uint64_t max_gauss_hermite;
    uint64_t max_L_solid_harmonic;
    uint64_t max_rys_points;

    checkCuestErrors(cuestParametersQuery(
        CUEST_HANDLE_PARAMETERS, 
        handle_parameters,
        CUEST_HANDLE_PARAMETERS_MAX_GAUSS_HERMITE,
        &max_gauss_hermite,
        sizeof(uint64_t)));
    checkCuestErrors(cuestParametersQuery(
        CUEST_HANDLE_PARAMETERS, 
        handle_parameters,
        CUEST_HANDLE_PARAMETERS_MAX_L_SOLID_HARMONIC,
        &max_L_solid_harmonic,
        sizeof(uint64_t)));
    checkCuestErrors(cuestParametersQuery(
        CUEST_HANDLE_PARAMETERS, 
        handle_parameters,
        CUEST_HANDLE_PARAMETERS_MAX_RYS,
        &max_rys_points,
        sizeof(uint64_t)));

    fprintf(
        stdout, 
        "Maximum number of Gauss-Hermite quadrature points: %zu\n",
        max_gauss_hermite);
    fprintf(
        stdout, 
        "Maximum angular momentum (L) solid harmonic transformation: %zu\n",
        max_L_solid_harmonic);
    fprintf(
        stdout, 
        "Maximum number of Rys quadrature points: %zu %s\n",
        max_rys_points,
        (max_rys_points==0) ? "(zero requests the largest available table)" : " ");

    /* Create the cuEST handle. */
    checkCuestErrors(cuestCreate(
        handle_parameters, 
        &handle));

    /* The cuEST handle parameters may be destroyed immediately following handle creation. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_HANDLE_PARAMETERS, 
        handle_parameters));

    /* 
     * This is where the cuEST handle would normally be used to make additional
     * calls to the cuEST library.
     */

    /* 
     * Destroy the cuEST handle when no additional calls to the cuEST library will be made. 
     * This destroys the CUDA stream, cuBLAS and cuSolver handles held by the cuEST handle.
     */
    checkCuestErrors(cuestDestroy(handle));

    return 0;
}
