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
 * This sample shows a simple configuration that will create two cuEST
 * handles, each with their own CUDA streams. This is a way to make 
 * multiple cuEST calls simultaneously on a single GPU.
 *
 * Note that each cuEST handle will contain its own cuSolver handle.
 * The cuSolver handle requires a non-trivial amount of GPU resources
 * and will limit the number of cuEST handles that can be reasonably 
 * created on a single device.
 */
int main(int argc, char **argv)
{
    /* Declare the cuEST handle. */
    cuestHandle_t handle[2];

    /* Declare the parameters for cuEST handle creation. */
    cuestHandleParameters_t handle_parameters;

    /* Create the cuEST handle parameters. This sets reasonable defaults. */
    checkCuestErrors(cuestParametersCreate(
        CUEST_HANDLE_PARAMETERS, 
        &handle_parameters));

    /*
     * Each call to cuestCreate will initialize a unique CUDA stream for that handle.
     * The cuestHandleParameters_t is only used for handle creation and can be
     * used for both calls.
     */

    /* Create the first cuEST handle. */
    checkCuestErrors(cuestCreate(
        handle_parameters, 
        &(handle[0])));

    /* Create the second cuEST handle. */
    checkCuestErrors(cuestCreate(
        handle_parameters, 
        &(handle[1])));

    /* The cuEST handle parameters may be destroyed immediately following handle creation. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_HANDLE_PARAMETERS, 
        handle_parameters));

    /* 
     * This is where the cuEST handle would normally be used to make additional
     * calls to the cuEST library. The best practice would be to use two separate threads
     * to manage the calls on the two streams.
     */

    /* Destroy the cuEST handles when no additional calls to the cuEST library will be made. */
    checkCuestErrors(cuestDestroy(handle[0]));
    checkCuestErrors(cuestDestroy(handle[1]));

    return 0;
}
