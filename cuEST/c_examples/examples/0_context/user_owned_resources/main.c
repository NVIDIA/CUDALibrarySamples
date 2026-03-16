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

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>   

#include <cuest.h>

#include <helper_status.h>

/*
 * This sample shows how to create a CUDA stream, cuBLAS handle, and 
 * cuSolver handle and use these to create a custom cuEST handle. Ownership
 * of the handles is not transferred to the cuEST handle and it is 
 * incumbent on the user to free these handles.
 * 
 * Further, the cuBLAS and cuSolver handles must have their streams set
 * to the CUDA stream provided by the user. Failure to set the streams
 * will result in CUEST_STATUS_EXCEPTION.
 */
int main(int argc, char **argv)
{
    /* Declare and create the CUDA stream. */
    cudaStream_t stream_handle;
    if (cudaStreamCreate(&stream_handle) != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA stream.\n");
        exit(EXIT_FAILURE);
    }

    /* Declare and create the cuBLAS handle. */
    cublasHandle_t cublas_handle;
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to create cuBLAS handle.\n");
        exit(EXIT_FAILURE);
    }

    /* Declare and create the cuSolver handle. */
    cusolverDnHandle_t cusolver_handle;
    if (cusolverDnCreate(&cusolver_handle) != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to create cuSolverDn handle.\n");
        exit(EXIT_FAILURE);
    }

    /* Set the cuBLAS stream. */
    if (cublasSetStream(cublas_handle, stream_handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to set cuBLAS stream.\n");
        exit(EXIT_FAILURE);
    }

    /* Set the cuSolver stream. */
    if (cusolverDnSetStream(cusolver_handle, stream_handle) != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to set cuSolverDn stream.\n");
        exit(EXIT_FAILURE);
    }

    /* Declare the cuEST handle. */
    cuestHandle_t handle;

    /* Declare the parameters for cuEST handle creation. */
    cuestHandleParameters_t handle_parameters;

    /* Create the cuEST handle parameters. */
    checkCuestErrors(cuestParametersCreate(
        CUEST_HANDLE_PARAMETERS, 
        &handle_parameters));

    /* Configure the cuEST handle parameters to pass in the stream, cuBLAS and cuSolver handles. */
    checkCuestErrors(cuestParametersConfigure(
        CUEST_HANDLE_PARAMETERS, 
        handle_parameters,
        CUEST_HANDLE_PARAMETERS_CUDASTREAM,
        &stream_handle,
        sizeof(cudaStream_t)));
    checkCuestErrors(cuestParametersConfigure(
        CUEST_HANDLE_PARAMETERS, 
        handle_parameters,
        CUEST_HANDLE_PARAMETERS_CUBLAS,
        &cublas_handle,
        sizeof(cublasHandle_t)));
    checkCuestErrors(cuestParametersConfigure(
        CUEST_HANDLE_PARAMETERS, 
        handle_parameters,
        CUEST_HANDLE_PARAMETERS_CUSOLVER,
        &cusolver_handle,
        sizeof(cusolverDnHandle_t)));

    /* Create the cuEST handle. */
    checkCuestErrors(cuestCreate(
        handle_parameters, 
        &handle));

    /* The cuEST handle parameters may be destroyed immediately following handle creation.
       This only frees the parameter handle structure, not the stream, cuBLAS and cuSolver handles. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_HANDLE_PARAMETERS, 
        handle_parameters));

    /* 
     * This is where the cuEST handle would normally be used to make additional
     * calls to the cuEST library. Other GPU-based calls should use the same
     * stream, cuBLAS and cuSolver handles. This will ensure that device synchronization
     * errors do not appear.
     */

    /* Destroying the cuEST handle does not destroy the other handles. */
    checkCuestErrors(cuestDestroy(handle));

    /* Free cuSolver and cuBLAS handles and the CUDA stream. */
    cusolverDnDestroy(cusolver_handle);
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream_handle);

    return 0;
}
