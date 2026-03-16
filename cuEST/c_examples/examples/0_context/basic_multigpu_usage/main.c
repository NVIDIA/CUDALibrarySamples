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
#include <pthread.h>
#include <cuda_runtime.h>

#include <cuest.h>
#include <helper_status.h>

#include <unistd.h>

/* 
 * This sample shows how to create two cuEST handles on two GPUs. The pthreads
 * model is used to manage the threading. 
 */

/* This is a helper struct with function arguments for pthread_create */
typedef struct 
{
    int thread_id;
    cuestHandle_t* handle_ptr;
} thread_args_t;

/* 
 * This function creates a cuEST handle. It sets the device to
 * the ID equal to the thread ID. This cuEST handle is only valid
 * on that GPU.
 */
void* create_cuest_handle(void* arg)
{
    thread_args_t* args = (thread_args_t*) arg;

    // Switch to GPU equal to thread_id
    cudaError_t err = cudaSetDevice(args->thread_id);
    if (err != cudaSuccess) {
        fprintf(
            stderr, "Thread %d: cudaSetDevice failed: %s\n",
            args->thread_id, 
            cudaGetErrorString(err));
        *args->handle_ptr = NULL;
        return NULL;
    }

    fprintf(
        stdout,
        "Thread %d creating cuEST handle on GPU %d.\n", 
        args->thread_id, 
        args->thread_id);

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
    *args->handle_ptr = handle;

    return NULL;
}

void* destroy_cuest_handle(void* arg)
{
    thread_args_t* args = (thread_args_t*) arg;

    // Switch to GPU equal to thread_id
    cudaError_t err = cudaSetDevice(args->thread_id);
    if (err != cudaSuccess) {
        fprintf(
            stderr, "Thread %d: cudaSetDevice failed: %s\n",
            args->thread_id, 
            cudaGetErrorString(err));
        *args->handle_ptr = NULL;
        return NULL;
    }

    fprintf(
        stdout,
        "Thread %d destroying cuEST handle on GPU %d.\n", 
        args->thread_id, 
        args->thread_id);

    /* Destroy the cuEST handle. */
    checkCuestErrors(cuestDestroy(*(args->handle_ptr)));

    return NULL;
}

int main(int argc, char **argv)
{
    /* Declare two threads and cuEST handles for each. */
    pthread_t thread1, thread2;
    cuestHandle_t handles[2];
    thread_args_t args[2];

    args[0].thread_id = 0;
    args[0].handle_ptr = &handles[0];

    args[1].thread_id = 1;
    args[1].handle_ptr = &handles[1];

    /* Create threads to initialize handles and set GPU device. */
    if (pthread_create(&thread1, NULL, create_cuest_handle, &args[0]) != 0) {
        fprintf(stderr, "Failed to create thread 1\n");
        exit(EXIT_FAILURE);
    }
    if (pthread_create(&thread2, NULL, create_cuest_handle, &args[1]) != 0) {
        fprintf(stderr, "Failed to create thread 2\n");
        exit(EXIT_FAILURE);
    }

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    /* 
     * Here, two cuEST handles have been initialized -- one on GPU 0 and one on
     * GPU 1. Additional calls to cuEST can be made using a similar threading
     * model and using the cuEST handles.
     */

    /* Create threads to destroy handles with GPU device set correctly. */
    if (pthread_create(&thread1, NULL, destroy_cuest_handle, &args[0]) != 0) {
        fprintf(stderr, "Failed to create thread 1\n");
        exit(EXIT_FAILURE);
    }
    if (pthread_create(&thread2, NULL, destroy_cuest_handle, &args[1]) != 0) {
        fprintf(stderr, "Failed to create thread 2\n");
        exit(EXIT_FAILURE);
    }

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    return 0;
}

