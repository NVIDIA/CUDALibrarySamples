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

#ifndef COMMON_HELPER_WORKSPACES
#define COMMON_HELPER_WORKSPACES

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "helper_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef CUESTAPI

/**
 * Allocates and initializes a cuestWorkspace_t structure based on a cuestWorkspaceDescriptor_t.
 *
 * This function allocates memory for a new cuestWorkspace_t structure based on the buffer sizes specified
 * in the provided workspaceDescriptor. Memory is allocated for the host buffer using malloc, and for the device
 * buffer using cudaMalloc. If any allocation fails, an error message is printed and the program is terminated.
 *
 * @param[in] workspaceDescriptor  Pointer to a descriptor structure specifying buffer sizes for allocation.
 *                                 If NULL, an error message is printed and the program is terminated.
 *
 * @return Pointer to a fully allocated and populated cuestWorkspace_t structure. The caller is responsible
 *         for freeing all resources using freeWorkspace().
 *
 */
static cuestWorkspace_t* allocateWorkspace(const cuestWorkspaceDescriptor_t* workspaceDescriptor)
{
    /* Check that a valid workspace descriptor has been provided. */
    if (workspaceDescriptor == NULL) {
        fprintf(stderr, "Invalid argument: workspaceDescriptor must not be NULL\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate the workspace structure. */
    cuestWorkspace_t* workspace = (cuestWorkspace_t*) malloc(sizeof(cuestWorkspace_t));
    if (!workspace) {
        fprintf(stderr, "Failed to allocate cuestWorkspace_t struct\n");
        exit(EXIT_FAILURE);
    }

    /* Set the length of the host and device buffers. */
    workspace->hostBufferSizeInBytes   = workspaceDescriptor->hostBufferSizeInBytes;
    workspace->deviceBufferSizeInBytes = workspaceDescriptor->deviceBufferSizeInBytes;
    workspace->hostBuffer              = (uintptr_t) NULL;
    workspace->deviceBuffer            = (uintptr_t) NULL;

    /* Allocate the host buffer if the size is non-zero. */
    if (workspace->hostBufferSizeInBytes) {
        void* hostPtr = (void*) malloc(workspace->hostBufferSizeInBytes);
        if (!hostPtr) {
            fprintf(stderr, "Failed to allocate host buffer\n");
            free(workspace);
            exit(EXIT_FAILURE);
        }
        workspace->hostBuffer = (uintptr_t) hostPtr;
    }

    /* Allocate the device buffer if the size is non-zero. */
    if (workspace->deviceBufferSizeInBytes) {
        void* devicePtr = NULL;
        cudaError_t err = cudaMalloc(&devicePtr, workspace->deviceBufferSizeInBytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate device buffer: %s\n", cudaGetErrorString(err));
            if (workspace->hostBuffer) free((void*) workspace->hostBuffer);
            free(workspace);
            exit(EXIT_FAILURE);
        }
        workspace->deviceBuffer = (uintptr_t) devicePtr;
    }

    /* Return the fully allocated workspace. */
    return workspace;
}

/**
 * Frees all memory associated with a cuestWorkspace_t.
 *
 * This function releases memory for both the host and device buffers pointed to by the supplied workspace,
 * and finally frees the workspace structure itself. Device memory is released using cudaFree, and host memory
 * and the structure are freed using free. If cudaFree fails, an error message is printed and the program is terminated.
 *
 * @param[in] workspace  Pointer to the cuestWorkspace_t structure to be freed. If NULL, the function does nothing.
 *
 */
static void freeWorkspace(cuestWorkspace_t* workspace)
{
    if (!workspace) return;

    /* Frss the host buffer if it is not NULL. */
    if (workspace->hostBuffer) {
        free((void*) workspace->hostBuffer);
    }

    /* Frss the device buffer if it is not NULL. */
    if (workspace->deviceBuffer) {
        cudaError_t err = cudaFree((void*) workspace->deviceBuffer);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
            free(workspace);
            exit(EXIT_FAILURE);
        }
    }

    /* Free the workspace structure. */
    free(workspace);
}

#endif

#ifdef __cplusplus
} 
#endif

#endif /* COMMON_HELPER_WORKSPACES */
