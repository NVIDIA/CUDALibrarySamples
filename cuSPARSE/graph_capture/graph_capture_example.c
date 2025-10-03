/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpVV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <stdint.h>           // int64_t

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int main(void) {
    //==========================================================================
    // INITIALIZATION
    //==========================================================================
    // Host problem definition
    float   result;
    int64_t size       = 65536; // 2^16
    int64_t nnz        = 65536; // 2^16
    int*    hX_indices = (int*) malloc(nnz * sizeof(int));
    for (int i = 0; i < nnz; i++)
        hX_indices[i] = i;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dX_indices;
    float *dY, *dX_values;
    CHECK_CUDA( cudaMalloc((void**) &dX_indices, nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dX_values,  nnz * sizeof(float))  )
    CHECK_CUDA( cudaMalloc((void**) &dY,         size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dX_indices, hX_indices, nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // Timer setup
    float       cuda_elapsed_ms  = 0;
    float       graph_elapsed_ms = 0;
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA( cudaEventCreate(&start_event) )
    CHECK_CUDA( cudaEventCreate(&stop_event) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;
    size_t               bufferSize = 0;
    void*                dBuffer    = NULL;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse vector X
    CHECK_CUSPARSE( cusparseCreateSpVec(&vecX, size, nnz, dX_indices, dX_values,
                                        CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, size, dY, CUDA_R_32F) )

    CHECK_CUSPARSE( cusparseSpVV_bufferSize(handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            vecX, vecY, &result, CUDA_R_32F,
                                            &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    CHECK_CUSPARSE( cusparseSpVV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 vecX, vecY, &result, CUDA_R_32F, dBuffer) )

    //==========================================================================
    // STANDARD CALL
    //==========================================================================
    CHECK_CUDA( cudaEventRecord(start_event, NULL) )

    // execute SpVV
    for (int j = 0; j < 1000; j++) {
        CHECK_CUSPARSE( cusparseSpVV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     vecX, vecY, &result, CUDA_R_32F, dBuffer) )
    }

    CHECK_CUDA( cudaEventRecord(stop_event, NULL) )
    CHECK_CUDA( cudaEventSynchronize(stop_event) )
    CHECK_CUDA( cudaEventElapsedTime(&cuda_elapsed_ms, start_event,
                                     stop_event) )
    //==========================================================================
    // GRAPH CAPTURE
    //==========================================================================
    cudaGraph_t     graph;
    cudaStream_t    stream;
    cudaGraphExec_t graph_exec;
    CHECK_CUDA( cudaStreamCreate(&stream) )
    CHECK_CUSPARSE( cusparseSetStream(handle, stream) )
    CHECK_CUDA( cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) )

    for (int j = 0; j < 1000; j++) {
        CHECK_CUSPARSE( cusparseSpVV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     vecX, vecY, &result, CUDA_R_32F, dBuffer) )
    }

    CHECK_CUDA( cudaStreamEndCapture(stream, &graph) )
    CHECK_CUDA( cudaDeviceSynchronize() )
    CHECK_CUDA( cudaGetLastError() )
    CHECK_CUDA( cudaGraphInstantiateWithFlags(&graph_exec, graph, 0) )
    //==========================================================================
    // GRAPH EXECUTION
    //==========================================================================
    CHECK_CUDA( cudaEventRecord(start_event, NULL) )

    CHECK_CUDA( cudaGraphLaunch(graph_exec, stream) )

    CHECK_CUDA( cudaEventRecord(stop_event, NULL) )
    CHECK_CUDA( cudaEventSynchronize(stop_event) )
    CHECK_CUDA( cudaEventElapsedTime(&graph_elapsed_ms, start_event,
                                     stop_event) )
    //==========================================================================
    //==========================================================================
    // destroy events
    CHECK_CUDA( cudaEventDestroy(start_event) )
    CHECK_CUDA( cudaEventDestroy(stop_event) )
    // destroy graph
    CHECK_CUDA( cudaDeviceSynchronize() )
    CHECK_CUDA( cudaGraphExecDestroy(graph_exec) )
    CHECK_CUDA( cudaGraphDestroy(graph) )
    CHECK_CUDA( cudaStreamDestroy(stream) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    // device memory deallocation
    CHECK_CUDA( cudaFree(dX_indices) )
    CHECK_CUDA( cudaFree(dX_values)  )
    CHECK_CUDA( cudaFree(dY) )
    CHECK_CUDA( cudaFree(dBuffer) )
    free(hX_indices);
    //--------------------------------------------------------------------------
    // device result check
    float speedup = ((cuda_elapsed_ms - graph_elapsed_ms) / cuda_elapsed_ms)
                    * 100.0f;
    printf("\nStandard call:    %.1f ms"
           "\nGraph Capture:    %.1f ms"
           "\nPerf improvement: %.1f%%\n\n",
           cuda_elapsed_ms, graph_elapsed_ms, speedup);
    //--------------------------------------------------------------------------
    return EXIT_SUCCESS;
}