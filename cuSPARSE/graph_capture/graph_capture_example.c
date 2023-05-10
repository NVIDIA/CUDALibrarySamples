/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
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
