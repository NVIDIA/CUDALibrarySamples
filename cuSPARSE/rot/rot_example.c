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
#include <cusparse.h>         // cusparseRot
#include <math.h>             // abs
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

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
    // Host problem definition
    int   size         = 8;
    int   nnz          = 4;
    int   hX_indices[] = { 0, 3, 4, 7 };
    float hX_values[]  = { 1.0f, 2.0f, 3.0f, 4.0f };
    float hY[]         = { 1.0f, 2.0f, 3.0f, 4.0f,
                           5.0f, 6.0f, 7.0f, 8.0f};
    float hX_result[]  = { 1.366025f, 4.464100f, 5.830125f, 8.928200f };
    float hY_result[]  = { -0.366025f, 2.0f, 3.0f, 0.267950f,
                           -0.098075f, 6.0f, 7.0f, 0.535900f };
    float c_coeff      = 0.5f;
    float s_coeff      = 0.866025f;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dX_indices;
    float *dY, *dX_values;
    CHECK_CUDA( cudaMalloc((void**) &dX_indices, nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dX_values,  nnz * sizeof(float))  )
    CHECK_CUDA( cudaMalloc((void**) &dY,         size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dX_indices, hX_indices, nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX_values, hX_values, nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, hY, size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse vector X
    CHECK_CUSPARSE( cusparseCreateSpVec(&vecX, size, nnz, dX_indices, dX_values,
                                        CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, size, dY, CUDA_R_32F) )

    // execute Rot
    CHECK_CUSPARSE( cusparseRot(handle, &c_coeff, &s_coeff, vecX, vecY) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, size * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hX_values, dX_values, nnz * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < size; i++) {
        if (fabsf(hY[i] - hY_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < nnz; i++) {
        if (fabsf(hX_values[i] - hX_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("rot_example test PASSED\n");
    else
        printf("rot_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dX_indices) )
    CHECK_CUDA( cudaFree(dX_values)  )
    CHECK_CUDA( cudaFree(dY) )
    return EXIT_SUCCESS;
}