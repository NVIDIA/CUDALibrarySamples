/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef CUSPARSE_ENABLE_EXPERIMENTAL_API
#define CUSPARSE_ENABLE_EXPERIMENTAL_API
#endif
#include "cusparse.h"         // cusparseSpMVOp
#include <nvrtc.h>            // nvrtc
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <string.h>           // strcpy, strcat
#include <stdbool.h>

#define CHECK_CUDA(func)                                                       \
    {                                                                          \
        cudaError_t status = (func);                                           \
        if (status != cudaSuccess) {                                           \
            printf("CUDA API failed at line %d with error: %s (%d)\n",         \
                   __LINE__, cudaGetErrorString(status), status);              \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    }

#define CHECK_CUSPARSE(func)                                                   \
    {                                                                          \
        cusparseStatus_t status = (func);                                      \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n",     \
                   __LINE__, cusparseGetErrorString(status), status);          \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    }

#define CHECK_NVRTC(func)                                                      \
    {                                                                          \
        nvrtcResult status_ = (func);                                          \
        if (status_ != NVRTC_SUCCESS) {                                        \
            printf("NVRTC API failed at line %d with error: %s (%d)\n",        \
                   __LINE__, nvrtcGetErrorString(status_), status_);           \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

//------------------------------------------------------------------------------

void nvrtc_compile(int          sm_version,
                   const char*  device_fun_str,
                   const char** extra_options,
                   int          num_extra_options,
                   char**       lto_buffer,
                   size_t*      lto_buffer_size);

void nvrtc_compile(int          sm_version,
                   const char*  device_fun_str,
                   const char** extra_options,
                   int          num_extra_options,
                   char**       lto_buffer,
                   size_t*      lto_buffer_size) {
    nvrtcProgram prog;
    CHECK_NVRTC(nvrtcCreateProgram(&prog, device_fun_str, NULL, 0, NULL, NULL))
    char        arch_str[20];
    const char* arch_str_prefix = "-arch=compute_";
    snprintf(arch_str, sizeof(arch_str), "%s%d", arch_str_prefix, sm_version);

    int          num_options   = 4 + num_extra_options;
    const char** nvrtc_options = (const char**) malloc(num_options *
                                                       sizeof(const char*));
    nvrtc_options[0]           = arch_str;
    nvrtc_options[1]           = "-rdc=true";
    nvrtc_options[2]           = "-dlto";
    nvrtc_options[3]           = "-std=c++11";
    for (int i = 0; i < num_extra_options; i++)
        nvrtc_options[4 + i] = extra_options[i];
    nvrtcResult status = nvrtcCompileProgram(prog, num_options, nvrtc_options);
    free(nvrtc_options);
    if (status != NVRTC_SUCCESS) {
        size_t log_size = 0;
        CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &log_size))
        char* log = (char*) malloc(log_size);
        CHECK_NVRTC(nvrtcGetProgramLog(prog, log))
        printf("@@@ DEVICE CODE:\n%s\n---------------------------------\n",
               device_fun_str);
        printf("@@@ NVRTC LOG:\n%s\n-----------------------------------\n",
               log);
        CHECK_NVRTC(nvrtcDestroyProgram(&prog))
        free(log);
        printf("NVRTC FAILED");
        exit(EXIT_FAILURE);
    }
    CHECK_NVRTC(nvrtcGetLTOIRSize(prog, lto_buffer_size))
    *lto_buffer = (char*) malloc(*lto_buffer_size);
    CHECK_NVRTC(nvrtcGetLTOIR(prog, *lto_buffer))
    CHECK_NVRTC(nvrtcDestroyProgram(&prog))
}

//------------------------------------------------------------------------------
typedef struct {
    double a;
    double b;
} epilogue_data_t;

// device source code for epilogue
const char epilogue_src_code[] =
"struct epilogue_data_t {                                               \n\
    double a = 0;                                                       \n\
    double b = 0;                                                       \n\
};                                                                      \n\
                                                                        \n\
__constant__ epilogue_data_t epilogue_data;                             \n\
__device__ double spmvop_epilogue(int row, double x) {                  \n\
    return row % 2 == 0 ? epilogue_data.a * x : epilogue_data.b * x;    \n\
}";

static const char* epilogue_data_symbol = "epilogue_data";

double epilogue(epilogue_data_t epilogue_data, int64_t row, double x);
double epilogue(epilogue_data_t epilogue_data, int64_t row, double x) {
    return row % 2 == 0 ? epilogue_data.a * x : epilogue_data.b * x;
}

//------------------------------------------------------------------------------

char* get_cuda_header_path_from_envs(void);

char* get_cuda_header_path_from_envs(void) {
    char* cuda_path = getenv("CUDA_PATH");
    if (cuda_path == NULL) {
        return NULL;
    }
    const char* prefix      = "-I";
    const char* include_dir = "/include";
    char* cuda_header_path = (char*) malloc(strlen(prefix) + strlen(cuda_path) +
                                            strlen(include_dir) + 1);
    if (cuda_header_path == NULL) {
        return NULL;
    }
    strcpy(cuda_header_path, prefix);
    strcat(cuda_header_path, cuda_path);
    strcat(cuda_header_path, include_dir);
    return cuda_header_path;
}

//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    bool is_default_epilogue = false;
    if (argc >= 2) {
        is_default_epilogue = (bool)(atoi(argv[1]));
    }

    // Host problem definition
    int   A_num_rows       = 4;
    int   A_num_cols       = 4;
    int   A_nnz            = 9;
    int   hA_offsets[]     = { 0, 3, 4, 7, 9 };
    int   hA_columns[]     = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    double hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    double hX[]            = { 1.0f, 2.0f, 3.0f, 4.0f };
    double hY[]            = { 5.0f, 6.0f, 7.0f, 8.0f };
    double hZ_result[]     = { 0.0f, 0.0f, 0.0f, 0.0f };    // host result
    double hZ_dev_result[] = { 0.0f, 0.0f, 0.0f, 0.0f };    // device result copied to host

    //--------------------------------------------------------------------------
    // Device memory management
    int*  dA_offsets, *dA_columns;
    double* dA_values, *dX, *dY, *dZ;
    int alias_Y_Z = 0;
    CHECK_CUDA(cudaMalloc((void**) &dA_offsets, (A_num_rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**) &dA_values,  A_nnz * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**) &dX,         A_num_cols * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**) &dY,         A_num_rows * sizeof(double)))
    if (alias_Y_Z) {
        dZ = dY;
    } else {
        CHECK_CUDA(cudaMalloc((void**) &dZ, A_num_rows * sizeof(double)))
    }

    CHECK_CUDA(cudaMemcpy(dA_offsets, hA_offsets,
                          (A_num_rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(double),
                          cudaMemcpyHostToDevice))

    CHECK_CUDA(cudaMemcpy(dX, hX, A_num_cols * sizeof(double),
                          cudaMemcpyHostToDevice) )
    CHECK_CUDA(cudaMemcpy(dY, hY, A_num_rows * sizeof(double),
                          cudaMemcpyHostToDevice) )

    struct cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0))
    int sm = prop.major * 10 + prop.minor;
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY, vecZ;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                     dA_offsets, dA_columns, dA_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_64F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecZ, A_num_rows, dZ, CUDA_R_64F) )
    //--------------------------------------------------------------------------
    char * lto_buffer = NULL;
    size_t lto_buffer_size = 0;
    // extra options can be useful for providing cuda header location
    // (e.g. cuComplex.h)
    char* cuda_header_path = get_cuda_header_path_from_envs();
#if defined(_WIN32)
    const char* options[] = {cuda_header_path != NULL ? cuda_header_path
                                                      : "-IC:/cuda/include"};
#else
    const char* options[] = {cuda_header_path != NULL
                                 ? cuda_header_path
                                 : "-I/usr/local/cuda/include"};
#endif
    if (!is_default_epilogue) {
        nvrtc_compile(sm, epilogue_src_code, options, 1, &lto_buffer, &lto_buffer_size);
    }

    // For the following bufferSize() and createDescr() we can use vecX/Y/Z constructed
    // above but it's not necessary. Dummy descriptors is currently sufficient to detect
    // their accidental use.
    size_t buffer_size;
    void* d_buffer;
    cusparseSpMVOpDescr_t descr;
    {
        void* dummy_ptr = (void*)(0x100);
        cusparseDnVecDescr_t dummyX, dummyY, dummyZ;
        CHECK_CUSPARSE( cusparseCreateDnVec(&dummyX, A_num_rows, dummy_ptr, CUDA_R_64F) )
        CHECK_CUSPARSE( cusparseCreateDnVec(&dummyY, A_num_cols, dummy_ptr, CUDA_R_64F) )
        CHECK_CUSPARSE( cusparseCreateDnVec(&dummyZ, A_num_rows, dummy_ptr, CUDA_R_64F) )

        CHECK_CUSPARSE(cusparseSpMVOp_bufferSize(handle,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 matA,
                                                 dummyX,
                                                 dummyY,
                                                 dummyZ,
                                                 CUDA_R_64F,
                                                 &buffer_size));
        CHECK_CUDA(cudaMalloc(&d_buffer, buffer_size));
        CHECK_CUSPARSE(cusparseSpMVOp_createDescr(handle, &descr,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  matA, vecX, vecY, vecZ,
                                                  CUDA_R_64F,
                                                  d_buffer));
    }

    cusparseSpMVOpPlan_t plan;
    CHECK_CUSPARSE(cusparseSpMVOp_createPlan(handle, descr, &plan, lto_buffer, lto_buffer_size));

    epilogue_data_t epilogue_data;
    if (!is_default_epilogue) {
        epilogue_data.a = 2.0f;
        epilogue_data.b = 4.0f;

        CHECK_CUSPARSE(cusparseSpMVOp_setGlobalUserData(
                handle, plan, epilogue_data_symbol, &epilogue_data,
                sizeof(epilogue_data_t)));
    }

    // execute SpMV
    double alpha = 1.0f;
    double beta = 3.0f;
    CHECK_CUSPARSE(cusparseSpMVOp(handle, plan, &alpha, &beta, vecX, vecY, vecZ));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecZ) )
    CHECK_CUSPARSE(cusparseSpMVOp_destroyPlan(plan));
    CHECK_CUSPARSE(cusparseSpMVOp_destroyDescr(descr));
    CHECK_CUSPARSE(cusparseDestroy(handle))
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA(
        cudaMemcpy(hZ_dev_result, dZ, A_num_rows * sizeof(double), cudaMemcpyDeviceToHost))

    int correct = 1;
    double tol = 1e-14;
    for (int i = 0; i < A_num_rows; i++) {
        double sum = 0;
        for (int j = hA_offsets[i]; j < hA_offsets[i + 1]; ++j) {
            int k = hA_columns[j];
            sum += alpha * hA_values[j] * hX[k];
        }
        sum += beta * hY[i];
        if (!is_default_epilogue) {
            hZ_result[i] = epilogue(epilogue_data, i, sum);
        } else {
            hZ_result[i] = sum;
        }

        double err = fabs(hZ_dev_result[i] - hZ_result[i]);
        if (err > tol) {
            printf("%d, %f, %f\n", i, hZ_result[i], hZ_dev_result[i]);
            correct = 0; // direct floating point comparison is not reliable
            break;
        }
    }

    if (correct)
        printf("spmv_csr_example test PASSED\n");
    else
        printf("spmv_csr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // host memory deallocation
    if (cuda_header_path != NULL)
        free(cuda_header_path);
    free(lto_buffer);
    // device memory deallocation
    CHECK_CUDA(cudaFree(dA_offsets))
    CHECK_CUDA(cudaFree(dA_columns))
    CHECK_CUDA(cudaFree(dA_values))
    CHECK_CUDA(cudaFree(dX))
    CHECK_CUDA(cudaFree(dY))
    CHECK_CUDA(cudaFree(d_buffer))
    if (!alias_Y_Z) CHECK_CUDA(cudaFree(dZ))
    return EXIT_SUCCESS;
}

