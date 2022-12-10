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
#include <cusparse.h>         // cusparseSpMM
#include <nvrtc.h>            // nvrtc
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

#define CHECK_NVRTC(func)                                                      \
{                                                                              \
    nvrtcResult status = (func);                                               \
    if (status != NVRTC_SUCCESS) {                                             \
        printf("NVRTC API failed at line %d with error: %s (%d)\n",            \
               __LINE__, nvrtcGetErrorString(status), status);                 \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

void nvrtc_compile(int          sm_version,
                   const char*  device_fun_str,
                   const char** extra_options,
                   int          num_extra_options,
                   void**       nvvm_buffer,
                   size_t*      nvvm_buffer_size);

void nvrtc_compile(int          sm_version,
                   const char*  device_fun_str,
                   const char** extra_options,
                   int          num_extra_options,
                   void**       nvvm_buffer,
                   size_t*      nvvm_buffer_size) {
    nvrtcProgram prog;
    CHECK_NVRTC( nvrtcCreateProgram(&prog, device_fun_str, NULL, 0, NULL, NULL))
    char arch_str[17]  = "-arch=compute_";
    arch_str[14]       = '0' + sm_version / 10;
    arch_str[15]       = '0' + sm_version % 10;
    arch_str[16]       = '\0';
    int num_options    = 4 + num_extra_options;
    const char** nvrtc_options = (const char**) malloc(num_options *
                                                       sizeof(const char*));
    nvrtc_options[0]   = arch_str;
    nvrtc_options[1]   = "-rdc=true";
    nvrtc_options[2]   = "-dlto";
    nvrtc_options[3]   = "-std=c++11";
    for (int i = 0; i < num_extra_options; i++)
        nvrtc_options[4 + i] = extra_options[i];
    nvrtcResult status = nvrtcCompileProgram(prog, num_options, nvrtc_options);
    free(nvrtc_options);
    if (status != NVRTC_SUCCESS) {
        size_t log_size = 0;
        CHECK_NVRTC( nvrtcGetProgramLogSize(prog, &log_size) )
        char* log = (char*) malloc(log_size);
        CHECK_NVRTC( nvrtcGetProgramLog(prog, log) )
        printf("@@@ DEVICE CODE:\n%s\n---------------------------------\n",
               device_fun_str);
        printf("@@@ NVRTC LOG:\n%s\n-----------------------------------\n",
               log);
        CHECK_NVRTC( nvrtcDestroyProgram(&prog) )
        free(log);
        printf("NVRTC FAILED");
        exit(EXIT_FAILURE);
    }
    CHECK_NVRTC( nvrtcGetNVVMSize(prog, nvvm_buffer_size) )
    *nvvm_buffer = malloc(*nvvm_buffer_size);
    CHECK_NVRTC( nvrtcGetNVVM(prog, *nvvm_buffer) )
    CHECK_NVRTC( nvrtcDestroyProgram(&prog) )
}

//------------------------------------------------------------------------------

const char AddOp[] =
"__device__                                                                  \n\
float add_op(float value1, float value2) {                                   \n\
    return value1 + value2;                                                  \n\
}";

const char MulOp[] =
"__device__                                                                  \n\
float mul_op(float valueA, float valueB) {                                   \n\
    return valueA * valueB;                                                  \n\
}";

const char Epilogue[] =
"__device__                                                                  \n\
float epilogue(float accumulation, float old_C_value) {                      \n\
    return accumulation * 2.0f + old_C_value;                                \n\
}";

//------------------------------------------------------------------------------

int main(void) {
    // Host problem definition
    int   A_num_rows      = 4;
    int   A_num_cols      = 4;
    int   A_nnz           = 9;
    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = 3;
    int   ldb             = B_num_cols;
    int   ldc             = B_num_cols;
    int   B_size          = ldb * B_num_rows;
    int   C_size          = ldc * A_num_rows;
    int   hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    int   hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    float hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              6.0f, 7.0f, 8.0f, 9.0f };
    float hB[]            = {  1.0f,  2.0f,  3.0f,
                               4.0f,  5.0f,  6.0f,
                               7.0f,  8.0f,  9.0f,
                              10.0f, 11.0f, 12.0f };
    float hC[]            = { 1.0f, 1.0f, 1.0f,
                              1.0f, 1.0f, 1.0f,
                              1.0f, 1.0f, 1.0f,
                              1.0f, 1.0f, 1.0f };
    float hC_result[]     = {  91.0f, 103.0f, 115.0f,
                               33.0f,  41.0f,  49.0f,
                              235.0f, 271.0f, 307.0f,
                              245.0f, 279.0f, 313.0f };
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))  )
    CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    struct cudaDeviceProp prop;
    CHECK_CUDA( cudaGetDeviceProperties(&prop, 0) )
    int sm = prop.major * 10 + prop.minor;
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    cusparseSpMMOpPlan_t plan;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    //--------------------------------------------------------------------------
    void* nvvm_buffer_add, *nvvm_buffer_mul, *nvvm_buffer_epilogue;
    size_t nvvm_buffer_add_size, nvvm_buffer_mul_size,
           nvvm_buffer_epilogue_size;
    // extra options can be useful for providing cuda header location
    // (e.g. cuComplex.h)
#if defined(_WIN32)
    const char* options[] = { "-IC:/cuda/include" };
#else
    const char* options[] = { "-I/usr/local/cuda/include" };
#endif
    nvrtc_compile(sm, AddOp, options, 1,
                  &nvvm_buffer_add, &nvvm_buffer_add_size);
    nvrtc_compile(sm, MulOp, options, 1,
                  &nvvm_buffer_mul, &nvvm_buffer_mul_size);
    nvrtc_compile(sm, Epilogue, options, 1,
                  &nvvm_buffer_epilogue, &nvvm_buffer_epilogue_size);

    CHECK_CUSPARSE(
        cusparseSpMMOp_createPlan(handle, &plan,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  matA, matB, matC, CUDA_R_32F,
                                  CUSPARSE_SPMM_OP_ALG_DEFAULT,
                                  nvvm_buffer_add, nvvm_buffer_add_size,
                                  nvvm_buffer_mul, nvvm_buffer_mul_size,
                                  nvvm_buffer_epilogue,
                                  nvvm_buffer_epilogue_size,
                                  &bufferSize) )

    // allocate an external buffer if needed
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMMOp(plan, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        for (int j = 0; j < B_num_cols; j++) {
            if (hC[i * ldc + j] != hC_result[i * ldc + j]) {
                correct = 0; // direct floating point comparison is not reliable
                break;
            }
        }
    }
    if (correct)
        printf("spmm_csr_op_example test PASSED\n");
    else
        printf("spmm_csr_op_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // host memory deallocation
    free(nvvm_buffer_add);
    free(nvvm_buffer_mul);
    free(nvvm_buffer_epilogue);
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    return EXIT_SUCCESS;
}
