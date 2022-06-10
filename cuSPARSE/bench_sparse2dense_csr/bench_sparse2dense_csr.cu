/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
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
#include <cusparse.h>         // cusparseSparseToDense
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cusp/csr_matrix.h>  // cusp::csr_matrix<>
#include <utils/generate_random_data.h>
#include <utils/helper_string.h>

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    }

int main(const int argc, const char** argv)
{
    // Host problem definition
    int num_rows = getCmdLineArgumentInt(argc, argv, "num_cols");
    int num_cols = getCmdLineArgumentInt(argc, argv, "num_cols");
    float sparsity = getCmdLineArgumentFloat(argc, argv, "sparsity");
    if (argc != 4){
        printf("Usage: %s --num_rows=## --num_cols=## --sparsity=0.##\n", argv[0]);
        return EXIT_FAILURE;
    }
    printf("num_rows: %d\n", num_rows);
    printf("num_cols: %d\n", num_cols);
    printf("sparsity: %f\n", sparsity);

    // ***** END OF HOST PROBLEM DEFINITION *****
    // int   nnz              = 11;
    int nnz = num_rows * num_cols * sparsity;
    int ld = num_cols;
    int dense_size = ld * num_rows;
    // initializing data
    // int   h_csr_offsets[]  = { 0, 3, 4, 7, 9, 11 };
    // int   h_csr_columns[]  = { 0, 2, 3, 1, 0, 2, 3, 1, 3, 1, 2 };
    // float h_csr_values[]   = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
    //                            7.0f, 8.0f, 9.0f, 10.0f, 11.0f };
    cusp::csr_matrix<int, float, cusp::host_memory> h_csr = generate_random_sparse_matrix<cusp::csr_matrix<int, float, cusp::host_memory>>(num_rows, num_cols, nnz);
    cusp::csr_matrix<int, float, cusp::device_memory> d_csr(h_csr);
    // float h_dense[]        = { 0.0f, 0.0f, 0.0f, 0.0f,
    //                            0.0f, 0.0f, 0.0f, 0.0f,
    //                            0.0f, 0.0f, 0.0f, 0.0f,
    //                            0.0f, 0.0f, 0.0f, 0.0f,
    //                            0.0f, 0.0f, 0.0f, 0.0f };
    //--------------------------------------------------------------------------
    // Device memory management
    // int   *d_csr_offsets, *d_csr_columns;
    // float *d_csr_values,
    float *d_dense;
    // CHECK_CUDA( cudaMalloc((void**) &d_csr_offsets,
    //                        (num_rows + 1) * sizeof(int)) )
    // CHECK_CUDA( cudaMalloc((void**) &d_csr_columns, nnz * sizeof(int))         )
    // CHECK_CUDA( cudaMalloc((void**) &d_csr_values,  nnz * sizeof(float))       )
    CHECK_CUDA(cudaMalloc((void **)&d_dense, dense_size * sizeof(float)))
    CHECK_CUDA(cudaMemset(d_dense, 0, dense_size * sizeof(float)))

    // CHECK_CUDA( cudaMemcpy(d_csr_offsets, h_csr_offsets,
    //                        (num_rows + 1) * sizeof(int),
    //                        cudaMemcpyHostToDevice) )
    //  CHECK_CUDA( cudaMemcpy(d_csr_columns, h_csr_columns, nnz * sizeof(int),
    //                         cudaMemcpyHostToDevice) )
    //  CHECK_CUDA( cudaMemcpy(d_csr_values, h_csr_values, nnz * sizeof(float),
    //                         cudaMemcpyHostToDevice) )
    //  CHECK_CUDA( cudaMemcpy(d_dense, h_dense, dense_size * sizeof(float),
    //                         cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, num_rows, num_cols, nnz,
                                     // d_csr_offsets, d_csr_columns,
                                     // d_csr_values,
                                     (void *)thrust::raw_pointer_cast(d_csr.row_offsets.data()),
                                     (void *)thrust::raw_pointer_cast(d_csr.column_indices.data()),
                                     (void *)thrust::raw_pointer_cast(d_csr.values.data()),
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    // Create dense matrix B
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, num_rows, num_cols, ld, d_dense,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW))
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSparseToDense_bufferSize(
        handle, matA, matB,
        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
        &bufferSize))
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

    // execute Sparse to Dense conversion
    CHECK_CUSPARSE(cusparseSparseToDense(handle, matA, matB,
                                         CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                         dBuffer))
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
    CHECK_CUSPARSE(cusparseDestroy(handle))
    //--------------------------------------------------------------------------
    // device result check
    // CHECK_CUDA( cudaMemcpy(h_dense, d_dense, dense_size * sizeof(float),
    //                        cudaMemcpyDeviceToHost) )
    // int correct = 1;
    // for (int i = 0; i < num_rows; i++) {
    //     for (int j = 0; j < num_cols; j++) {
    //         if (h_dense[i * ld + j] != h_dense_result[i * ld + j]) {
    //             correct = 0;
    //             printf("%d %d %f!=%f\n", i, j, h_dense[i * ld + j], h_dense_result[i * ld + j]);
    //             break;
    //         }
    //         else{
    //             printf("%d %d %f==%f\n", i, j, h_dense[i * ld + j]);
    //         }
    //     }
    // }
    // if (correct)
    //     printf("sparse2dense_example test PASSED\n");
    // else
    //     printf("sparse2dense_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer))
    // CHECK_CUDA( cudaFree(d_csr_offsets) )
    // CHECK_CUDA( cudaFree(d_csr_columns) )
    // CHECK_CUDA( cudaFree(d_csr_values) )
    CHECK_CUDA(cudaFree(d_dense))
    return EXIT_SUCCESS;
}
