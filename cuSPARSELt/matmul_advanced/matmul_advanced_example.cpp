/*
 * Copyright 1993-2023 NVIDIA Corporation.  All rights reserved.
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
#include <algorithm>          // std::min
#include <cstdio>             // printf
#include <cstdlib>            // std::rand
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header

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

void print_matrix(const __half* matrix,
                  int64_t       height,
                  int64_t       width,
                  int64_t       ld,
                  int           num_batches,
                  int64_t       batch_stride) {
    for (int b = 0; b < num_batches; b++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                printf("%3.0f",
                     static_cast<float>(matrix[b * batch_stride + i * ld + j]));
            }
            printf("\n");
        }
        if (b == num_batches - 1)
            printf("================================================\n");
        else
            printf("------------------------------------------------\n");
    }
    printf("\n");
}

__half random_half_gen() {
    return static_cast<__half>(static_cast<float>(std::rand() % 10));
}

constexpr int EXIT_UNSUPPORTED = 2;

int main(void) {
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                       cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                       cudaDevAttrComputeCapabilityMinor, 0) )
    if (!(major_cc == 8 && minor_cc == 0) &&
        !(major_cc == 8 && minor_cc == 6) &&
        !(major_cc == 8 && minor_cc == 9)) {
        std::printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability == 8.0, 8.6, 8.9 current: %d.%d\n\n",
                     major_cc, minor_cc);
        return EXIT_UNSUPPORTED;
    }
    constexpr bool print_sparse_matrix = true;
    // Host problem definition, row-major order
    constexpr int     num_batches   = 2;
    constexpr int64_t m             = 16;
    constexpr int64_t n             = 16;
    constexpr int64_t k             = 16;
    constexpr int64_t batch_strideA = m * k + 128;
    constexpr int64_t batch_strideB = k * n + 128;
    constexpr int64_t batch_strideC = m * n + 128;
    constexpr auto    order         = CUSPARSE_ORDER_ROW;
    constexpr auto    opA           = CUSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr auto    opB           = CUSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr auto    type          = CUDA_R_16F;
    constexpr auto    compute_type  = CUSPARSE_COMPUTE_16F;

    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    auto     num_A_rows     = (isA_transposed) ? k : m;
    auto     num_A_cols     = (isA_transposed) ? m : k;
    auto     num_B_rows     = (isB_transposed) ? n : k;
    auto     num_B_cols     = (isB_transposed) ? k : n;
    auto     num_C_rows     = m;
    auto     num_C_cols     = n;
    unsigned alignment      = 16;
    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
    auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;
    auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;
    auto     A_width        = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     B_width        = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     A_size         = num_batches * batch_strideA;
    auto     B_size         = num_batches * batch_strideB;
    auto     C_size         = num_batches * batch_strideC;
    auto     A_size_bytes   = num_batches * batch_strideA * sizeof(__half);
    auto     B_size_bytes   = num_batches * batch_strideB * sizeof(__half);
    auto     C_size_bytes   = num_batches * batch_strideC * sizeof(__half);
    auto     hA             = new __half[A_size];
    auto     hB             = new __half[B_size];
    auto     hC             = new __half[C_size]();
    for (int b = 0; b < num_batches; b++) {
        for (int i = 0; i < A_height; i++) {
            for (int j = 0; j < A_width; j++)
                hA[b * batch_strideA + i * lda + j] = random_half_gen();
        }
    }
    for (int b = 0; b < num_batches; b++) {
        for (int i = 0; i < B_height; i++) {
            for (int j = 0; j < B_width; j++)
                hB[b * batch_strideB + i * ldb + j] = random_half_gen();
        }
    }
    if (print_sparse_matrix)
        print_matrix(hA, A_height, A_width, lda, num_batches, batch_strideA);
    float alpha = 1.0f;
    float beta  = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    __half *dA, *dB, *dC, *dD, *dA_compressed;
    int    *d_valid;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size_bytes) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size_bytes) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size_bytes) )
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(int)) )
    dD = dC;

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size_bytes, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size_bytes, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size_bytes, cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;
    CHECK_CUSPARSE( cusparseLtInit(&handle) )
    // matrix descriptor initialization
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &handle, &matA, num_A_rows,
                                            num_A_cols, lda, alignment,
                                            type, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matB, num_B_rows,
                                            num_B_cols, ldb, alignment,
                                            type, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            type, order) )
    //--------------------------------------------------------------------------
    // SET NUM BATCHES
    CHECK_CUSPARSE( cusparseLtMatDescSetAttribute(&handle, &matA,
                                            CUSPARSELT_MAT_NUM_BATCHES,
                                            &num_batches, sizeof(num_batches)) )
    CHECK_CUSPARSE( cusparseLtMatDescSetAttribute(&handle, &matB,
                                            CUSPARSELT_MAT_NUM_BATCHES,
                                            &num_batches, sizeof(num_batches)) )
    CHECK_CUSPARSE( cusparseLtMatDescSetAttribute(&handle, &matC,
                                            CUSPARSELT_MAT_NUM_BATCHES,
                                            &num_batches, sizeof(num_batches)) )
    //--------------------------------------------------------------------------
    // SET BATCH STRIDE
    // if batch_strideA = 0, the matrix multiplication performs a broadcast of
    // the matrix A
    CHECK_CUSPARSE(  cusparseLtMatDescSetAttribute(&handle, &matA,
                                                CUSPARSELT_MAT_BATCH_STRIDE,
                                                &batch_strideA,
                                                sizeof(batch_strideA)) )
    CHECK_CUSPARSE(  cusparseLtMatDescSetAttribute(&handle, &matB,
                                                CUSPARSELT_MAT_BATCH_STRIDE,
                                                &batch_strideB,
                                                sizeof(batch_strideB)) )
    CHECK_CUSPARSE(  cusparseLtMatDescSetAttribute(&handle, &matC,
                                                CUSPARSELT_MAT_BATCH_STRIDE,
                                                &batch_strideC,
                                                sizeof(batch_strideC)) )
    //--------------------------------------------------------------------------
    // MATMUL DESCRIPTOR INITIALIZATION
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(&handle, &matmul, opA, opB,
                                                   &matA, &matB, &matC, &matC,
                                                   compute_type) )
    //--------------------------------------------------------------------------
    // ENABLE ReLU ACTIVATION FUNCTION
    int   true_value       = 1;
    float relu_upper_bound = 15.0f;
    float relu_threshold   = 1.0f;
    CHECK_CUSPARSE( cusparseLtMatmulDescSetAttribute(&handle, &matmul,
                                            CUSPARSELT_MATMUL_ACTIVATION_RELU,
                                            &true_value, sizeof(true_value)) )
    CHECK_CUSPARSE( cusparseLtMatmulDescSetAttribute(
                                &handle, &matmul,
                                CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND,
                                &(relu_upper_bound),
                                sizeof(relu_upper_bound)) )
    CHECK_CUSPARSE( cusparseLtMatmulDescSetAttribute(
                                &handle, &matmul,
                                CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD,
                                &(relu_threshold), sizeof(relu_threshold)) )
    //--------------------------------------------------------------------------
    // SET BIAS POINTER
    void* dBias;
    auto  hBias = new float[m];
    for (int i = 0; i < m; i++)
        hBias[i] = 1.0f;
    CHECK_CUDA( cudaMalloc((void**) &dBias, m * sizeof(float)) )
    CHECK_CUDA( cudaMemcpy(dBias, hBias, m * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUSPARSE( cusparseLtMatmulDescSetAttribute(&handle, &matmul,
                                                CUSPARSELT_MATMUL_BIAS_POINTER,
                                                &dBias, sizeof(dBias)) )

    //--------------------------------------------------------------------------
    // Algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )

    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel) )
    //--------------------------------------------------------------------------
    // Split-K Mode
    int splitK, splitKBuffers;
    cusparseLtSplitKMode_t splitKMode;
    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel,
                                       CUSPARSELT_MATMUL_SPLIT_K,
                                       &splitK, sizeof(splitK)) )

    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel,
                                       CUSPARSELT_MATMUL_SPLIT_K_MODE,
                                       &splitKMode, sizeof(splitKMode)) )

    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel,
                                       CUSPARSELT_MATMUL_SPLIT_K_BUFFERS,
                                       &splitKBuffers, sizeof(splitKBuffers)) )
    auto mode = splitKMode == CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL
                    ? "ONE_KERNEL"  :
                (splitKMode == CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS
                    ? "TWO_KERNELS" : "invalid");
    printf("splitK=%d, splitK-mode=%s, splitK-buffers=%d\n\n",
           splitK, mode, splitKBuffers);

    //--------------------------------------------------------------------------
    // Prune the A matrix (in-place) and check the correctness
    CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
                                         CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
                                              d_valid, stream) )
    int is_valid;
    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
                                cudaMemcpyDeviceToHost, stream) )
    CHECK_CUDA( cudaStreamSynchronize(stream) )
    if (is_valid != 0) {
        std::printf("!!!! The matrix has been pruned in a wrong way. "
                    "cusparseLtMatmul will not provide correct results\n");
        return EXIT_FAILURE;
    }
    //--------------------------------------------------------------------------
    // Compress the A matrix
    size_t compressed_size, compressed_buffer_size;
    void*  dA_compressedBuffer;
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
                                                  &compressed_size,
                                                  &compressed_buffer_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressedBuffer,
                           compressed_buffer_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed,
                                            dA_compressedBuffer,stream) )
    //--------------------------------------------------------------------------
    // Plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel))

    void*  d_workspace    = nullptr;
    size_t workspace_size = 0;
    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &plan,
                                                 &workspace_size) )

    CHECK_CUDA( cudaMalloc((void**) &d_workspace, workspace_size) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Perform the matrix multiplication
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;

    CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                     &beta, dC, dD, d_workspace, streams,
                                     num_streams) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // destroy plan and handle
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )
    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )
    //--------------------------------------------------------------------------
    // device result check
    // matrix A has been pruned
    CHECK_CUDA( cudaMemcpy(hA, dA, A_size_bytes, cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size_bytes, cudaMemcpyDeviceToHost) )

    if (print_sparse_matrix)
        print_matrix(hA, A_height, A_width, lda, num_batches, batch_strideA);
    bool A_std_layout = (is_rowmajor != isA_transposed);
    bool B_std_layout = (is_rowmajor != isB_transposed);
    auto ReLU         = [=](float value) {
        if (value <= relu_threshold)
            return 0.0f;
        return std::min(value, relu_upper_bound);
    };
    // host computation
    auto hC_result = new float[C_size];
    for (int b = 0; b < num_batches; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k1 = 0; k1 < k; k1++) {
                    auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
                    auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
                    posA     += b * batch_strideA;
                    posB     += b * batch_strideB;
                    sum      += static_cast<float>(hA[posA]) *  // [i][k]
                                static_cast<float>(hB[posB]);   // [k][j]
                }
                auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                posC           += b * batch_strideC;
                hC_result[posC] = ReLU(sum + 1.0f /*bias*/);  // [i][j]
            }
        }
    }
    // host-device comparison
    int correct = 1;
    for (int b = 0; b < num_batches; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                pos              += b * batch_strideC;
                auto device_value = static_cast<float>(hC[pos]);
                auto host_value   = hC_result[pos];
                if (device_value != host_value) {
                    // direct floating point comparison is not reliable
                    correct = 0;
                    break;
                }
            }
        }
    }
    if (correct)
        std::printf("matmul_advanced_example test PASSED\n");
    else
        std::printf("matmul_advanced_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // host memory deallocation
    delete[] hA;
    delete[] hB;
    delete[] hC;
    delete[] hC_result;
    delete[] hBias;
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(dBias) )
    CHECK_CUDA( cudaFree(d_valid) )
    CHECK_CUDA( cudaFree(d_workspace) )
    CHECK_CUDA( cudaFree(dA_compressedBuffer) )
    return EXIT_SUCCESS;
}
