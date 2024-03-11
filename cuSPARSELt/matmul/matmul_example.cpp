/******************************************************************************
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 ******************************************************************************/

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header
#include <cstdio>             // printf
#include <cstdlib>            // std::rand
                              
// #define AB_TYPE_IS_FP8

#ifdef AB_TYPE_IS_FP8 
#include <cuda_fp8.h>
#endif
                              
#ifdef AB_TYPE_IS_FP8
using AB_CUDA_TYPE = __nv_fp8_e4m3;
#else 
using AB_CUDA_TYPE = __half;
#endif

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

constexpr int EXIT_UNSUPPORTED = 2;


int main(void) {
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                       cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                       cudaDevAttrComputeCapabilityMinor, 0) )
    if (!(major_cc == 8 && minor_cc == 0) &&
        !(major_cc == 8 && minor_cc == 6) &&
        !(major_cc == 8 && minor_cc == 9) &&
        !(major_cc == 9 && minor_cc == 0)) {
        std::printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability == 8.0, 8.6, 8.9, 9.0 current: %d.%d\n\n",
                     major_cc, minor_cc);
        return EXIT_UNSUPPORTED;
    }
    // Host problem definition, row-major order
    // bigger sizes may require dynamic allocations
    constexpr int m            = 32;
    constexpr int n            = 32;
    constexpr int k            = 64;
    auto          order        = CUSPARSE_ORDER_ROW;
    auto          opA          = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          opB          = CUSPARSE_OPERATION_TRANSPOSE;
#ifdef AB_TYPE_IS_FP8
    printf("AB_type = e4m3\n");
    auto          type_AB      = CUDA_R_8F_E4M3;
#else
    auto          type_AB      = CUDA_R_16F;
    printf("AB_type = fp16\n");
#endif
    auto          type_C       = CUDA_R_16F;
    auto          compute_type = CUSPARSE_COMPUTE_32F;

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
    auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
    
    auto     A_size         = A_height * lda * sizeof(AB_CUDA_TYPE);
    auto     B_size         = B_height * ldb * sizeof(AB_CUDA_TYPE);
    auto     C_size         = C_height * ldc * sizeof(__half);

    AB_CUDA_TYPE hA[m * k];
    AB_CUDA_TYPE hB[k * n];
    __half hC[m * n] = {};

    for (int i = 0; i < m * k; i++)
        hA[i] = static_cast<AB_CUDA_TYPE>(static_cast<float>(std::rand() % 10));

    for (int i = 0; i < k * n; i++)
        hB[i] = static_cast<AB_CUDA_TYPE>(static_cast<float>(std::rand() % 10));

    for (int i = 0; i < m * n; i++)
        hC[i] = static_cast<__half>(static_cast<float>(std::rand() % 10));

    float alpha = 1.0f;
    float beta  = 2.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    __half *dA, *dB, *dC, *dD, *dA_compressed;
    int    *d_valid;

    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(int)) )
    dD = dC;

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
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
                                            type_AB, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )

    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matB, num_B_rows,
                                            num_B_cols, ldb, alignment,
                                            type_AB, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            type_C, order) )

    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &handle, &matmul, opA, opB,
                                            &matA, &matB, &matC, &matC,
                                            compute_type) )

    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )

    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel))

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
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha,
                                           dA_compressed, dB, &beta,
                                           dC, dD, nullptr,
                                           streams, num_streams) )
    // otherwise, it is possible to set it directly:
    //int alg = 0;
    //CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(
    //                                        &handle, &alg_sel,
    //                                        CUSPARSELT_MATMUL_ALG_CONFIG_ID,
    //                                        &alg, sizeof(alg)))
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    size_t workspace_size;
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel))

    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &plan,
                                                 &workspace_size))
    void* d_workspace;
    CHECK_CUDA( cudaMalloc((void**) &d_workspace, workspace_size) )
    // Perform the matrix multiplication
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
    CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )

    bool A_std_layout = (is_rowmajor != isA_transposed);
    bool B_std_layout = (is_rowmajor != isB_transposed);
    // host computation
    float hC_result[m * n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum  = 0.0f;
            for (int k1 = 0; k1 < k; k1++) {
                auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
                auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
                sum      += static_cast<float>(hA[posA]) *  // [i][k]
                            static_cast<float>(hB[posB]);   // [k][j]
            }
            auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;
            hC_result[posC] = alpha * sum + beta * static_cast<float>(hC[posC]);  // [i][j]
        }
    }
    // host-device comparison
    int correct = 1;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
            auto device_value = static_cast<float>(hC[pos]);
            auto host_value   = hC_result[pos];
            if (device_value != host_value) {
                // direct floating point comparison is not reliable
                std::printf("(%d, %d):\t%f vs. %f\n",
                            i, j, host_value, device_value);
                correct = 0;
                break;
            }
        }
    }

    if (correct) {
        std::printf("matmul_example test PASSED\n");
    }
    else {
        std::printf("matmul_example test FAILED: wrong result\n");
    }

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )
    CHECK_CUDA( cudaFree(d_workspace) )
    CHECK_CUDA( cudaFree(dA_compressedBuffer) )

    return EXIT_SUCCESS;
}
