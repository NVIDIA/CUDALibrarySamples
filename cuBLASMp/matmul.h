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

#pragma once

#include <cublasmp.h>
#include <mpi.h>

#include <algorithm>
#include <cstring>

#include "helpers.h"

static inline cudaDataType_t cublas_scale_type(cublasComputeType_t compute_type, cudaDataType_t output_type)
{
    switch (compute_type)
    {
        case CUBLAS_COMPUTE_32I:
        case CUBLAS_COMPUTE_32I_PEDANTIC: return CUDA_R_32I;
        case CUBLAS_COMPUTE_16F:
        case CUBLAS_COMPUTE_16F_PEDANTIC: return CUDA_R_16F;
        case CUBLAS_COMPUTE_32F:
        case CUBLAS_COMPUTE_32F_PEDANTIC:
        case CUBLAS_COMPUTE_32F_FAST_16F:
        case CUBLAS_COMPUTE_32F_FAST_16BF:
        case CUBLAS_COMPUTE_32F_FAST_TF32:
        case CUBLAS_COMPUTE_32F_EMULATED_16BFX9: return CUDA_R_32F;
        case CUBLAS_COMPUTE_64F:
        case CUBLAS_COMPUTE_64F_PEDANTIC:
#if CUBLAS_VERSION >= 130002
        case CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT:
#endif
            return CUDA_R_64F;
        default: return output_type;
    }
}

static inline cublasLtMatmulMatrixScale_t cublasmp_to_cublaslt_matrix_scale_mode(cublasMpMatmulMatrixScale_t scale_mode)
{
    switch (scale_mode)
    {
        case CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32: return CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3: return CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC32_UE8M0: return CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        case CUBLASMP_MATMUL_MATRIX_SCALE_OUTER_VEC_FP32: return CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC128_FP32: return CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
        case CUBLASMP_MATMUL_MATRIX_SCALE_BLK128x128_FP32: return CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
        default: return cublasLtMatmulMatrixScale_t(-1);
    }
}

static inline cudaDataType_t cublaslt_c_type_for_d_type(cudaDataType_t d_type)
{
    switch (d_type)
    {
        case CUDA_R_4F_E2M1:
        case CUDA_R_8F_E4M3:
        case CUDA_R_8F_E5M2: return CUDA_R_16F;
        default: return d_type;
    }
}

static inline size_t scale_element_size(cublasMpMatmulMatrixScale_t scale_mode)
{
    switch (scale_mode)
    {
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3: return sizeof(__nv_fp8_e4m3);
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC32_UE8M0: return sizeof(__nv_fp8_e8m0);
        default: return sizeof(float);
    }
}

static inline int64_t scale_row_block_size(cublasMpMatmulMatrixScale_t scale_mode)
{
    switch (scale_mode)
    {
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3: return 16;
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC32_UE8M0: return 32;
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC128_FP32:
        case CUBLASMP_MATMUL_MATRIX_SCALE_BLK128x128_FP32: return 128;
        default: return 1;
    }
}

static inline int64_t scale_col_block_size(cublasMpMatmulMatrixScale_t scale_mode)
{
    switch (scale_mode)
    {
        case CUBLASMP_MATMUL_MATRIX_SCALE_BLK128x128_FP32: return 128;
        default: return 1;
    }
}

static inline int64_t scale_factor_rows(cublasMpMatmulMatrixScale_t scale_mode, int64_t rows)
{
    switch (scale_mode)
    {
        case CUBLASMP_MATMUL_MATRIX_SCALE_OUTER_VEC_FP32: return 1;
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3: return (rows + 15) / 16;
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC32_UE8M0: return (rows + 31) / 32;
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC128_FP32:
        case CUBLASMP_MATMUL_MATRIX_SCALE_BLK128x128_FP32: return (rows + 127) / 128;
        default: return 1;
    }
}

static inline int64_t scale_factor_cols(cublasMpMatmulMatrixScale_t scale_mode, int64_t cols)
{
    switch (scale_mode)
    {
        case CUBLASMP_MATMUL_MATRIX_SCALE_BLK128x128_FP32: return (cols + 127) / 128;
        default: return cols;
    }
}

// Byte offset of the scale factor for scale-row `sr` (= element_row / vscale_size) and column `col`
// inside cuBLASLt's Tiled32x4x4 swizzled scale layout for an operand with `rows` rows.
static inline int64_t scale_swizzle_offset(int64_t sr, int64_t col, int64_t rows, int vscale_size)
{
    constexpr int64_t BC = 32, BR = 4, BI = 4;
    constexpr int64_t BCOLS = BC * BR; // 128
    const int64_t ld = roundup((rows + vscale_size - 1) / vscale_size, 4);
    const int64_t block = (sr / BI) + (col / BCOLS) * (ld / BI);
    return block * (BC * BR * BI) + (col % BC) * (BR * BI) + ((col / BC) % BR) * BI + (sr % BI);
}

static inline int64_t scale_layout_offset(
    cublasMpMatmulMatrixScale_t scale_mode,
    int64_t sr,
    int64_t sc,
    int64_t rows,
    int64_t cols)
{
    switch (scale_mode)
    {
        case CUBLASMP_MATMUL_MATRIX_SCALE_OUTER_VEC_FP32: return sc;
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3: return scale_swizzle_offset(sr, sc, rows, 16);
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC32_UE8M0: return scale_swizzle_offset(sr, sc, rows, 32);
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC128_FP32: return sc + sr * roundup(cols, 4);
        case CUBLASMP_MATMUL_MATRIX_SCALE_BLK128x128_FP32: return sr + sc * roundup((rows + 127) / 128, 4);
        default: return 0;
    }
}

template <typename T>
static inline T read_scale_value(const void* scale, int64_t offset)
{
    T value {};
    std::memcpy(&value, static_cast<const uint8_t*>(scale) + offset * sizeof(T), sizeof(T));
    return value;
}

template <>
inline __nv_fp8_e4m3 read_scale_value<__nv_fp8_e4m3>(const void* scale, int64_t offset)
{
    uint8_t bits = 0;
    std::memcpy(&bits, static_cast<const uint8_t*>(scale) + offset, sizeof(bits));
    if ((bits & 0x7fU) == 0x7fU)
    {
        bits = 0x7eU;
    }

    __nv_fp8_e4m3 value {};
    value.__x = bits;
    return value;
}

static inline double scale_value_to_double(
    const void* scale,
    cublasMpMatmulMatrixScale_t scale_mode,
    int64_t row,
    int64_t col,
    int64_t rows,
    int64_t cols)
{
    if (scale == nullptr)
    {
        return 1.0;
    }

    int64_t sr = 0;
    int64_t sc = col;
    switch (scale_mode)
    {
        case CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32:
        {
            return read_scale_value<float>(scale, 0);
        }
        case CUBLASMP_MATMUL_MATRIX_SCALE_OUTER_VEC_FP32:
        {
            break;
        }
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3:
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC32_UE8M0:
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC128_FP32:
        {
            sr = row / scale_row_block_size(scale_mode);
            break;
        }
        case CUBLASMP_MATMUL_MATRIX_SCALE_BLK128x128_FP32:
        {
            sr = row / scale_row_block_size(scale_mode);
            sc = col / scale_col_block_size(scale_mode);
            break;
        }
        default: return 1.0;
    }

    const int64_t offset = scale_layout_offset(scale_mode, sr, sc, rows, cols);
    switch (scale_mode)
    {
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3: return to_double(read_scale_value<__nv_fp8_e4m3>(scale, offset));
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC32_UE8M0: return to_double(read_scale_value<__nv_fp8_e8m0>(scale, offset));
        default: return read_scale_value<float>(scale, offset);
    }
}

struct ScaledAllcloseProlog
{
    const void* scale;
    cublasMpMatmulMatrixScale_t scale_mode;
    int64_t rows;
    int64_t cols;

    template <typename T>
    double operator()(const T* values, int64_t idx, int64_t row, int64_t col) const
    {
        return matrix_value_to_double(values, idx) * scale_value_to_double(scale, scale_mode, row, col, rows, cols);
    }
};

struct DistributedScaledAllcloseProlog
{
    const void* scale;
    cublasMpMatmulMatrixScale_t scale_mode;
    int64_t local_rows;
    int64_t local_cols;
    int64_t full_rows;
    int64_t full_cols;
    size_t local_scale_bytes;
    int nranks;

    template <typename T>
    double operator()(const T* values, int64_t idx, int64_t row, int64_t col) const
    {
        int64_t rank = 0;
        int64_t local_row = row;
        int64_t local_col = col;
        const bool row_split = (full_rows == local_rows * nranks) && (full_cols == local_cols);
        const bool col_split = (full_cols == local_cols * nranks) && (full_rows == local_rows);

        if (row_split)
        {
            rank = std::min<int64_t>(row / local_rows, nranks - 1);
            local_row = row - rank * local_rows;
        }
        else if (col_split)
        {
            rank = std::min<int64_t>(col / local_cols, nranks - 1);
            local_col = col - rank * local_cols;
        }

        const void* local_scale = static_cast<const uint8_t*>(scale) + rank * local_scale_bytes;
        return matrix_value_to_double(values, idx) *
               scale_value_to_double(local_scale, scale_mode, local_row, local_col, local_rows, local_cols);
    }
};

static void* allgather_scale_tensor(
    void* local_scale,
    cublasMpMatmulMatrixScale_t scale_mode,
    int64_t local_rows,
    int64_t local_cols,
    ncclComm_t comm,
    cudaStream_t stream)
{
    if (local_scale == nullptr || scale_mode == CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32)
    {
        return local_scale;
    }

    int nranks = 0;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    const size_t local_scale_bytes = get_scaling_tensor_size(local_rows, local_cols, scale_mode);
    void* gathered_scale = nullptr;
    CUDA_CHECK(cudaMalloc(&gathered_scale, local_scale_bytes * nranks));
    NCCL_CHECK(ncclAllGather(local_scale, gathered_scale, local_scale_bytes, ncclUint8, comm, stream));
    return gathered_scale;
}

// Reconstruct the full-matrix scale tensor (as expected by a single-GPU cuBLASLt reference) from
// per-rank scale tiles. Each rank owns a scale tile indexed in that tile's local layout. A plain
// AllGather only concatenates those tiles; that is not the full cuBLASLt layout for row-split
// Tiled32x4x4 modes or for Hopper's M/N-major Vec128 layout.
static void* gather_scale_tensor(
    void* local_scale,
    cublasMpMatmulMatrixScale_t scale_mode,
    int64_t local_rows,
    int64_t local_cols,
    int64_t full_rows,
    int64_t full_cols,
    ncclComm_t comm,
    cudaStream_t stream,
    int64_t local_col_chunks = 1)
{
    if (local_scale == nullptr || scale_mode == CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32)
    {
        return local_scale;
    }

    int nranks = 0;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    const size_t local_scale_bytes = get_scaling_tensor_size(local_rows, local_cols, scale_mode);
    void* gathered_scale = nullptr;
    CUDA_CHECK(cudaMalloc(&gathered_scale, local_scale_bytes * nranks));
    NCCL_CHECK(ncclAllGather(local_scale, gathered_scale, local_scale_bytes, ncclUint8, comm, stream));

    const bool row_split = (full_rows == local_rows * nranks) && (full_cols == local_cols);
    const bool col_split = (full_cols == local_cols * nranks) && (full_rows == local_rows);

    if (nranks == 1 || (!row_split && !col_split))
    {
        return gathered_scale;
    }

    const size_t element_size = scale_element_size(scale_mode);
    const size_t full_scale_bytes = get_scaling_tensor_size(full_rows, full_cols, scale_mode);
    const int64_t chunks = std::max<int64_t>(1, local_col_chunks);
    const int64_t chunk_cols = std::max<int64_t>(1, local_cols / chunks);
    const size_t chunk_scale_bytes = local_scale_bytes / chunks;
    std::vector<uint8_t> h_gathered(local_scale_bytes * nranks);
    std::vector<uint8_t> h_full(full_scale_bytes, 0);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_gathered.data(), gathered_scale, h_gathered.size(), cudaMemcpyDeviceToHost));

    const int64_t full_scale_rows = scale_factor_rows(scale_mode, full_rows);
    const int64_t full_scale_cols = scale_factor_cols(scale_mode, full_cols);
    const int64_t row_block = scale_row_block_size(scale_mode);
    const int64_t col_block = scale_col_block_size(scale_mode);
    for (int64_t sr = 0; sr < full_scale_rows; sr++)
    {
        for (int64_t sc = 0; sc < full_scale_cols; sc++)
        {
            int64_t rank = 0;
            int64_t src_sr = sr;
            int64_t src_sc = sc;
            if (row_split)
            {
                const int64_t global_row =
                    (scale_mode == CUBLASMP_MATMUL_MATRIX_SCALE_OUTER_VEC_FP32) ? 0 : sr * row_block;
                rank = std::min<int64_t>(global_row / local_rows, nranks - 1);
                src_sr = (global_row - rank * local_rows) / row_block;
            }
            else
            {
                const int64_t global_col = sc * col_block;
                rank = std::min<int64_t>(global_col / local_cols, nranks - 1);
                src_sc = (global_col - rank * local_cols) / col_block;
            }

            const int64_t chunk = std::min<int64_t>((src_sc * col_block) / chunk_cols, chunks - 1);
            const int64_t chunk_sc = (src_sc * col_block - chunk * chunk_cols) / col_block;
            const int64_t src_element = scale_layout_offset(scale_mode, src_sr, chunk_sc, local_rows, chunk_cols);
            const int64_t dst_element = scale_layout_offset(scale_mode, sr, sc, full_rows, full_cols);
            std::copy_n(
                h_gathered.data() + rank * local_scale_bytes + chunk * chunk_scale_bytes + src_element * element_size,
                element_size,
                h_full.data() + dst_element * element_size);
        }
    }

    void* full_scale = nullptr;
    CUDA_CHECK(cudaMalloc(&full_scale, full_scale_bytes));
    CUDA_CHECK(cudaMemcpy(full_scale, h_full.data(), full_scale_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFree(gathered_scale));
    return full_scale;
}

template <typename AType, typename BType, typename DType, typename ScaleType>
static cublasStatus_t cublaslt_matmul(
    cublasLtHandle_t handle,
    cublasOperation_t transA,
    cublasOperation_t transB,
    int64_t m,
    int64_t n,
    int64_t k,
    const ScaleType* alpha,
    const AType* A,
    int64_t lda,
    const BType* B,
    int64_t ldb,
    const ScaleType* beta,
    int64_t ldc,
    DType* D,
    int64_t ldd,
    cublasComputeType_t compute_type,
    cudaStream_t stream,
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT,
    void* bias_pointer = nullptr,
    int64_t bias_batch_stride = 0,
    int32_t bias_data_type = -1,
    void* epilogue_aux_pointer = nullptr,
    int64_t epilogue_aux_ld = 0,
    int64_t epilogue_aux_batch_stride = 0,
    int32_t epilogue_aux_data_type = -1,
    void* epilogue_aux_scale_pointer = nullptr,
    void* epilogue_aux_amax_pointer = nullptr,
    int32_t epilogue_aux_scale_mode = 0,
    void* a_scale = nullptr,
    cublasLtMatmulMatrixScale_t a_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
    void* b_scale = nullptr,
    cublasLtMatmulMatrixScale_t b_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
    void* d_scale = nullptr,
    cublasLtMatmulMatrixScale_t d_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
    void* d_out_scale = nullptr,
    cublasLtMatmulMatrixScale_t d_out_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
    void* amax_d_pointer = nullptr)
{
    cublasLtMatrixLayout_t descA = nullptr;
    cublasLtMatrixLayout_t descB = nullptr;
    cublasLtMatrixLayout_t descC = nullptr;
    cublasLtMatrixLayout_t descD = nullptr;
    cublasLtMatmulDesc_t matmul_desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    void* workspace = nullptr;
    const size_t workspace_size = 32 * 1024 * 1024;

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &descA, CudaTypeTraits<AType>::typeEnum, transA == CUBLAS_OP_N ? m : k, transA == CUBLAS_OP_N ? k : m, lda));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &descB, CudaTypeTraits<BType>::typeEnum, transB == CUBLAS_OP_N ? k : n, transB == CUBLAS_OP_N ? n : k, ldb));
    const cudaDataType_t typeC = cublaslt_c_type_for_d_type(CudaTypeTraits<DType>::typeEnum);
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&descC, typeC, m, n, ldc));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&descD, CudaTypeTraits<DType>::typeEnum, m, n, ldd));
    CUBLAS_CHECK(cublasLtMatmulDescCreate(
        &matmul_desc, compute_type, cublas_scale_type(compute_type, CudaTypeTraits<DType>::typeEnum)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    if (epilogue > CUBLASLT_EPILOGUE_DEFAULT)
    {
        const cublasLtEpilogue_t epilogue_without_aux = static_cast<cublasLtEpilogue_t>(epilogue & 0xFFFFFFFE);
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue_without_aux, sizeof(epilogue_without_aux)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer, sizeof(bias_pointer)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE, &bias_batch_stride, sizeof(bias_batch_stride)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
            &epilogue_aux_pointer,
            sizeof(epilogue_aux_pointer)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &epilogue_aux_ld, sizeof(epilogue_aux_ld)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
            &epilogue_aux_batch_stride,
            sizeof(epilogue_aux_batch_stride)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
            &epilogue_aux_data_type,
            sizeof(epilogue_aux_data_type)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
            &epilogue_aux_scale_pointer,
            sizeof(epilogue_aux_scale_pointer)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER,
            &epilogue_aux_amax_pointer,
            sizeof(epilogue_aux_amax_pointer)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE,
            &epilogue_aux_scale_mode,
            sizeof(epilogue_aux_scale_mode)));
    }

    if (a_scale)
    {
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &a_scale_mode, sizeof(a_scale_mode)));
    }

    if (b_scale)
    {
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &b_scale_mode, sizeof(b_scale_mode)));
    }

    if (d_scale)
    {
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale, sizeof(d_scale)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &d_scale_mode, sizeof(d_scale_mode)));
    }

    if (d_out_scale)
    {
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &d_out_scale, sizeof(d_out_scale)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &d_out_scale_mode, sizeof(d_out_scale_mode)));
    }

    if (amax_d_pointer)
    {
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amax_d_pointer, sizeof(amax_d_pointer)));
    }

    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

    int returned_results = 0;
    cublasLtMatmulHeuristicResult_t heuristic = {};
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        handle, matmul_desc, descA, descB, descC, descD, preference, 1, &heuristic, &returned_results);
    if (status == CUBLAS_STATUS_SUCCESS && returned_results == 0)
    {
        status = CUBLAS_STATUS_NOT_SUPPORTED;
    }

    if (status == CUBLAS_STATUS_SUCCESS)
    {
        status = cublasLtMatmul(
            handle,
            matmul_desc,
            alpha,
            A,
            descA,
            B,
            descB,
            beta,
            nullptr,
            descC,
            D,
            descD,
            &heuristic.algo,
            workspace,
            workspace_size,
            stream);
    }

    CUDA_CHECK(cudaFree(workspace));
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(matmul_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(descD));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(descC));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(descB));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(descA));
    return status;
}

template <typename AType, typename BType, typename DType>
static inline double matmul_default_rtol(cublasComputeType_t compute_type)
{
    double rtol = default_rtol<DType>();
    if (compute_type == CUBLAS_COMPUTE_32F || compute_type == CUBLAS_COMPUTE_32F_PEDANTIC)
    {
        constexpr bool input_is_fp4 = std::is_same_v<AType, __nv_fp4_e2m1> || std::is_same_v<BType, __nv_fp4_e2m1>;
        constexpr bool input_is_fp8 = std::is_same_v<AType, __nv_fp8_e4m3> || std::is_same_v<AType, __nv_fp8_e5m2> ||
                                      std::is_same_v<BType, __nv_fp8_e4m3> || std::is_same_v<BType, __nv_fp8_e5m2>;
        if constexpr (input_is_fp4)
        {
            rtol = 7e-1;
        }
        else if constexpr (input_is_fp8)
        {
            if constexpr (std::is_same_v<DType, __nv_fp8_e4m3> || std::is_same_v<DType, __nv_fp8_e5m2>)
                rtol = 6.5e-1;
            else if constexpr (std::is_same_v<DType, __half>)
                rtol = 2e-1;
            else if constexpr (std::is_same_v<DType, __nv_bfloat16>)
                rtol = 8e-1;
            else
                rtol = 5e-4;
        }
        else if constexpr (std::is_same_v<DType, __half>)
            rtol = 3e-3;
        else if constexpr (std::is_same_v<DType, __nv_bfloat16>)
            rtol = 2e-2;
        else
            rtol = 3e-5;
    }
    return rtol;
}

// Shared correctness check for the matmul_ag / matmul_ar / matmul_rs samples. Each gathers the distributed
// A, B and D operands onto rank 0, recomputes the GEMM with a single-GPU cublasLt reference, and compares.
//
// gather_d_scales controls how the D-side scales are handled: matmul_ag/matmul_rs leave D distributed, so
// d_scale / d_out_scale must be gathered to match the full matrix; matmul_ar leaves the full D replicated on
// every rank after the AllReduce, so its D-side scales are already full and must be passed through unchanged.
template <typename AType, typename BType, typename DType, typename ScaleType>
static bool check_matmul_result(
    const char* name,
    cublasMpHandle_t mp_handle,
    ncclComm_t comm,
    cudaStream_t stream,
    int rank,
    cublasMpGrid_t gridA,
    int nprowA,
    int npcolA,
    int myprowA,
    int mypcolA,
    cublasMpGrid_t gridB,
    int nprowB,
    int npcolB,
    int myprowB,
    int mypcolB,
    cublasMpGrid_t gridD,
    int nprowD,
    int npcolD,
    int myprowD,
    int mypcolD,
    cublasOperation_t transA,
    cublasOperation_t transB,
    int64_t m,
    int64_t n,
    int64_t k,
    const ScaleType* alpha,
    AType* d_A,
    cublasMpMatrixDescriptor_t descA,
    BType* d_B,
    cublasMpMatrixDescriptor_t descB,
    const ScaleType* beta,
    DType* d_D,
    cublasMpMatrixDescriptor_t descD,
    cublasComputeType_t compute_type,
    void* d_a_scale,
    cublasMpMatmulMatrixScale_t a_scale_mode,
    int64_t a_scale_rows,
    int64_t a_scale_cols,
    void* d_b_scale,
    cublasMpMatmulMatrixScale_t b_scale_mode,
    int64_t b_scale_rows,
    int64_t b_scale_cols,
    void* d_d_scale,
    cublasMpMatmulMatrixScale_t d_scale_mode,
    int64_t d_scale_rows,
    int64_t d_scale_cols,
    void* d_d_out_scale,
    cublasMpMatmulMatrixScale_t d_out_scale_mode,
    int64_t d_out_scale_rows,
    int64_t d_out_scale_cols,
    bool gather_d_scales)
{
    const int64_t a_rows = (transA == CUBLAS_OP_N) ? m : k;
    const int64_t a_cols = (transA == CUBLAS_OP_N) ? k : m;
    const int64_t b_rows = (transB == CUBLAS_OP_N) ? k : n;
    const int64_t b_cols = (transB == CUBLAS_OP_N) ? n : k;

    AType* full_A = nullptr;
    BType* full_B = nullptr;
    DType* full_D_result = nullptr;
    DType* full_D_ref = nullptr;
    int64_t full_A_lld = 0;
    int64_t full_B_lld = 0;
    int64_t full_D_result_lld = 0;
    int64_t full_D_ref_lld = 0;

    gather_matrix(
        mp_handle,
        comm,
        stream,
        a_rows,
        a_cols,
        d_A,
        1,
        1,
        descA,
        gridA,
        nprowA,
        npcolA,
        myprowA,
        mypcolA,
        &full_A,
        &full_A_lld);
    gather_matrix(
        mp_handle,
        comm,
        stream,
        b_rows,
        b_cols,
        d_B,
        1,
        1,
        descB,
        gridB,
        nprowB,
        npcolB,
        myprowB,
        mypcolB,
        &full_B,
        &full_B_lld);
    gather_matrix(
        mp_handle,
        comm,
        stream,
        m,
        n,
        d_D,
        1,
        1,
        descD,
        gridD,
        nprowD,
        npcolD,
        myprowD,
        mypcolD,
        &full_D_result,
        &full_D_result_lld);
    constexpr int rsrc = 0;
    constexpr int csrc = 0;
    full_D_ref_lld = std::max<int64_t>(1, cublasMpNumroc(m, m, myprowD, rsrc, nprowD));
    const int64_t full_D_ref_cols = std::max<int64_t>(1, cublasMpNumroc(n, n, mypcolD, csrc, npcolD));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&full_D_ref), full_D_ref_lld * full_D_ref_cols * sizeof(DType)));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    void* full_a_scale =
        gather_scale_tensor(d_a_scale, a_scale_mode, a_scale_rows, a_scale_cols, a_rows, a_cols, comm, stream);
    int nranks = 0;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
    const int b_scale_col_chunks = (b_scale_rows * nranks == b_rows && b_scale_cols == b_cols) ? nranks : 1;
    void* full_b_scale = gather_scale_tensor(
        d_b_scale, b_scale_mode, b_scale_rows, b_scale_cols, b_rows, b_cols, comm, stream, b_scale_col_chunks);
    void* full_d_scale =
        gather_d_scales ? gather_scale_tensor(d_d_scale, d_scale_mode, d_scale_rows, d_scale_cols, m, n, comm, stream)
                        : d_d_scale;
    void* result_d_out_scale =
        gather_d_scales
            ? allgather_scale_tensor(d_d_out_scale, d_out_scale_mode, d_out_scale_rows, d_out_scale_cols, comm, stream)
            : d_d_out_scale;
    void* full_d_out_scale_ref = nullptr;
    if (result_d_out_scale && d_out_scale_mode != CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32)
    {
        CUDA_CHECK(cudaMalloc(&full_d_out_scale_ref, get_scaling_tensor_size(m, n, d_out_scale_mode)));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    bool passed = true;
    if (rank == 0)
    {
        cublasLtHandle_t cublaslt_handle = nullptr;
        CUBLAS_CHECK(cublasLtCreate(&cublaslt_handle));
        CUDA_CHECK(cudaMemsetAsync(full_D_ref, 0, full_D_ref_lld * full_D_ref_cols * sizeof(DType), stream));
        CUBLAS_CHECK(cublaslt_matmul(
            cublaslt_handle,
            transA,
            transB,
            m,
            n,
            k,
            alpha,
            full_A,
            full_A_lld,
            full_B,
            full_B_lld,
            beta,
            full_D_ref_lld,
            full_D_ref,
            full_D_ref_lld,
            compute_type,
            stream,
            CUBLASLT_EPILOGUE_DEFAULT,
            nullptr,
            0,
            -1,
            nullptr,
            0,
            0,
            -1,
            nullptr,
            nullptr,
            0,
            full_a_scale,
            cublasmp_to_cublaslt_matrix_scale_mode(a_scale_mode),
            full_b_scale,
            cublasmp_to_cublaslt_matrix_scale_mode(b_scale_mode),
            full_d_scale,
            cublasmp_to_cublaslt_matrix_scale_mode(d_scale_mode),
            full_d_out_scale_ref ? full_d_out_scale_ref : result_d_out_scale,
            cublasmp_to_cublaslt_matrix_scale_mode(d_out_scale_mode)));
        const double rtol = matmul_default_rtol<AType, BType, DType>(compute_type);
        if (full_d_out_scale_ref)
        {
            const size_t scale_bytes = get_scaling_tensor_size(m, n, d_out_scale_mode);
            std::vector<uint8_t> h_full_d_out_scale_ref(scale_bytes);
            CUDA_CHECK(cudaMemcpyAsync(
                h_full_d_out_scale_ref.data(), full_d_out_scale_ref, scale_bytes, cudaMemcpyDeviceToHost, stream));
            if (gather_d_scales)
            {
                int nranks = 0;
                MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
                const size_t local_scale_bytes =
                    get_scaling_tensor_size(d_out_scale_rows, d_out_scale_cols, d_out_scale_mode);
                std::vector<uint8_t> h_result_d_out_scale(local_scale_bytes * nranks);
                CUDA_CHECK(cudaMemcpyAsync(
                    h_result_d_out_scale.data(),
                    result_d_out_scale,
                    h_result_d_out_scale.size(),
                    cudaMemcpyDeviceToHost,
                    stream));
                passed = allclose_device(
                    name,
                    full_D_result,
                    full_D_result_lld,
                    full_D_ref,
                    full_D_ref_lld,
                    m,
                    n,
                    stream,
                    rtol,
                    default_atol<DType>(),
                    DistributedScaledAllcloseProlog { h_result_d_out_scale.data(),
                                                      d_out_scale_mode,
                                                      d_out_scale_rows,
                                                      d_out_scale_cols,
                                                      m,
                                                      n,
                                                      local_scale_bytes,
                                                      nranks },
                    ScaledAllcloseProlog { h_full_d_out_scale_ref.data(), d_out_scale_mode, m, n });
            }
            else
            {
                std::vector<uint8_t> h_result_d_out_scale(scale_bytes);
                CUDA_CHECK(cudaMemcpyAsync(
                    h_result_d_out_scale.data(), result_d_out_scale, scale_bytes, cudaMemcpyDeviceToHost, stream));
                passed = allclose_device(
                    name,
                    full_D_result,
                    full_D_result_lld,
                    full_D_ref,
                    full_D_ref_lld,
                    m,
                    n,
                    stream,
                    rtol,
                    default_atol<DType>(),
                    ScaledAllcloseProlog { h_result_d_out_scale.data(), d_out_scale_mode, m, n },
                    ScaledAllcloseProlog { h_full_d_out_scale_ref.data(), d_out_scale_mode, m, n });
            }
        }
        else
        {
            passed = allclose_device(
                name,
                full_D_result,
                full_D_result_lld,
                full_D_ref,
                full_D_ref_lld,
                m,
                n,
                stream,
                rtol,
                default_atol<DType>());
        }
        CUBLAS_CHECK(cublasLtDestroy(cublaslt_handle));
    }

    if (full_a_scale != d_a_scale) CUDA_CHECK(cudaFree(full_a_scale));
    if (full_b_scale != d_b_scale) CUDA_CHECK(cudaFree(full_b_scale));
    if (full_d_scale != d_d_scale) CUDA_CHECK(cudaFree(full_d_scale));
    if (result_d_out_scale != d_d_out_scale) CUDA_CHECK(cudaFree(result_d_out_scale));
    if (full_d_out_scale_ref) CUDA_CHECK(cudaFree(full_d_out_scale_ref));
    CUDA_CHECK(cudaFree(full_A));
    CUDA_CHECK(cudaFree(full_B));
    CUDA_CHECK(cudaFree(full_D_result));
    CUDA_CHECK(cudaFree(full_D_ref));

    int passed_int = passed ? 1 : 0;
    MPI_CHECK(MPI_Bcast(&passed_int, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return passed_int != 0;
}
