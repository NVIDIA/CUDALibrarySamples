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

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <mpi.h>
#include <string.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#define MPI_CHECK(call)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        int status = call;                                                                                             \
        if (status != MPI_SUCCESS)                                                                                     \
        {                                                                                                              \
            fprintf(stderr, "MPI error at %s:%d : %d\n", __FILE__, __LINE__, status);                                  \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define NCCL_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t status = call;                                                                                    \
        if (status != ncclSuccess)                                                                                     \
        {                                                                                                              \
            fprintf(stderr, "NCCL error at %s:%d : %d\n", __FILE__, __LINE__, status);                                 \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            fprintf(stderr, "CUDA error at %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(status));             \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define CUBLASMP_CHECK(call)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasMpStatus_t status = call;                                                                                \
        if (status != CUBLASMP_STATUS_SUCCESS)                                                                         \
        {                                                                                                              \
            fprintf(stderr, "cuBLASMp error at %s:%d : %d\n", __FILE__, __LINE__, status);                             \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define CUBLASMP_CHECK_STATUS(status_value)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        const cublasMpStatus_t cublasmp_status_ = (status_value);                                                      \
        if (cublasmp_status_ != CUBLASMP_STATUS_SUCCESS)                                                               \
        {                                                                                                              \
            fprintf(stderr, "cuBLASMp error at %s:%d : %d\n", __FILE__, __LINE__, cublasmp_status_);                   \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

static inline bool skip_if_cublasmp_not_supported(
    const char* sample_name,
    const char* call_name,
    cublasMpStatus_t status,
    int rank)
{
    if (status != CUBLASMP_STATUS_NOT_SUPPORTED)
    {
        return false;
    }

    if (rank == 0)
    {
        fprintf(stderr, "%s: %s returned NOT_SUPPORTED, skipping\n", sample_name, call_name);
    }
    return true;
}

#define CUBLAS_CHECK(call)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS)                                                                           \
        {                                                                                                              \
            fprintf(stderr, "cuBLAS error at %s:%d : %s\n", __FILE__, __LINE__, cublasGetStatusString(status));        \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

template <typename T>
struct CudaTypeTraits;

#define MAKE_TYPE_TRAITS(T, type_enum)                                                                                 \
    template <>                                                                                                        \
    struct CudaTypeTraits<T>                                                                                           \
    {                                                                                                                  \
        static constexpr cudaDataType_t typeEnum = type_enum;                                                          \
        static constexpr size_t typeSize = sizeof(T);                                                                  \
    };

MAKE_TYPE_TRAITS(__nv_fp4_e2m1, CUDA_R_4F_E2M1);
MAKE_TYPE_TRAITS(__nv_fp8_e4m3, CUDA_R_8F_E4M3);
MAKE_TYPE_TRAITS(__nv_fp8_e5m2, CUDA_R_8F_E5M2);
MAKE_TYPE_TRAITS(__nv_fp8_e8m0, CUDA_R_8F_UE8M0);
MAKE_TYPE_TRAITS(__nv_bfloat16, CUDA_R_16BF);
MAKE_TYPE_TRAITS(__half, CUDA_R_16F);
MAKE_TYPE_TRAITS(float, CUDA_R_32F);
MAKE_TYPE_TRAITS(double, CUDA_R_64F);

#include "matrix_generator.hxx"

cudaDataType_t string_to_cuda_data_type(const char* type)
{
    if (strcmp(type, "fp4_e2m1") == 0)
    {
        return CUDA_R_4F_E2M1;
    }
    else if (strcmp(type, "fp8_e4m3") == 0)
    {
        return CUDA_R_8F_E4M3;
    }
    else if (strcmp(type, "fp8_e5m2") == 0)
    {
        return CUDA_R_8F_E5M2;
    }
    else if (strcmp(type, "bf16") == 0)
    {
        return CUDA_R_16BF;
    }
    else if (strcmp(type, "fp16") == 0)
    {
        return CUDA_R_16F;
    }
    else if (strcmp(type, "fp32") == 0)
    {
        return CUDA_R_32F;
    }
    else if (strcmp(type, "fp64") == 0)
    {
        return CUDA_R_64F;
    }
    else if (strcmp(type, "cfp32") == 0)
    {
        return CUDA_C_32F;
    }
    else if (strcmp(type, "cfp64") == 0)
    {
        return CUDA_C_64F;
    }
    else
    {
        throw std::runtime_error("unsupported datatype");
    }
}

bool string_to_bool(const char* value)
{
    if (strcmp(value, "true") == 0 || strcmp(value, "1") == 0 || strcmp(value, "yes") == 0 || strcmp(value, "on") == 0)
    {
        return true;
    }
    else if (
        strcmp(value, "false") == 0 || strcmp(value, "0") == 0 || strcmp(value, "no") == 0 || strcmp(value, "off") == 0)
    {
        return false;
    }
    else
    {
        throw std::runtime_error("unsupported boolean value");
    }
}

cublasMpMatmulMatrixScale_t string_to_scale_type(const char* scale)
{
    if (scale == nullptr)
    {
        return CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32; // default
    }
    else if (strcmp(scale, "scalar_fp32") == 0)
    {
        return CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32;
    }
    else if (strcmp(scale, "vec16_ue4m3") == 0)
    {
        return CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    }
    else if (strcmp(scale, "vec32_ue8m0") == 0)
    {
        return CUBLASMP_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    }
    else if (strcmp(scale, "outer_vec_fp32") == 0)
    {
        return CUBLASMP_MATMUL_MATRIX_SCALE_OUTER_VEC_FP32;
    }
    else if (strcmp(scale, "vec128_fp32") == 0)
    {
        return CUBLASMP_MATMUL_MATRIX_SCALE_VEC128_FP32;
    }
    else if (strcmp(scale, "blk128x128_fp32") == 0)
    {
        return CUBLASMP_MATMUL_MATRIX_SCALE_BLK128x128_FP32;
    }
    else
    {
        throw std::runtime_error("unsupported scale type");
    }
}

cublasMpEmulationStrategy_t string_to_emulation_strategy(const char* strategy)
{
    if (strategy == nullptr || strcmp(strategy, " ") == 0)
    {
        return cublasMpEmulationStrategy_t(-1);
    }

    if (strcmp(strategy, "default") == 0)
    {
        return CUBLASMP_EMULATION_STRATEGY_DEFAULT;
    }
    else if (strcmp(strategy, "performant") == 0)
    {
        return CUBLASMP_EMULATION_STRATEGY_PERFORMANT;
    }
    else if (strcmp(strategy, "eager") == 0)
    {
        return CUBLASMP_EMULATION_STRATEGY_EAGER;
    }

    return cublasMpEmulationStrategy_t(-1);
}

inline int64_t roundup(int64_t x, int64_t y)
{
    return ((x + y - 1) / y) * y;
}

static inline int ceildiv(int numerator, int denominator)
{
    return (numerator + denominator - 1) / denominator;
}

size_t get_scaling_tensor_size(int64_t m, int64_t n, cublasMpMatmulMatrixScale_t scale_mode)
{
    switch (scale_mode)
    {
        case CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32: return sizeof(float);
        case CUBLASMP_MATMUL_MATRIX_SCALE_OUTER_VEC_FP32: return n * sizeof(float);
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3:
            return roundup((m + 15) / 16, 4) * roundup(n, 128) * sizeof(__nv_fp8_e4m3);
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC32_UE8M0:
            return roundup((m + 31) / 32, 4) * roundup(n, 128) * sizeof(__nv_fp8_e8m0);
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC128_FP32: return ((m + 127) / 128) * roundup(n, 4) * sizeof(float);
        case CUBLASMP_MATMUL_MATRIX_SCALE_BLK128x128_FP32:
            return (roundup((m + 127) / 128, 4) * ((n + 127) / 128)) * sizeof(float);
        default: return 0;
    }
}

void* allocate_and_init_scaling_factors(int64_t m, int64_t n, cublasMpMatmulMatrixScale_t scale_mode, int rank)
{
    size_t scale_size = get_scaling_tensor_size(m, n, scale_mode);
    if (scale_size == 0) return nullptr;

    void* d_scale = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scale, scale_size));
    const int seed = (scale_mode == CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32 ||
                      scale_mode == CUBLASMP_MATMUL_MATRIX_SCALE_OUTER_VEC_FP32)
                         ? 0
                         : rank;

    switch (scale_mode)
    {
        case CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32:
        {
            generate_values(seed, reinterpret_cast<float*>(d_scale), 1, true, 1, 10);
            break;
        }

        case CUBLASMP_MATMUL_MATRIX_SCALE_OUTER_VEC_FP32:
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC128_FP32:
        case CUBLASMP_MATMUL_MATRIX_SCALE_BLK128x128_FP32:
        {
            generate_values(seed, reinterpret_cast<float*>(d_scale), scale_size / sizeof(float), true, 1, 10);
            break;
        }

        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3:
        {
            generate_values(
                seed, reinterpret_cast<__nv_fp8_e4m3*>(d_scale), scale_size / sizeof(__nv_fp8_e4m3), true, 0.25, 1);
            break;
        }

        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC32_UE8M0:
        {
            generate_values(
                seed, reinterpret_cast<__nv_fp8_e8m0*>(d_scale), scale_size / sizeof(__nv_fp8_e8m0), true, 1, 10);
            break;
        }

        default:
        {
            CUDA_CHECK(cudaFree(d_scale));
            return nullptr;
        }
    }

    return d_scale;
}

cublasOperation_t char_to_cublas_operation(char op)
{
    switch (std::tolower(op))
    {
        case 'n': return CUBLAS_OP_N;
        case 't': return CUBLAS_OP_T;
        case 'c': return CUBLAS_OP_C;
        default: throw std::runtime_error("unsupported operation");
    }
}

static inline cublasMpGridLayout_t char_to_grid_layout(char layout)
{
    switch (std::tolower(layout))
    {
        case 'c': return CUBLASMP_GRID_LAYOUT_COL_MAJOR;
        case 'r': return CUBLASMP_GRID_LAYOUT_ROW_MAJOR;
        default: throw std::runtime_error("unsupported grid layout");
    }
}

struct Options
{
    // problem properties
    int m;
    int n;
    int k;
    int mbA;
    int nbA;
    int mbB;
    int nbB;
    int mbC;
    int nbC;
    int ia;
    int ja;
    int ib;
    int jb;
    int ic;
    int jc;

    // grid
    int p;
    int q;
    char grid_layout;
    int gpus_per_process = 1;

    // data types
    cudaDataType_t typeA = CUDA_R_64F;
    cudaDataType_t typeB = CUDA_R_64F;
    cudaDataType_t typeC = CUDA_R_64F;
    cudaDataType_t typeD = CUDA_R_64F;

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;

    // others
    bool verbose = false;
    bool check_result = true;
    int cycles = 1;
    int warmup = 0;

    // FP4/FP8 scaling options
    const char* scaleA = nullptr;
    const char* scaleB = nullptr;
    const char* scaleD = nullptr;
    const char* scaleDOut = nullptr;

    const char* emulationStrategy = " ";

    void printHelp()
    {
        printf("Available options:\n"
               "    -m <int>\n"
               "    -n <int>\n"
               "    -k <int>\n"
               "    -mbA <int>\n"
               "    -nbA <int>\n"
               "    -mbB <int>\n"
               "    -nbB <int>\n"
               "    -mbC <int>\n"
               "    -nbC <int>\n"
               "    -ia <int>\n"
               "    -ja <int>\n"
               "    -ib <int>\n"
               "    -jb <int>\n"
               "    -ic <int>\n"
               "    -jc <int>\n"
               "    -typeA <string> (fp4_e2m1, fp8_e4m3, fp8_e5m2, bf16, fp16, fp32, fp64, cfp32, cfp64)\n"
               "    -typeB <string> (fp4_e2m1, fp8_e4m3, fp8_e5m2, bf16, fp16, fp32, fp64, cfp32, cfp64)\n"
               "    -typeC <string> (fp4_e2m1, fp8_e4m3, fp8_e5m2, bf16, fp16, fp32, fp64, cfp32, cfp64)\n"
               "    -typeD <string> (fp4_e2m1, fp8_e4m3, fp8_e5m2, bf16, fp16, fp32, fp64, cfp32, cfp64)\n"
               "    -transA <char> (n, t, c)\n"
               "    -transB <char> (n, t, c)\n"
               "    -scaleA <string> (scalar_fp32, vec16_ue4m3, vec32_ue8m0, outer_vec_fp32, vec128_fp32, "
               "blk128x128_fp32)\n"
               "    -scaleB <string> (scalar_fp32, vec16_ue4m3, vec32_ue8m0, outer_vec_fp32, vec128_fp32, "
               "blk128x128_fp32)\n"
               "    -scaleD <string> (scalar_fp32, vec16_ue4m3, vec32_ue8m0, outer_vec_fp32, vec128_fp32, "
               "blk128x128_fp32)\n"
               "    -scaleDOut <string> (scalar_fp32, vec16_ue4m3, vec32_ue8m0, outer_vec_fp32, vec128_fp32, "
               "blk128x128_fp32)\n"
               "    -p <int>\n"
               "    -q <int>\n"
               "    -gridLayout <char> (c, r)\n"
               "    -gpus-per-process <int> (number of local GPU worker threads per MPI process; default: 1)\n"
               "    -emulationStrategy <string> (default, performant, eager)\n"
               "    -checkResult <bool> (true, false)\n"
               "    -no-check\n"
               "    -cycles <int>\n"
               "    -warmup <int>\n"
               "    -verbose\n"
               "    -help\n");
    }

    void print()
    {
        printf(
            "Parameters: "
            "m=%d n=%d k=%d "
            "mbA=%d nbA=%d mbB=%d nbB=%d mbC=%d nbC=%d "
            "ia=%d ja=%d ib=%d jb=%d ic=%d jc=%d "
            "typeA=%d typeB=%d typeC=%d typeD=%d "
            "transA=%d transB=%d "
            "scaleA=%s scaleB=%s scaleD=%s scaleDOut=%s "
            "p=%d q=%d gridLayout=%c gpusPerProcess=%d "
            "emulationStrategy=%s "
            "cycles=%d warmup=%d "
            "verbose=%s checkResult=%s\n",
            m,
            n,
            k,
            mbA,
            nbA,
            mbB,
            nbB,
            mbC,
            nbC,
            ia,
            ja,
            ib,
            jb,
            ic,
            jc,
            typeA,
            typeB,
            typeC,
            typeD,
            transA,
            transB,
            scaleA ? scaleA : "none",
            scaleB ? scaleB : "none",
            scaleD ? scaleD : "none",
            scaleDOut ? scaleDOut : "none",
            p,
            q,
            grid_layout,
            gpus_per_process,
            emulationStrategy,
            cycles,
            warmup,
            verbose ? "true" : "false",
            check_result ? "true" : "false");
    }

    void parse(int argc, char** argv)
    {
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i], "-m") == 0)
            {
                m = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-n") == 0)
            {
                n = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-k") == 0)
            {
                k = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-mbA") == 0)
            {
                mbA = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-nbA") == 0)
            {
                nbA = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-mbB") == 0)
            {
                mbB = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-nbB") == 0)
            {
                nbB = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-mbC") == 0)
            {
                mbC = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-nbC") == 0)
            {
                nbC = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-ia") == 0)
            {
                ia = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-ja") == 0)
            {
                ja = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-ib") == 0)
            {
                ib = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-jb") == 0)
            {
                jb = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-ic") == 0)
            {
                ic = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-jc") == 0)
            {
                jc = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-p") == 0)
            {
                p = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-q") == 0)
            {
                q = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-gridLayout") == 0)
            {
                grid_layout = *argv[++i];
            }
            else if (strcmp(argv[i], "-gpus-per-process") == 0)
            {
                if (++i >= argc)
                {
                    fprintf(stderr, "Error: -gpus-per-process expects a positive integer\n");
                    exit(1);
                }
                gpus_per_process = atoi(argv[i]);
            }
            else if (strcmp(argv[i], "-typeA") == 0)
            {
                typeA = string_to_cuda_data_type(argv[++i]);
            }
            else if (strcmp(argv[i], "-typeB") == 0)
            {
                typeB = string_to_cuda_data_type(argv[++i]);
            }
            else if (strcmp(argv[i], "-typeC") == 0)
            {
                typeC = string_to_cuda_data_type(argv[++i]);
            }
            else if (strcmp(argv[i], "-typeD") == 0)
            {
                typeD = string_to_cuda_data_type(argv[++i]);
            }
            else if (strcmp(argv[i], "-transA") == 0)
            {
                transA = char_to_cublas_operation(*argv[++i]);
            }
            else if (strcmp(argv[i], "-transB") == 0)
            {
                transB = char_to_cublas_operation(*argv[++i]);
            }
            else if (strcmp(argv[i], "-verbose") == 0)
            {
                verbose = true;
            }
            else if (strcmp(argv[i], "-cycles") == 0)
            {
                cycles = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-warmup") == 0)
            {
                warmup = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-scaleA") == 0)
            {
                scaleA = argv[++i];
            }
            else if (strcmp(argv[i], "-scaleB") == 0)
            {
                scaleB = argv[++i];
            }
            else if (strcmp(argv[i], "-scaleD") == 0)
            {
                scaleD = argv[++i];
            }
            else if (strcmp(argv[i], "-scaleDOut") == 0)
            {
                scaleDOut = argv[++i];
            }
            else if (strcmp(argv[i], "-emulationStrategy") == 0)
            {
                emulationStrategy = argv[++i];
            }
            else if (strcmp(argv[i], "-checkResult") == 0)
            {
                check_result = string_to_bool(argv[++i]);
            }
            else if (strcmp(argv[i], "-no-check") == 0)
            {
                check_result = false;
            }
            else if (strcmp(argv[i], "-help") == 0)
            {
                printHelp();
                exit(0);
            }
            else
            {
                printf("unknown option: %s\n", argv[i]);
                printHelp();
                exit(1);
            }
        }
    }

    void validate()
    {
        if (ia && mbA && (ia - 1) % mbA != 0)
        {
            fprintf(stderr, "Error: ia must be a multiple of mbA\n");
            exit(1);
        }

        if (ja && nbA && (ja - 1) % nbA != 0)
        {
            fprintf(stderr, "Error: ja must be a multiple of nbA\n");
            exit(1);
        }

        if (ib && mbB && (ib - 1) % mbB != 0)
        {
            fprintf(stderr, "Error: ib must be a multiple of mbB\n");
            exit(1);
        }

        if (jb && nbB && (jb - 1) % nbB != 0)
        {
            fprintf(stderr, "Error: jb must be a multiple of nbB\n");
            exit(1);
        }

        if (ic && mbC && (ic - 1) % mbC != 0)
        {
            fprintf(stderr, "Error: ic must be a multiple of mbC\n");
            exit(1);
        }

        if (jc && nbC && (jc - 1) % nbC != 0)
        {
            fprintf(stderr, "Error: jc must be a multiple of nbC\n");
            exit(1);
        }

        if (gpus_per_process <= 0)
        {
            fprintf(stderr, "Error: -gpus-per-process expects a positive integer\n");
            exit(1);
        }
    }
};

static inline int get_local_device()
{
    int localRank;
    MPI_Comm localComm;

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    return localRank % deviceCount;
}

static inline int get_local_device(int mpiLocalRank, int threadRank, int gpusPerProcess, int deviceCount)
{
    if (deviceCount <= 0)
    {
        fprintf(stderr, "Error: no visible CUDA devices\n");
        exit(EXIT_FAILURE);
    }

    return gpusPerProcess == 1 ? mpiLocalRank % deviceCount : mpiLocalRank * gpusPerProcess + threadRank;
}

static inline bool check_local_device_capacity(
    int mpiLocalRank,
    int localThreadCount,
    int gpusPerProcess,
    int deviceCount)
{
    if (deviceCount <= 0)
    {
        return false;
    }

    if (gpusPerProcess == 1)
    {
        return true;
    }

    return mpiLocalRank * gpusPerProcess + localThreadCount <= deviceCount;
}

static inline int select_local_device(int localDevice)
{
    CUDA_CHECK(cudaSetDevice(localDevice));
    CUDA_CHECK(cudaFree(nullptr));
    return localDevice;
}

static inline int select_local_device(int mpiLocalRank, int threadRank, int gpusPerProcess, int deviceCount)
{
    return select_local_device(get_local_device(mpiLocalRank, threadRank, gpusPerProcess, deviceCount));
}

static inline int select_local_device()
{
    return select_local_device(get_local_device());
}

struct Result
{
    cublasMpStatus_t status = CUBLASMP_STATUS_SUCCESS;
    double elapsed = 0.0;
};

static inline bool status_ok(cublasMpStatus_t status)
{
    return status == CUBLASMP_STATUS_SUCCESS || status == CUBLASMP_STATUS_NOT_SUPPORTED;
}

static inline Result make_result(bool passed, double elapsed = 0.0)
{
    return { passed ? CUBLASMP_STATUS_SUCCESS : CUBLASMP_STATUS_EXECUTION_FAILED, elapsed };
}

static inline Result make_result(cublasMpStatus_t status, double elapsed = 0.0)
{
    return { status, elapsed };
}

struct Comm
{
    int world_rank = 0;
    int world_size = 1;
    int nranks = 0;
    int gpus_per_process = 1;
    int local_thread_count = 0;
    bool setup_succeeded = true;
    ncclUniqueId nccl_id {};
    std::vector<int> ranks;
    std::vector<int> local_devices;
    std::vector<ncclComm_t> comms;
    bool mpi_initialized = false;
    bool mpi_finalized = false;

    Comm() = default;
    Comm(int requiredRanks, int gpusPerProcess)
    {
        nranks = requiredRanks;
        gpus_per_process = gpusPerProcess;

        int provided = MPI_THREAD_SINGLE;
        const int required_threading = gpusPerProcess > 1 ? MPI_THREAD_FUNNELED : MPI_THREAD_SINGLE;
        MPI_CHECK(MPI_Init_thread(nullptr, nullptr, required_threading, &provided));
        mpi_initialized = true;

        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

        if (provided < required_threading)
        {
            if (is_root())
            {
                fprintf(stderr, "Error: MPI does not provide the required thread support\n");
            }
            setup_succeeded = false;
            return;
        }

        const int max_available_ranks = world_size * gpusPerProcess;
        if (nranks <= 0)
        {
            if (is_root())
            {
                fprintf(stderr, "Error: -p and -q must define a positive process grid\n");
            }
            setup_succeeded = false;
            return;
        }

        if (gpusPerProcess == 1 && nranks != world_size)
        {
            if (is_root())
            {
                fprintf(
                    stderr,
                    "Error: process grid requires %d ranks, but the sample was launched with %d MPI processes\n",
                    nranks,
                    world_size);
            }
            setup_succeeded = false;
            return;
        }

        if (nranks > max_available_ranks)
        {
            if (is_root())
            {
                fprintf(
                    stderr,
                    "Error: process grid requires %d ranks, but only %d SPMG ranks are available\n",
                    nranks,
                    max_available_ranks);
            }
            setup_succeeded = false;
            return;
        }

        const int participating_processes = ceildiv(nranks, gpusPerProcess);
        const int color = (world_rank < participating_processes) ? 0 : MPI_UNDEFINED;

        MPI_Comm spmg_comm = MPI_COMM_NULL;
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &spmg_comm));

        if (spmg_comm == MPI_COMM_NULL)
        {
            return;
        }

        int mpi_rank = 0;
        MPI_CHECK(MPI_Comm_rank(spmg_comm, &mpi_rank));

        MPI_Comm local_comm = MPI_COMM_NULL;
        MPI_CHECK(MPI_Comm_split_type(spmg_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm));
        int mpi_local_rank = 0;
        MPI_CHECK(MPI_Comm_rank(local_comm, &mpi_local_rank));
        MPI_CHECK(MPI_Comm_free(&local_comm));

        const int first_rank = mpi_rank * gpusPerProcess;
        local_thread_count = std::min(gpusPerProcess, nranks - first_rank);

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        const int local_device_ok =
            check_local_device_capacity(mpi_local_rank, local_thread_count, gpusPerProcess, device_count) ? 1 : 0;
        int all_device_ok = 0;
        MPI_CHECK(MPI_Allreduce(&local_device_ok, &all_device_ok, 1, MPI_INT, MPI_MIN, spmg_comm));

        if (!all_device_ok)
        {
            if (mpi_rank == 0)
            {
                fprintf(stderr, "Error: not enough visible CUDA devices for requested SPMG placement\n");
            }
            setup_succeeded = false;
            MPI_CHECK(MPI_Comm_free(&spmg_comm));
            return;
        }

        if (mpi_rank == 0)
        {
            NCCL_CHECK(ncclGetUniqueId(&nccl_id));
        }
        MPI_CHECK(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, spmg_comm));

        ranks.resize(local_thread_count);
        local_devices.resize(local_thread_count);
        comms.resize(local_thread_count, nullptr);

        for (int thread_rank = 0; thread_rank < local_thread_count; thread_rank++)
        {
            ranks[thread_rank] = mpi_rank * gpusPerProcess + thread_rank;
            local_devices[thread_rank] = get_local_device(mpi_local_rank, thread_rank, gpusPerProcess, device_count);
        }

        MPI_CHECK(MPI_Comm_free(&spmg_comm));
    }

    Comm(const Comm&) = delete;
    Comm& operator=(const Comm&) = delete;

    Comm(Comm&& other) noexcept { *this = std::move(other); }

    Comm& operator=(Comm&& other) noexcept
    {
        if (this != &other)
        {
            finalize();

            world_rank = other.world_rank;
            world_size = other.world_size;
            nranks = other.nranks;
            gpus_per_process = other.gpus_per_process;
            local_thread_count = other.local_thread_count;
            setup_succeeded = other.setup_succeeded;
            nccl_id = other.nccl_id;
            ranks = std::move(other.ranks);
            local_devices = std::move(other.local_devices);
            comms = std::move(other.comms);
            mpi_initialized = other.mpi_initialized;
            mpi_finalized = other.mpi_finalized;

            other.comms.clear();
            other.mpi_initialized = false;
            other.mpi_finalized = true;
        }
        return *this;
    }

    ~Comm() { finalize(); }

    bool is_root() const { return world_rank == 0; }
    bool is_spmg() const { return gpus_per_process > 1; }

    ncclComm_t init_thread(int threadRank)
    {
        select_local_device(local_devices[threadRank]);
        NCCL_CHECK(ncclCommInitRank(&comms[threadRank], nranks, nccl_id, ranks[threadRank]));
        return comms[threadRank];
    }

    void finalize_thread(int threadRank)
    {
        ncclComm_t& comm = comms[threadRank];
        if (comm)
        {
            select_local_device(local_devices[threadRank]);
            NCCL_CHECK(ncclCommFinalize(comm));
            NCCL_CHECK(ncclCommDestroy(comm));
            comm = nullptr;
        }
    }

    void finalize()
    {
        for (size_t i = 0; i < comms.size(); i++)
        {
            ncclComm_t& comm = comms[i];
            if (comm)
            {
                if (i < local_devices.size())
                {
                    select_local_device(local_devices[i]);
                }
                NCCL_CHECK(ncclCommFinalize(comm));
                NCCL_CHECK(ncclCommDestroy(comm));
                comm = nullptr;
            }
        }

        if (mpi_initialized && !mpi_finalized)
        {
            MPI_CHECK(MPI_Finalize());
            mpi_finalized = true;
        }
    }

    int allreduce_min(int localValue) const
    {
        int globalValue = 0;
        MPI_CHECK(MPI_Allreduce(&localValue, &globalValue, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        return globalValue;
    }

    template <typename Worker>
    Result collective_launch(Worker worker)
    {
        Result local_result;
        std::exception_ptr first_exception = nullptr;

        if (!setup_succeeded)
        {
            local_result.status = CUBLASMP_STATUS_EXECUTION_FAILED;
        }
        else if (local_thread_count > 0)
        {
            std::vector<Result> thread_results(local_thread_count);
            std::vector<std::thread> workers;
            workers.reserve(local_thread_count);
            std::mutex exception_mutex;

            auto run_worker = [&](int thread_rank) {
                bool initialized = false;
                try
                {
                    ncclComm_t nccl_comm = init_thread(thread_rank);
                    initialized = true;
                    thread_results[thread_rank] = worker(nccl_comm);
                }
                catch (...)
                {
                    thread_results[thread_rank].status = CUBLASMP_STATUS_INTERNAL_ERROR;
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    if (!first_exception)
                    {
                        first_exception = std::current_exception();
                    }
                }

                if (initialized)
                {
                    finalize_thread(thread_rank);
                }
            };

            if (local_thread_count == 1)
            {
                run_worker(0);
            }
            else
            {
                for (int thread_rank = 0; thread_rank < local_thread_count; thread_rank++)
                {
                    workers.emplace_back(run_worker, thread_rank);
                }

                for (auto& worker_thread : workers)
                {
                    worker_thread.join();
                }
            }

            local_result.status = first_exception ? CUBLASMP_STATUS_INTERNAL_ERROR : CUBLASMP_STATUS_SUCCESS;
            for (const Result& thread_result : thread_results)
            {
                local_result.elapsed = std::max(local_result.elapsed, thread_result.elapsed);
            }

            for (const Result& thread_result : thread_results)
            {
                if (!status_ok(thread_result.status))
                {
                    local_result.status = CUBLASMP_STATUS_INTERNAL_ERROR;
                    break;
                }
                else if (
                    local_result.status == CUBLASMP_STATUS_SUCCESS &&
                    thread_result.status == CUBLASMP_STATUS_NOT_SUPPORTED)
                {
                    local_result.status = CUBLASMP_STATUS_NOT_SUPPORTED;
                }
            }
        }

        const int local_error = status_ok(local_result.status) ? 0 : 1;
        const int local_waved = local_result.status == CUBLASMP_STATUS_NOT_SUPPORTED ? 1 : 0;
        int global_error = 0;
        int global_waved = 0;
        Result global_result;
        MPI_CHECK(MPI_Allreduce(&local_error, &global_error, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Allreduce(&local_waved, &global_waved, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Allreduce(&local_result.elapsed, &global_result.elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
        if (global_error)
        {
            global_result.status = CUBLASMP_STATUS_INTERNAL_ERROR;
        }
        else if (global_waved)
        {
            global_result.status = CUBLASMP_STATUS_NOT_SUPPORTED;
        }
        else
        {
            global_result.status = CUBLASMP_STATUS_SUCCESS;
        }

        if (first_exception)
        {
            std::rethrow_exception(first_exception);
        }

        return global_result;
    }
};

static inline int get_nccl_rank(ncclComm_t comm)
{
    int rank = 0;
    NCCL_CHECK(ncclCommUserRank(comm, &rank));
    return rank;
}

static bool device_supports_fp8(int localDevice)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, localDevice));
    return prop.major >= 9 || (prop.major == 8 && prop.minor == 9);
}

static bool device_supports_fp8()
{
    return device_supports_fp8(get_local_device());
}

static bool device_supports_fp4(int localDevice)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, localDevice));
    return prop.major >= 10;
}

static bool device_supports_fp4()
{
    return device_supports_fp4(get_local_device());
}

static int local_devices_support_types(const Comm& comm, bool needsFp8, bool needsFp4)
{
    if (!comm.setup_succeeded)
    {
        return 0;
    }

    for (int local_device : comm.local_devices)
    {
        if ((needsFp8 && !device_supports_fp8(local_device)) || (needsFp4 && !device_supports_fp4(local_device)))
        {
            return 0;
        }
    }

    return 1;
}

static inline bool is_fp8(cudaDataType_t type)
{
    return type == CUDA_R_8F_E4M3 || type == CUDA_R_8F_E5M2;
}

static inline bool is_fp4(cudaDataType_t type)
{
    return type == CUDA_R_4F_E2M1;
}

template <typename T>
static inline double default_rtol()
{
    if constexpr (std::is_same_v<T, double>)
    {
        return 1e-12;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return 5e-5;
    }
    else if constexpr (std::is_same_v<T, __half>)
    {
        return 3e-3;
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return 2e-2;
    }
    else if constexpr (std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>)
    {
        return 6.5e-1;
    }
    else if constexpr (std::is_same_v<T, __nv_fp4_e2m1>)
    {
        return 7e-1;
    }
    else
    {
        return 1e-5;
    }
}

template <typename T>
static inline double default_atol()
{
    if constexpr (std::is_same_v<T, double>)
    {
        return 1e-12;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return 1e-6;
    }
    else if constexpr (std::is_same_v<T, __half>)
    {
        return 5e-3;
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return 8e-3;
    }
    else if constexpr (std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>)
    {
        return 5e-2;
    }
    else if constexpr (std::is_same_v<T, __nv_fp4_e2m1>)
    {
        return 5e-1;
    }
    else
    {
        return 1e-6;
    }
}

template <typename T>
static inline double to_double(T val)
{
    if constexpr (std::is_same_v<T, __half>)
    {
        return static_cast<double>(__half2float(val));
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return static_cast<double>(__bfloat162float(val));
    }
    else
    {
        return static_cast<double>(val);
    }
}

template <typename T>
static inline double matrix_value_to_double(const T* values, int64_t idx)
{
    if constexpr (std::is_same_v<T, __nv_fp4_e2m1>)
    {
        return static_cast<double>(fp4x2::read_element(values, idx));
    }
    else
    {
        return to_double(values[idx]);
    }
}

struct AllcloseIdentityProlog
{
    template <typename T>
    double operator()(const T* values, int64_t idx, int64_t /*row*/, int64_t /*col*/) const
    {
        return matrix_value_to_double(values, idx);
    }
};

template <typename T, typename ResultProlog, typename ReferenceProlog>
static bool allclose_host(
    const char* name,
    const T* result,
    int64_t result_lld,
    const T* reference,
    int64_t reference_lld,
    int64_t rows,
    int64_t cols,
    double rtol,
    double atol,
    ResultProlog result_prolog,
    ReferenceProlog reference_prolog)
{
    double max_abs = 0.0;
    double max_rel = 0.0;
    int64_t bad_row = -1;
    int64_t bad_col = -1;
    double bad_result = 0.0;
    double bad_reference = 0.0;

    for (int64_t col = 0; col < cols; ++col)
    {
        for (int64_t row = 0; row < rows; ++row)
        {
            const double r = result_prolog(result, row + col * result_lld, row, col);
            const double ref = reference_prolog(reference, row + col * reference_lld, row, col);
            const double diff = std::abs(r - ref);
            const double allowed = atol + rtol * std::abs(ref);
            const double rel = diff / std::max(std::abs(ref), std::numeric_limits<double>::min());
            max_abs = std::max(max_abs, diff);
            max_rel = std::max(max_rel, rel);
            if ((std::isnan(r) || std::isnan(ref) || diff > allowed) && bad_row < 0)
            {
                bad_row = row;
                bad_col = col;
                bad_result = r;
                bad_reference = ref;
            }
        }
    }

    if (bad_row >= 0)
    {
        fprintf(
            stderr,
            "Verification %s: FAILED at (%ld,%ld), actual=%0.17g expected=%0.17g max_abs=%E max_rel=%E "
            "rtol=%E atol=%E\n",
            name,
            static_cast<long>(bad_row),
            static_cast<long>(bad_col),
            bad_result,
            bad_reference,
            max_abs,
            max_rel,
            rtol,
            atol);
        return false;
    }

    printf("Verification %s: PASSED max_abs=%E max_rel=%E\n", name, max_abs, max_rel);
    return true;
}

template <typename T>
static void gather_matrix(
    cublasMpHandle_t handle,
    ncclComm_t comm,
    cudaStream_t stream,
    int64_t m,
    int64_t n,
    T* src,
    int64_t ia,
    int64_t ja,
    cublasMpMatrixDescriptor_t src_desc,
    cublasMpGrid_t grid,
    int nprow,
    int npcol,
    int myprow,
    int mypcol,
    T** dst,
    int64_t* dst_lld)
{
    constexpr int rsrc = 0;
    constexpr int csrc = 0;
    const int64_t loc_rows = cublasMpNumroc(m, m, myprow, rsrc, nprow);
    const int64_t loc_cols = cublasMpNumroc(n, n, mypcol, csrc, npcol);
    *dst_lld = std::max<int64_t>(1, loc_rows);
    if constexpr (std::is_same_v<T, __nv_fp4_e2m1>)
    {
        *dst_lld = roundup(std::max<int64_t>(2, *dst_lld), 2);
    }
    const int64_t alloc_cols = std::max<int64_t>(1, loc_cols);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(dst), (*dst_lld) * alloc_cols * sizeof(T)));

    cublasMpMatrixDescriptor_t dst_desc = nullptr;
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(m, n, m, n, rsrc, csrc, *dst_lld, CudaTypeTraits<T>::typeEnum, grid, &dst_desc));

    size_t workspace_device = 0;
    size_t workspace_host = 0;
    CUBLASMP_CHECK(cublasMpGemr2D_bufferSize(
        handle, m, n, src, ia, ja, src_desc, *dst, 1, 1, dst_desc, &workspace_device, &workspace_host, comm));

    void* d_work = nullptr;
    if (workspace_device > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_work, workspace_device));
    }
    std::vector<int8_t> h_work(workspace_host);

    CUBLASMP_CHECK(cublasMpGemr2D(
        handle,
        m,
        n,
        src,
        ia,
        ja,
        src_desc,
        *dst,
        1,
        1,
        dst_desc,
        d_work,
        workspace_device,
        h_work.data(),
        workspace_host,
        comm));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (d_work)
    {
        CUDA_CHECK(cudaFree(d_work));
    }

    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(dst_desc));
}

template <typename T>
static bool allclose_host(
    const char* name,
    const T* result,
    int64_t result_lld,
    const T* reference,
    int64_t reference_lld,
    int64_t rows,
    int64_t cols,
    double rtol,
    double atol)
{
    return allclose_host(
        name,
        result,
        result_lld,
        reference,
        reference_lld,
        rows,
        cols,
        rtol,
        atol,
        AllcloseIdentityProlog {},
        AllcloseIdentityProlog {});
}

template <typename T, typename ResultProlog, typename ReferenceProlog>
static bool allclose_device(
    const char* name,
    const T* result,
    int64_t result_lld,
    const T* reference,
    int64_t reference_lld,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream,
    double rtol,
    double atol,
    ResultProlog result_prolog,
    ReferenceProlog reference_prolog)
{
    std::vector<T> h_result(result_lld * cols);
    std::vector<T> h_reference(reference_lld * cols);
    CUDA_CHECK(cudaMemcpyAsync(h_result.data(), result, h_result.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(h_reference.data(), reference, h_reference.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return allclose_host(
        name,
        h_result.data(),
        result_lld,
        h_reference.data(),
        reference_lld,
        rows,
        cols,
        rtol,
        atol,
        result_prolog,
        reference_prolog);
}

template <typename T>
static bool allclose_device(
    const char* name,
    const T* result,
    int64_t result_lld,
    const T* reference,
    int64_t reference_lld,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream,
    double rtol = default_rtol<T>(),
    double atol = default_atol<T>())
{
    return allclose_device(
        name,
        result,
        result_lld,
        reference,
        reference_lld,
        rows,
        cols,
        stream,
        rtol,
        atol,
        AllcloseIdentityProlog {},
        AllcloseIdentityProlog {});
}
