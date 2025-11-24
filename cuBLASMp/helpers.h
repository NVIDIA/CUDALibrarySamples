/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <mpi.h>
#include <string.h>

#include <cctype>
#include <cstdint>
#include <stdexcept>
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

#define NVSHMEM_CHECK(call)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        int status = call;                                                                                             \
        if (status != 0)                                                                                               \
        {                                                                                                              \
            fprintf(stderr, "NVSHMEM error at %s:%d : %d\n", __FILE__, __LINE__, status);                              \
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

size_t get_scaling_tensor_size(int64_t m, int64_t n, cublasMpMatmulMatrixScale_t scale_mode)
{
    switch (scale_mode)
    {
        case CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32: return sizeof(float);
        case CUBLASMP_MATMUL_MATRIX_SCALE_OUTER_VEC_FP32: return n * sizeof(float);
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3:
            return roundup(m, 4 * 16) / 16 * roundup(n, 128) * sizeof(__nv_fp8_e4m3);
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC32_UE8M0:
            return roundup(m, 4 * 32) / 32 * roundup(n, 128) * sizeof(__nv_fp8_e8m0);
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC128_FP32: return roundup((m + 127) / 128, 4) * n * sizeof(float);
        case CUBLASMP_MATMUL_MATRIX_SCALE_BLK128x128_FP32:
            return (roundup((m + 127) / 128, 4) * ((n + 127) / 128)) * sizeof(float);
        default: return 0;
    }
}

void* allocate_and_init_scaling_factors(int64_t m, int64_t n, cublasMpMatmulMatrixScale_t scale_mode)
{
    size_t scale_size = get_scaling_tensor_size(m, n, scale_mode);
    if (scale_size == 0) return nullptr;

    int rank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    std::srand(rank);

    void* d_scale = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scale, scale_size));

    switch (scale_mode)
    {
        case CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32:
        {
            generate_values(rank, reinterpret_cast<float*>(d_scale), 1, true, 1, 10);
            break;
        }

        case CUBLASMP_MATMUL_MATRIX_SCALE_OUTER_VEC_FP32:
        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC128_FP32:
        case CUBLASMP_MATMUL_MATRIX_SCALE_BLK128x128_FP32:
        {
            generate_values(rank, reinterpret_cast<float*>(d_scale), scale_size / sizeof(float), true, 1, 10);
            break;
        }

        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3:
        {
            generate_values(
                rank, reinterpret_cast<__nv_fp8_e4m3*>(d_scale), scale_size / sizeof(__nv_fp8_e4m3), true, 1, 10);
            break;
        }

        case CUBLASMP_MATMUL_MATRIX_SCALE_VEC32_UE8M0:
        {
            generate_values(
                rank, reinterpret_cast<__nv_fp8_e8m0*>(d_scale), scale_size / sizeof(__nv_fp8_e8m0), true, 1, 10);
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

    // data types
    cudaDataType_t typeA = CUDA_R_64F;
    cudaDataType_t typeB = CUDA_R_64F;
    cudaDataType_t typeC = CUDA_R_64F;
    cudaDataType_t typeD = CUDA_R_64F;

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;

    // others
    bool verbose = false;
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
        printf(
            "Available options:\n"
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
            "    -scaleA <string> (scalar_fp32, vec16_ue4m3, vec32_ue8m0, outer_vec_fp32, vec128_fp32, blk128x128_fp32)\n"
            "    -scaleB <string> (scalar_fp32, vec16_ue4m3, vec32_ue8m0, outer_vec_fp32, vec128_fp32, blk128x128_fp32)\n"
            "    -scaleD <string> (scalar_fp32, vec16_ue4m3, vec32_ue8m0, outer_vec_fp32, vec128_fp32, blk128x128_fp32)\n"
            "    -scaleDOut <string> (scalar_fp32, vec16_ue4m3, vec32_ue8m0, outer_vec_fp32, vec128_fp32, blk128x128_fp32)\n"
            "    -p <int>\n"
            "    -q <int>\n"
            "    -gridLayout <char> (c, r)\n"
            "    -emulationStrategy <string> (default, performant, eager)\n"
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
            "p=%d q=%d gridLayout=%c "
            "emulationStrategy=%s "
            "cycles=%d warmup=%d "
            "verbose=%s\n",
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
            scaleA,
            scaleB,
            scaleD,
            scaleDOut,
            p,
            q,
            grid_layout,
            emulationStrategy,
            cycles,
            warmup,
            verbose ? "true" : "false");
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
    }
};

static inline int getLocalDevice()
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

static bool deviceSupportsFp8()
{
    int local_device = getLocalDevice();
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, local_device));
    return prop.major >= 9 || (prop.major == 8 && prop.minor == 9);
}

static bool deviceSupportsFp4()
{
    int local_device = getLocalDevice();
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, local_device));
    return prop.major >= 10;
}