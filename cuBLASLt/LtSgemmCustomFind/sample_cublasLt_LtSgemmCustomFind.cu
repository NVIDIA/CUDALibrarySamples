/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublasLt.h>

#include "sample_cublasLt_LtSgemmCustomFind.h"
#include "helpers.h"

/* Structure to store information about different run trials */
typedef struct {
    cublasLtMatmulAlgo_t algo;
    cublasStatus_t status;
    float time;
    size_t workspaceSize;  // actual memory workspace needed
    cublasMath_t mathMode;
    cublasLtReductionScheme_t reductionScheme;
    int customOption;
    float wavesCount;
} customMatmulPerf_t;

/* CAUTION : must match cublasLtMatmulTile_t */
const char * const matmulTileName[] = {
    "UNDEF",
    "8x8",
    "8x16",
    "16x8"   ,
    "8x32"   ,
    "16x16"  ,
    "32x8"   ,
    "8x64"   ,
    "16x32"  ,
    "32x16"  ,
    "64x8"   ,
    "32x32"  ,
    "32x64"  ,
    "64x32"  ,
    "32x128" ,
    "64x64"  ,
    "128x32" ,
    "64x128" ,
    "128x64" ,
    "64x256" ,
    "128x128",
    "256x64" ,
    "64x512" ,
    "128x256",
    "256x128",
    "512x64" ,
    "64x96",
    "96x64",
    "96x128",
    "128x160",
    "160x128",
    "192x128",
    "128x192",
    "128x96",
    "32x256",
    "256x32",
    "8x128",
    "8x192",
    "8x256",
    "8x320",
    "8x384",
    "8x448",
    "8x512",
    "8x576",
    "8x640",
    "8x704",
    "8x768",
    "16x64",
    "16x128",
    "16x192",
    "16x256",
    "16x320",
    "16x384",
    "16x448",
    "16x512",
    "16x576",
    "16x640",
    "16x704",
    "16x768",
    "24x64",
    "24x128",
    "24x192",
    "24x256",
    "24x320",
    "24x384",
    "24x448",
    "24x512",
    "24x576",
    "24x640",
    "24x704",
    "24x768",
    "32x192",
    "32x320",
    "32x384",
    "32x448",
    "32x512",
    "32x576",
    "32x640",
    "32x704",
    "32x768",
    "40x64",
    "40x128",
    "40x192",
    "40x256",
    "40x320",
    "40x384",
    "40x448",
    "40x512",
    "40x576",
    "40x640",
    "40x704",
    "40x768",
    "48x64",
    "48x128",
    "48x192",
    "48x256",
    "48x320",
    "48x384",
    "48x448",
    "48x512",
    "48x576",
    "48x640",
    "48x704",
    "48x768",
    "56x64",
    "56x128",
    "56x192",
    "56x256",
    "56x320",
    "56x384",
    "56x448",
    "56x512",
    "56x576",
    "56x640",
    "56x704",
    "56x768",
    "64x192",
    "64x320",
    "64x384",
    "64x448",
    "64x576",
    "64x640",
    "64x704",
    "64x768",
    "72x64",
    "72x128",
    "72x192",
    "72x256",
    "72x320",
    "72x384",
    "72x448",
    "72x512",
    "72x576",
    "72x640",
    "80x64",
    "80x128",
    "80x192",
    "80x256",
    "80x320",
    "80x384",
    "80x448",
    "80x512",
    "80x576",
    "88x64",
    "88x128",
    "88x192",
    "88x256",
    "88x320",
    "88x384",
    "88x448",
    "88x512",
    "96x192",
    "96x256",
    "96x320",
    "96x384",
    "96x448",
    "96x512",
    "104x64",
    "104x128",
    "104x192",
    "104x256",
    "104x320",
    "104x384",
    "104x448",
    "112x64",
    "112x128",
    "112x192",
    "112x256",
    "112x320",
    "112x384",
    "120x64",
    "120x128",
    "120x192",
    "120x256",
    "120x320",
    "120x384",
    "128x320",
    "128x384",
    "136x64",
    "136x128",
    "136x192",
    "136x256",
    "136x320",
    "144x64",
    "144x128",
    "144x192",
    "144x256",
    "144x320",
    "152x64",
    "152x128",
    "152x192",
    "152x256",
    "152x320",
    "160x64",
    "160x192",
    "160x256",
    "168x64",
    "168x128",
    "168x192",
    "168x256",
    "176x64",
    "176x128",
    "176x192",
    "176x256",
    "184x64",
    "184x128",
    "184x192",
    "184x256",
    "192x64",
    "192x192",
    "192x256",
    "200x64",
    "200x128",
    "200x192",
    "208x64",
    "208x128",
    "208x192",
    "216x64",
    "216x128",
    "216x192",
    "224x64",
    "224x128",
    "224x192",
    "232x64",
    "232x128",
    "232x192",
    "240x64",
    "240x128",
    "240x192",
    "248x64",
    "248x128",
    "248x192",
    "256x192",
    "264x64",
    "264x128",
    "272x64",
    "272x128",
    "280x64",
    "280x128",
    "288x64",
    "288x128",
    "296x64",
    "296x128",
    "304x64",
    "304x128",
    "312x64",
    "312x128",
    "320x64",
    "320x128",
    "328x64",
    "328x128",
    "336x64",
    "336x128",
    "344x64",
    "344x128",
    "352x64",
    "352x128",
    "360x64",
    "360x128",
    "368x64",
    "368x128",
    "376x64",
    "376x128",
    "384x64",
    "384x128",
    "392x64",
    "400x64",
    "408x64",
    "416x64",
    "424x64",
    "432x64",
    "440x64",
    "448x64",
    "456x64",
    "464x64",
    "472x64",
    "480x64",
    "488x64",
    "496x64",
    "504x64",
    "520x64",
    "528x64",
    "536x64",
    "544x64",
    "552x64",
    "560x64",
    "568x64",
    "576x64",
    "584x64",
    "592x64",
    "600x64",
    "608x64",
    "616x64",
    "624x64",
    "632x64",
    "640x64",
    "648x64",
    "656x64",
    "664x64",
    "672x64",
    "680x64",
    "688x64",
    "696x64",
    "704x64",
    "712x64",
    "720x64",
    "728x64",
    "736x64",
    "744x64",
    "752x64",
    "760x64",
    "768x64",
    "64x16",
    "64x24",
    "64x40",
    "64x48",
    "64x56",
    "64x72",
    "64x80",
    "64x88",
    "64x104",
    "64x112",
    "64x120",
    "64x136",
    "64x144",
    "64x152",
    "64x160",
    "64x168",
    "64x176",
    "64x184",
    "64x200",
    "64x208",
    "64x216",
    "64x224",
    "64x232",
    "64x240",
    "64x248",
    "64x264",
    "64x272",
    "64x280",
    "64x288",
    "64x296",
    "64x304",
    "64x312",
    "64x328",
    "64x336",
    "64x344",
    "64x352",
    "64x360",
    "64x368",
    "64x376",
    "64x392",
    "64x400",
    "64x408",
    "64x416",
    "64x424",
    "64x432",
    "64x440",
    "64x456",
    "64x464",
    "64x472",
    "64x480",
    "64x488",
    "64x496",
    "64x504",
    "64x520",
    "64x528",
    "64x536",
    "64x544",
    "64x552",
    "64x560",
    "64x568",
    "64x584",
    "64x592",
    "64x600",
    "64x608",
    "64x616",
    "64x624",
    "64x632",
    "64x648",
    "64x656",
    "64x664",
    "64x672",
    "64x680",
    "64x688",
    "64x696",
    "64x712",
    "64x720",
    "64x728",
    "64x736",
    "64x744",
    "64x752",
    "64x760",
    "128x8",
    "128x16",
    "128x24",
    "128x40",
    "128x48",
    "128x56",
    "128x72",
    "128x80",
    "128x88",
    "128x104",
    "128x112",
    "128x120",
    "128x136",
    "128x144",
    "128x152",
    "128x168",
    "128x176",
    "128x184",
    "128x200",
    "128x208",
    "128x216",
    "128x224",
    "128x232",
    "128x240",
    "128x248",
    "128x264",
    "128x272",
    "128x280",
    "128x288",
    "128x296",
    "128x304",
    "128x312",
    "128x328",
    "128x336",
    "128x344",
    "128x352",
    "128x360",
    "128x368",
    "128x376",
    "128x392",
    "128x400",
    "128x408",
    "128x416",
    "128x424",
    "128x432",
    "128x440",
    "128x448",
    "128x456",
    "128x464",
    "128x472",
    "128x480",
    "128x488",
    "128x496",
    "128x504",
    "128x512",
    "192x8",
    "192x16",
    "192x24",
    "192x32",
    "192x40",
    "192x48",
    "192x56",
    "192x72",
    "192x80",
    "192x88",
    "192x96",
    "192x104",
    "192x112",
    "192x120",
    "192x136",
    "192x144",
    "192x152",
    "192x160",
    "192x168",
    "192x176",
    "192x184",
    "192x200",
    "192x208",
    "192x216",
    "192x224",
    "192x232",
    "192x240",
    "192x248",
    "192x264",
    "192x272",
    "192x280",
    "192x288",
    "192x296",
    "192x304",
    "192x312",
    "192x320",
    "192x328",
    "192x336",
    "256x8",
    "256x16",
    "256x24",
    "256x40",
    "256x48",
    "256x56",
    "256x72",
    "256x80",
    "256x88",
    "256x96",
    "256x104",
    "256x112",
    "256x120",
    "256x136",
    "256x144",
    "256x152",
    "256x160",
    "256x168",
    "256x176",
    "256x184",
    "256x200",
    "256x208",
    "256x216",
    "256x224",
    "256x232",
    "256x240",
    "256x248",
    "256x256",
    "320x8",
    "320x16",
    "320x24",
    "320x32",
    "320x40",
    "320x48",
    "320x56",
    "320x72",
    "320x80",
    "320x88",
    "320x96",
    "320x104",
    "320x112",
    "320x120",
    "320x136",
    "320x144",
    "320x152",
    "320x160",
    "320x168",
    "320x176",
    "320x184",
    "320x192",
    "320x200",
    "384x8",
    "384x16",
    "384x24",
    "384x32",
    "384x40",
    "384x48",
    "384x56",
    "384x72",
    "384x80",
    "384x88",
    "384x96",
    "384x104",
    "384x112",
    "384x120",
    "384x136",
    "384x144",
    "384x152",
    "384x160",
    "384x168",
    "448x8",
    "448x16",
    "448x24",
    "448x32",
    "448x40",
    "448x48",
    "448x56",
    "448x72",
    "448x80",
    "448x88",
    "448x96",
    "448x104",
    "448x112",
    "448x120",
    "448x128",
    "448x136",
    "448x144",
    "512x8",
    "512x16",
    "512x24",
    "512x32",
    "512x40",
    "512x48",
    "512x56",
    "512x72",
    "512x80",
    "512x88",
    "512x96",
    "512x104",
    "512x112",
    "512x120",
    "512x128",
    "576x8",
    "576x16",
    "576x24",
    "576x32",
    "576x40",
    "576x48",
    "576x56",
    "576x72",
    "576x80",
    "576x88",
    "576x96",
    "576x104",
    "576x112",
    "640x8",
    "640x16",
    "640x24",
    "640x32",
    "640x40",
    "640x48",
    "640x56",
    "640x72",
    "640x80",
    "640x88",
    "640x96",
    "704x8",
    "704x16",
    "704x24",
    "704x32",
    "704x40",
    "704x48",
    "704x56",
    "704x72",
    "704x80",
    "704x88",
    "768x8",
    "768x16",
    "768x24",
    "768x32",
    "768x40",
    "768x48",
    "768x56",
    "768x72",
    "768x80",
    "256x512",
    "256x1024",
    "512x512",
    "512x1024",
};

// Utility function to print customMatmulPerf_t structure
static void printPerfStructure(const customMatmulPerf_t &perf) {
    int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme, stages;

    const cublasLtMatmulAlgo_t *matmulAlgo = &perf.algo;
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), NULL);

    assert((unsigned)tile < sizeof(matmulTileName) / sizeof(matmulTileName[0]));

    printf("algo={ Id=%d, tileIdx=%d (%s) splitK=%d reduc=%d swizzle=%d custom=%d stages=%d} status %d "
        "time %f workspace=%d mathMode=%d waves=%f\n",
        algoId, tile, matmulTileName[tile],
        numSplitsK, reductionScheme,
        swizzle, customOption, stages,
        perf.status,
        perf.time,
        (int)perf.workspaceSize,
        (int)perf.mathMode,
        perf.wavesCount);
}

static inline bool time_compare(const customMatmulPerf_t &perf_a, const customMatmulPerf_t &perf_b) {
    return ((perf_a.status == CUBLAS_STATUS_SUCCESS) && (perf_a.time < perf_b.time));
}

static cublasStatus_t customMatmulRun(cublasLtHandle_t ltHandle,  // to get the capabilities (required a GPU)
                 cublasLtMatmulDesc_t operationDesc,
                 const void *alpha, /* host or device pointer */
                 const void *A,
                 cublasLtMatrixLayout_t Adesc,
                 const void *B,
                 cublasLtMatrixLayout_t Bdesc,
                 const void *beta, /* host or device pointer */
                 const void *C,
                 cublasLtMatrixLayout_t Cdesc,
                 void *D,
                 cublasLtMatrixLayout_t Ddesc,
                 const cublasLtMatmulAlgo_t &algo,
                 int kernelRepeats,
                 void *workSpace,
                 size_t workSpaceSizeInBytes,
                 customMatmulPerf_t &perfResults,
                 cudaStream_t stream,
                 cudaEvent_t &startEvent,
                 cudaEvent_t &stopEvent) {
    cublasLtMatmulHeuristicResult_t heurResult;
    /* Looping over the Algo */
    int repeats = kernelRepeats;

    cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck( ltHandle,
                                                         operationDesc,
                                                         Adesc,
                                                         Bdesc,
                                                         Cdesc,
                                                         Ddesc,
                                                         &algo,
                                                         &heurResult);

    if (algoStatus == CUBLAS_STATUS_SUCCESS) {
        if (heurResult.workspaceSize <= workSpaceSizeInBytes) {
            cudaError_t err, err1, err2, err3;
            err  = cudaEventRecord(startEvent, stream);
            for (int loop = 0; loop < repeats; loop++) {
                cublasStatus_t oneRunStatus = cublasLtMatmul( ltHandle,
                                                              operationDesc,
                                                              alpha,
                                                              A, Adesc,
                                                              B, Bdesc,
                                                              beta,
                                                              C, Cdesc,
                                                              D, Ddesc,
                                                              &algo,
                                                              workSpace,
                                                              workSpaceSizeInBytes,
                                                              stream);
                if (oneRunStatus != CUBLAS_STATUS_SUCCESS) {
                    algoStatus = oneRunStatus;
                    break;
                }
            }
            err1 = cudaEventRecord(stopEvent, stream);
            err2 = cudaEventSynchronize(stopEvent);
            float time;
            err3 = cudaEventElapsedTime(&time, startEvent, stopEvent);
            if ((err != cudaSuccess) || (err1 != cudaSuccess) || (err2 != cudaSuccess) || (err3 != cudaSuccess)) {
                algoStatus = CUBLAS_STATUS_INTERNAL_ERROR;
            }
            // For the moment only add successful findings
            if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                perfResults.algo = algo;
                perfResults.time = time;
                perfResults.workspaceSize = heurResult.workspaceSize;
                perfResults.wavesCount = heurResult.wavesCount;
            }
        }
        else {
            algoStatus = CUBLAS_STATUS_NOT_SUPPORTED; //Not enough workspace
        }
    }

    return algoStatus;
}

/// Sample wrapper running through multiple algo and config attributes combination for single precision gemm using cublasLt low-level API
void LtSgemmCustomFind(cublasLtHandle_t ltHandle,
                      cublasOperation_t transa,
                      cublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const float *alpha, /* host pointer */
                      const float *A,
                      int lda,
                      const float *B,
                      int ldb,
                      const float *beta, /* host pointer */
                      float *C,
                      int ldc,
                      void *workSpace,
                      size_t workSpaceSize) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;
    cudaEvent_t startEvent = NULL, stopEvent = NULL;
    cudaStream_t stream = NULL;
    // SplitK values that we are going to try when SplitK is supported for a given algo
    const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};
     // Let try a fixed number of combinations
    #define ALGO_COMBINATIONS 100
    int AlgoCombinations = ALGO_COMBINATIONS;
    int AlgoCount = 0;
    int kernelRepeats = 10; //number of time the CUDA kernels will be run back to back
    customMatmulPerf_t perfResults[ALGO_COMBINATIONS];
    int nbAlgoIds = 0;
    #define ALGO_IDS 16
    int algoIdA[ALGO_IDS];
    cudaDataType_t scaleType = CUDA_R_32F, Atype = CUDA_R_32F, Btype = CUDA_R_32F, Ctype = CUDA_R_32F;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, Atype, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, Btype, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc));

    checkCublasStatus(cublasLtMatmulAlgoGetIds(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, ALGO_IDS, algoIdA, &nbAlgoIds));

    // Create CUDA event to time the execution time of each algo
    checkCudaStatus(cudaEventCreate(&startEvent, cudaEventBlockingSync));
    checkCudaStatus(cudaEventCreate(&stopEvent, cudaEventBlockingSync));

    int device;
    checkCudaStatus(cudaGetDevice(&device));
    int clusterLaunchSupported;
    checkCudaStatus(cudaDeviceGetAttribute(&clusterLaunchSupported, cudaDevAttrClusterLaunch, device));
    uint16_t cgaLastShape = clusterLaunchSupported ? CUBLASLT_CLUSTER_SHAPE_END : CUBLASLT_CLUSTER_SHAPE_AUTO + 1;

    // Loop over the Algo IDs
    for (int idx = 0; (idx < nbAlgoIds) && (AlgoCount < AlgoCombinations); idx++) {
        cublasLtMatmulAlgo_t algo;
        size_t sizeWritten = 0;
        // Initialize algo structure with given Algo ID
        status = cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIdA[idx], &algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        // Query the tiles enums supported by that algo
        checkCublasStatus(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0, &sizeWritten));
        int nbTiles = int(sizeWritten/sizeof(int));
        int *tileA = new int[nbTiles == 0 ? 1 : nbTiles];
        if (nbTiles == 0) {
            tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles = 1;
        }

        checkCublasStatus(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, NULL, 0, &sizeWritten));
        int nbStages = int(sizeWritten/sizeof(int));
        std::vector<int> stagesA(nbStages == 0 ? 1 : nbStages);
        if (nbStages == 0) {
            stagesA[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED;
            nbStages = 1;
        } else {
            checkCublasStatus(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stagesA.data(), sizeof(int)*nbStages, &sizeWritten));
        }

        int splitkSupport, redMask, swizzlingMax, customOptionMax;
        // Retrieve Algo Capabilities attributes to be able to setup loop over the different combinations
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA, sizeof(int)*nbTiles, &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten);

        // Loop over the different tiles
        for (int tileIdx = 0; tileIdx < nbTiles; tileIdx++) {
            checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx], sizeof(tileA[tileIdx])));
            // Loop over different stages count
            for (int stagesIdx = 0; stagesIdx < nbStages; stagesIdx++) {
                checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesA[stagesIdx], sizeof(stagesA[stagesIdx])));
                // Loop over different cluster configurations on devices that support it
                for (uint16_t cgaIdx = 0; cgaIdx < cgaLastShape; cgaIdx++) {
                    checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cgaIdx, sizeof(cgaIdx)));
                    // Loop over the different custom option if any
                    for (int customOption = 0; customOption <= customOptionMax; customOption++) {
                        checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption)));
                        // loop over the CTAs swizzling support
                        for (int k = 0; k <= swizzlingMax; k++) {
                            int splitK_trial = 0;
                            if (splitkSupport) {
                                splitK_trial += sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
                            }
                            // Loop over the splitK value over a fixed sequence splitKSequenceA in addtion to the case where splitK is not enabled
                            for (int l = 0; (l < (1 + splitK_trial)) && (AlgoCount < AlgoCombinations); l++) {
                                int splitK_val = 0;
                                int redScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
                                checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val)));
                                checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k)));
                                checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(int)));

                                if (l > 0) { // Split-K case
                                    splitK_val = splitKSequenceA[l - 1];
                                    checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKSequenceA[l - 1], sizeof(splitKSequenceA[l - 1])));
                                    // Going over all the reduction schemes
                                    for (redScheme = 1 ; redScheme < (int)CUBLASLT_REDUCTION_SCHEME_MASK && (AlgoCount < AlgoCombinations); redScheme = redScheme << 1) {
                                        if (redScheme & redMask) {
                                            checkCublasStatus(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(redScheme)));

                                            status = customMatmulRun( ltHandle,
                                                                    operationDesc,
                                                                    alpha, // host or device pointer
                                                                    A, Adesc,
                                                                    B, Bdesc,
                                                                    beta, // host or device pointer
                                                                    C, Cdesc,
                                                                    C, Cdesc,
                                                                    algo,
                                                                    kernelRepeats,
                                                                    workSpace,
                                                                    workSpaceSize,
                                                                    perfResults[AlgoCount],
                                                                    stream,
                                                                    startEvent, stopEvent);
                                            perfResults[AlgoCount].status = status;
                                            if (status == CUBLAS_STATUS_SUCCESS) AlgoCount++;

                                        } // end if
                                    } // end for
                                } else { // Non-splitK case
                                    // if user preference is ok with workspace
                                    if (AlgoCount < AlgoCombinations) {
                                        status = customMatmulRun( ltHandle,
                                                                operationDesc,
                                                                alpha, // host or device pointer
                                                                A, Adesc,
                                                                B, Bdesc,
                                                                beta, // host or device pointer
                                                                C, Cdesc,
                                                                C, Cdesc,
                                                                algo,
                                                                kernelRepeats,
                                                                workSpace,
                                                                workSpaceSize,
                                                                perfResults[AlgoCount],
                                                                stream,
                                                                startEvent, stopEvent);
                                        perfResults[AlgoCount].status = status;
                                        if (status == CUBLAS_STATUS_SUCCESS) AlgoCount++;
                                    }
                                }
                            }  // end l
                        }  // end k
                    } //end customOption
                } // end cgaIdx
            } // end stagesIdx
        } // end tileIdx
        delete [] tileA;
    } // end idx

    // Sort the results per run duration
    std::sort(perfResults, perfResults + AlgoCount, time_compare);
    // Print timing and perf details
    for (int i = 0; i < AlgoCount; i++) {
        printf( "result %03d : ", i);
        printPerfStructure(perfResults[i]);
    }

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    if (startEvent) checkCudaStatus(cudaEventDestroy(startEvent));
    if (stopEvent) checkCudaStatus(cudaEventDestroy(stopEvent));
}
