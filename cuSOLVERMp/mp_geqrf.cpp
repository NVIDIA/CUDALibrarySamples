/*
 * Copyright 2023 NVIDIA Corporation.  All rights reserved.
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


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cmath>
#include <vector>

#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

/* A is 1D laplacian, return A[N:-1:1, :] */
static void gen_matrix(
        int64_t M,
        int64_t N,
        double *A,
        int64_t lda)
{
    /* set A[0:N, 0:N] = 0 */
    for(auto J=0; J < N; J++) {
        for(auto I=0; I < M; I++) {
            A[ I + J * lda ] = J % 6 + I % 3 + 2*I/7 + 3*J/6;
        }
    }

    /* set entries */
    for(int J = 0 ; J < std::min(M,N); J++ ){
        /* main diagonal */
        A[ ((M-1)-J) + J * lda ] = 2.0;
        A[ J+J*lda] = 2.0;

        /* upper diagonal */
        if ( J > 0 ){
            A[ ((M-1)-(J-1)) + J * lda ] = -1.0;
        }
        /* lower diagonal */
        if ( J < (N-1) ){
            A[ ((M-1)-(J+1)) + J * lda ] = -1.0;
        }
    }
}

/* Print matrix */
static void print_host_matrix (
        int64_t M,
        int64_t N,
        double *A,
        int64_t lda,
        const char *msg)
{
    if (M * N > 2000) return;
    printf("print_host_matrix : %s\n", msg );

    for(auto i=0; i < M; i++) {
        for(auto j=0; j < N; j++) {
            printf("%.2lf  ", A[i + j * lda] );
        }
        printf("\n");
    }
}

int main (int argc, char *argv[])
{
    /* Initialize MPI  library */
    MPI_Init(NULL, NULL);

    /* Define dimensions, block sizes and offsets of A and B matrices */
    const int64_t M    = 10;
    const int64_t N    = 10;

    /* Tile sizes */
    const int64_t MA = 24;
    const int64_t NA = 24;

    /* Offsets of A and B matrices (base-1) */
    const int64_t IA = 4;
    const int64_t JA = 4;

    /* Define grid of processors */
    const int numRowDevices = 2;
    const int numColDevices = 1;

    /* Current implementation only allows RSRC,CSRC=(0,0) */
    const uint32_t RSRCA = 0;
    const uint32_t CSRCA = 0;
    assert(RSRCA == 0 && CSRCA == 0); // only RSRCA==0 and CSRC==0 are supported

    /* Get rank id and rank size of the com. */
    int rankSize, rankId;
    MPI_Comm_size( MPI_COMM_WORLD, &rankSize );
    MPI_Comm_rank( MPI_COMM_WORLD, &rankId   );

    /* Library handles */
    cusolverMpHandle_t cusolverMpHandle = nullptr;
    cal_comm_t cal_comm = nullptr;

    /* Error codes */
    cusolverStatus_t cusolverStat = CUSOLVER_STATUS_SUCCESS;
    calError_t  calStat = CAL_OK;
    cudaError_t cudaStat = cudaSuccess;

    /* User defined stream */
    cudaStream_t localStream = nullptr;

    /* 
     * localDeviceId is the deviceId from rank's point of view. This is
     * system-dependent. For example, setting one device per process,
     * Summit always sees the local device as device 0.
     */
    const int localDeviceId = getLocalRank();

    cudaStat = cudaSetDevice(localDeviceId);
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaFree(0);
    assert(cudaStat == cudaSuccess);

    /* Create communicator */
    cal_comm_create_params_t params;
    params.allgather = allgather;
    params.req_test = request_test;
    params.req_free = request_free;
    params.data = reinterpret_cast<void*>(MPI_COMM_WORLD);
    params.rank = rankId;
    params.nranks = rankSize;
    params.local_device = localDeviceId;

    calStat = cal_comm_create(params, &cal_comm);
    assert(calStat == CAL_OK); 

    /* Create local stream */
    cudaStat = cudaStreamCreate( &localStream );
    assert( cudaStat == cudaSuccess );

    /* Initialize cusolverMp library handle */
    cusolverStat = cusolverMpCreate(
            &cusolverMpHandle,
            localDeviceId,
            localStream);
    assert( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    /* cudaLigMg grids */
    cudaLibMpGrid_t gridA    = nullptr; 

    /* cudaLib matrix descriptors */
    cudaLibMpMatrixDesc_t descrA    = nullptr;

    /* Distributed matrices */
    void *d_A    = nullptr;
    void *d_tau  = nullptr;

    /* Distributed device workspace */
    void *d_work_geqrf = nullptr;

    /* Distributed host workspace */
    void *h_work_geqrf = nullptr;

    /* size of workspace on device */
    size_t workspaceInBytesOnDevice_geqrf = 0;

    /* size of workspace on host */
    size_t workspaceInBytesOnHost_geqrf = 0;

    /* error codes from cusolverMp (device) */
    int* d_info_geqrf = nullptr;

    /* error codes from cusolverMp (host) */
    int  h_info_geqrf = 0;

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    /* Single process per device */
    assert ( (numRowDevices * numColDevices) <= rankSize );

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    const int64_t lda   = (IA -1) + M;
    const int64_t colsA = (JA -1) + N;

    double *h_A = nullptr;
    double *h_QR = nullptr;
    double *h_tau = nullptr;

    void *d_global_Q = nullptr;
    void *d_global_R = nullptr;
    void *d_global_tau = nullptr;

    if ( rankId == 0 )
    {
        /* allocate host workspace */
        h_A = (double *)malloc(lda * colsA * sizeof(double));
        h_QR = (double *)malloc(lda * colsA * sizeof(double));
        memset(h_A, 0, lda * colsA * sizeof(double));
        double *_A = &h_A[(IA - 1) + (JA - 1) * lda]; // first entry of A
        gen_matrix(M, N, _A, lda);
        print_host_matrix(lda, colsA, h_A, lda, "Input matrix A");

        h_tau = (double *)malloc(lda * sizeof(double));
    }

    /* =========================================== */
    /*            COMPUTE LLDA AND LLDB            */
    /* =========================================== */

    /* 
     * Compute number of tiles per rank to store local portion of A
     *
     * Current implementation has the following restrictions on the size of
     * the device buffer size:
     *  - Rows of device buffer is a multiple of MA
     *  - Cols of device buffer is a multiple of NA
     *
     * This limitation will be removed on the official release.
     */
    const int64_t LLDA = cusolverMpNUMROC(lda, MA, RSRCA, rankId % numRowDevices, numRowDevices);
    const int64_t localColsA = cusolverMpNUMROC(colsA, NA, CSRCA, rankId / numRowDevices, numColDevices);

    /* 
     * Compute number of tiles per rank to store local portion of B
     *
     * Current implementation has the following restrictions on the size of
     * the device buffer size:
     *  - Rows of device buffer is a multiple of MB
     *  - Cols of device buffer is a multiple of NB
     *
     * This limitation will be removed on the official release.
     */
    /* Allocate global d_A */
    cudaStat = cudaMalloc( (void**)&d_A, localColsA * LLDA * sizeof(double) );
    assert( cudaStat == cudaSuccess );

    /* =========================================== */
    /*          CREATE GRID DESCRIPTORS            */
    /* =========================================== */
    cusolverStat = cusolverMpCreateDeviceGrid(
            cusolverMpHandle,
            &gridA,
            cal_comm,
            numRowDevices,
            numColDevices,
            CUDALIBMP_GRID_MAPPING_COL_MAJOR);
    assert ( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    /* =========================================== */
    /*        CREATE MATRIX DESCRIPTORS            */
    /* =========================================== */
    cusolverStat = cusolverMpCreateMatrixDesc(
            &descrA,
            gridA,
            CUDA_R_64F,
            (IA-1) + M,
            (JA-1) + N,
            MA,
            NA,
            RSRCA,
            CSRCA,
            LLDA);
            
    assert ( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    /* Allocate global d_tau */
    cudaStat = cudaMalloc((void **)&d_tau, localColsA * sizeof(double));
    assert(cudaStat == cudaSuccess);

    /* =========================================== */
    /*             ALLOCATE D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMalloc( &d_info_geqrf, sizeof(int));
    assert( cudaStat == cudaSuccess  );

    /* =========================================== */
    /*                RESET D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMemset( d_info_geqrf, 0, sizeof(int));
    assert( cudaStat == cudaSuccess  );

    /* =========================================== */
    /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
    /* =========================================== */

    cusolverStat = cusolverMpGeqrf_bufferSize(
            cusolverMpHandle,
            M,
            N,
            d_A,
            IA,
            JA,
            descrA,
            CUDA_R_64F,
            &workspaceInBytesOnDevice_geqrf,
            &workspaceInBytesOnHost_geqrf);
    assert ( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    /* =========================================== */
    /*         ALLOCATE Pgeqrf WORKSPACE            */
    /* =========================================== */

    cudaStat = cudaMalloc( (void**)&d_work_geqrf, workspaceInBytesOnDevice_geqrf );
    assert( cudaStat == cudaSuccess  );

    h_work_geqrf = (void*) malloc( workspaceInBytesOnHost_geqrf );
    assert( h_work_geqrf != nullptr );


    /* =========================================== */
    /*      SCATTER MATRICES A AND B FROM MASTER   */
    /* =========================================== */
    cusolverStat = cusolverMpMatrixScatterH2D (
            cusolverMpHandle,
            lda,
            colsA,
            (void*) d_A, /* routine requires void** */
            1,
            1,
            descrA,
            0, /* root rank */
            (void*) h_A,
            lda);
    assert( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    /* sync wait for data to arrive to device */ 
    calStat = cal_stream_sync( cal_comm, localStream );
    assert( calStat == CAL_OK );


    /* =========================================== */
    /*                   CALL Pgeqrf               */
    /* =========================================== */

    cusolverStat = cusolverMpGeqrf (
            cusolverMpHandle,
            M,
            N,
            d_A,
            IA,
            JA,
            descrA,
            d_tau,
            CUDA_R_64F,
            d_work_geqrf,
            workspaceInBytesOnDevice_geqrf,
            h_work_geqrf,
            workspaceInBytesOnHost_geqrf,
            d_info_geqrf);

    /* sync after cusolverMpgeqrf */
    calStat = cal_stream_sync( cal_comm, localStream );
    assert( calStat == CAL_OK );


    /* copy d_info_geqrf to host */
    cudaStat = cudaMemcpyAsync(
            &h_info_geqrf,
            d_info_geqrf,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            localStream );
    assert( cudaStat == cudaSuccess );

    /* =========================================== */
    /*      GATHER MATRICES A AND B FROM MASTER    */
    /* =========================================== */

    /* Copy solution to h_A */
    cusolverStat = cusolverMpMatrixGatherD2H(
        cusolverMpHandle,
        lda,
        colsA,
        (void *)d_A,
        1,
        1,
        descrA,
        0, /* master rank, destination */
        (void *)h_QR,
        lda);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync wait for data to arrive to host */
    calStat = cal_stream_sync(cal_comm, localStream);
    assert(calStat == CAL_OK);

    /* =========================================== */
    /*            PRINT A ON RANK 0                */
    /* =========================================== */
    if (rankId == 0)
    {
        print_host_matrix(lda, colsA, h_QR, lda, "Output matrix QR");
    }

    // allocate global GPU arrays and copy h_A to d_global_Q, d_global_R
    if (rankId == 0)
    {
        cudaStat = cudaMalloc((void **)&d_global_Q, lda * colsA * sizeof(double));
        assert(cudaStat == cudaSuccess);
        cudaStat = cudaMalloc((void **)&d_global_R, lda * colsA * sizeof(double));
        assert(cudaStat == cudaSuccess);
        cudaStat = cudaMalloc((void **)&d_global_tau, colsA * sizeof(double));
        assert(cudaStat == cudaSuccess);

        cudaStat = cudaMemcpy(d_global_Q, h_A, sizeof(double) * lda * colsA, cudaMemcpyHostToDevice);
        assert(cudaStat == cudaSuccess);
        cudaStat = cudaMemcpy(d_global_R, d_global_Q, sizeof(double) * lda * colsA, cudaMemcpyDeviceToDevice);
        assert(cudaStat == cudaSuccess);

        // compare to 1 gpu cusolverDnGeqrf
        cusolverDnParams_t dn_geqrf_params = nullptr;
        cusolverStat = cusolverDnCreateParams(&dn_geqrf_params);
        // cusolverDnHandle_t cudenseHandle = cusolverMpHandle->cusolverDnH;

        cusolverDnHandle_t cudenseHandle;
        cusolverStat = cusolverDnCreate(&cudenseHandle);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);
        cusolverStat = cusolverDnSetStream(cudenseHandle, localStream);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        const void *d_global_Q_ptr = (double *)d_global_Q + ((IA - 1) + (JA - 1) * lda);

        cusolverStat = cusolverDnXgeqrf( // overwrites A
            cudenseHandle,
            dn_geqrf_params,
            M,
            N,
            CUDA_R_64F,
            (void *)d_global_Q_ptr, // in/out
            lda,
            CUDA_R_64F,
            (void *)d_global_tau,
            CUDA_R_64F,
            (void *)d_work_geqrf,
            workspaceInBytesOnDevice_geqrf,
            (void *)h_work_geqrf,
            workspaceInBytesOnHost_geqrf,
            d_info_geqrf);

        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cudaStat = cudaMemcpy(h_A, d_global_Q, sizeof(double) * lda * colsA, cudaMemcpyDeviceToHost);
        assert(cudaStat == cudaSuccess);

        print_host_matrix(lda, colsA, h_A, lda, "A after cusolverDnXgeqrf");

        int passed = 0;
        int failed = 0;
        int passed_abs = 0;
        int failed_abs = 0;

        for (int i = 0; i < (M + IA - 1) * (N + JA - 1); i++)
        {
            if (fabs(h_A[i] - h_QR[i]) < 0.001)
                passed++;
            else
                failed++;

            if (fabs(fabs(h_A[i]) - fabs(h_QR[i])) < 0.001)
                passed_abs++;
            else
                failed_abs++;

            h_A[i] = fabs(h_A[i] - h_QR[i]);
        }

        print_host_matrix(lda, colsA, h_A, lda, "difference");

        printf("passed_abs %d failed_abs %d passed %d failed %d\n", passed_abs, failed_abs, passed, failed);
    }

    /* =========================================== */
    /*        CLEAN UP HOST WORKSPACE ON MASTER    */ 
    /* =========================================== */
    if ( rankId == 0 )
    {
        if ( h_A ) { free( h_A ); h_A = nullptr; }
        if ( h_tau ) { free( h_tau ); h_tau = nullptr; }
    }

    /* =========================================== */
    /*           DESTROY MATRIX DESCRIPTORS        */ 
    /* =========================================== */

    cusolverStat = cusolverMpDestroyMatrixDesc( descrA );
    assert( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    /* =========================================== */
    /*             DESTROY MATRIX GRIDS            */ 
    /* =========================================== */

    cusolverStat = cusolverMpDestroyGrid( gridA );
    assert( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    /* =========================================== */
    /*          DEALLOCATE DEVICE WORKSPACE        */ 
    /* =========================================== */

    if ( d_A != nullptr ) {
        cudaStat = cudaFree( d_A );
        assert( cudaStat == cudaSuccess );
        d_A = nullptr;
    }

    if ( d_tau != nullptr ) {
        cudaStat = cudaFree( d_tau );
        assert( cudaStat == cudaSuccess );
        d_tau = nullptr;
    }

    if ( d_work_geqrf != nullptr ) {
        cudaStat = cudaFree( d_work_geqrf );
        assert( cudaStat == cudaSuccess );
        d_work_geqrf = nullptr;
    }

    if ( d_info_geqrf != nullptr ) {
        cudaStat = cudaFree( d_info_geqrf );
        assert( cudaStat == cudaSuccess );
        d_info_geqrf = nullptr;
    }

    /* =========================================== */
    /*         DEALLOCATE HOST WORKSPACE           */ 
    /* =========================================== */
    if ( h_work_geqrf ) { free( h_work_geqrf ); h_work_geqrf = nullptr; }

    /* =========================================== */
    /*                      CLEANUP                */ 
    /* =========================================== */

    /* Destroy cusolverMp handle */
    cusolverStat = cusolverMpDestroy( cusolverMpHandle );
    assert( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    /* sync before cal_comm_destroy */
    calStat = cal_comm_barrier( cal_comm, localStream );
    assert( calStat == CAL_OK );

    /* destroy CAL communicator */
    calStat = cal_comm_destroy( cal_comm );
    assert( calStat == CAL_OK );

    /* destroy user stream */
    cudaStat = cudaStreamDestroy( localStream );
    assert( cudaStat == cudaSuccess );

    /* MPI barrier before MPI_Finalize */
    MPI_Barrier( MPI_COMM_WORLD );

    /* Finalize MPI environment */
    MPI_Finalize();

    return 0;
};
