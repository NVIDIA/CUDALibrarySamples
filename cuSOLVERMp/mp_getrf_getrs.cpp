/*
 * Copyright 2021 NVIDIA Corporation.  All rights reserved.
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
#include <cmath>
#include <assert.h>

#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

/* compute |x|_inf */
static double vec_nrm_inf(
        int64_t N,
        const double *X)
{
    double max_nrm = 0;

    for(auto row = 0; row < N ; row++) {
        double xi = X[row];
        max_nrm = ( max_nrm > fabs(xi) )? max_nrm : fabs(xi);
    }

    return max_nrm;
};

/* A is 1D laplacian, return A[N:-1:1, :] */
static void gen_1d_laplacian_perm(
        int64_t N,
        double *A,
        int64_t lda)
{
    /* set A[0:N, 0:N] = 0 */
    for(auto J=0; J < N; J++) {
        for(auto I=0; I < N; I++) {
            A[ I + J * lda ] = 0.0;
        }
    }

    /* set entries */
    for(int J = 0 ; J < N; J++ ){
        /* main diagonal */
        A[ ((N-1)-J) + J * lda ] = 2.0;

        /* upper diagonal */
        if ( J > 0 ){
            A[ ((N-1)-(J-1)) + J * lda ] = -1.0;
        }
        /* lower diagonal */
        if ( J < (N-1) ){
            A[ ((N-1)-(J+1)) + J * lda ] = -1.0;
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

    /* Get MPI global comm */
    MPI_Comm mpi_global_comm = MPI_COMM_WORLD;

    /* Define dimensions, block sizes and offsets of A and B matrices */
    const int64_t N    = 10;
    const int64_t NRHS = 1;

    /* Enable / disable pivoting */
    const bool enable_pivoting = true;

    /* Offsets of A and B matrices (base-1) */
    const int64_t IA = 3;
    const int64_t JA = 3;
    const int64_t IB = 1;
    const int64_t JB = 1;

    /* Tile sizes */
    const int64_t MA = 2;
    const int64_t NA = 2;
    const int64_t MB = 2;
    const int64_t NB = 2;

    /* Define grid of processors */
    const int numRowDevices = 2;
    const int numColDevices = 1;

    /* Current implementation only allows RSRC,CSRC=(0,0) */
    const uint32_t RSRCA = 0;
    const uint32_t CSRCA = 0;
    const uint32_t RSRCB = 0;
    const uint32_t CSRCB = 0;

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

    /* Boostrap of MPI communicator */
    calStat = cal_comm_create_distr( &mpi_global_comm, localDeviceId, &cal_comm );
    assert( calStat == CAL_OK );

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
    cudaLibMpGrid_t gridIpiv = nullptr;
    cudaLibMpGrid_t gridB    = nullptr; 

    /* cudaLib matrix descriptors */
    cudaLibMpMatrixDesc_t descrA    = nullptr;
    cudaLibMpMatrixDesc_t descrIpiv = nullptr;
    cudaLibMpMatrixDesc_t descrB    = nullptr;

    /* Distributed matrices */
    void    *d_A    = nullptr;
    int64_t *d_ipiv = nullptr;
    void    *d_B    = nullptr;

    /* Distributed device workspace */
    void *d_work_getrf = nullptr;
    void *d_work_getrs = nullptr;

    /* Distributed host workspace */
    void *h_work_getrf = nullptr;
    void *h_work_getrs = nullptr;

    /* size of workspace on device */
    size_t workspaceInBytesOnDevice_getrf = 0;
    size_t workspaceInBytesOnDevice_getrs = 0;

    /* size of workspace on host */
    size_t workspaceInBytesOnHost_getrf = 0;
    size_t workspaceInBytesOnHost_getrs = 0;

    /* error codes from cusolverMp (device) */
    int* d_info_getrf = nullptr;
    int* d_info_getrs = nullptr;

    /* error codes from cusolverMp (host) */
    int  h_info_getrf = 0;
    int  h_info_getrs = 0;

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    /* cusolverMpGetrs only supports NRHS == 1 at this point. */
    assert( NRHS == 1 );

    /* Single process per device */
    assert ( (numRowDevices * numColDevices) <= rankSize );

    const int myProcRow = rankId % numRowDevices;
    const int myProcCol = rankId / numRowDevices;

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    const int64_t lda   = (IA -1) + N;
    const int64_t colsA = (JA -1) + N;
    const int64_t ldb   = (IB -1) + N;
    const int64_t colsB = (JB -1) + NRHS;

    double  *h_A    = nullptr;
    double  *h_B    = nullptr;
    double  *h_X    = nullptr;

    if ( rankId == 0 )
    {
        /* allocate host workspace */
        h_A = (double* ) malloc ( lda * colsA * sizeof(double));
        h_X = (double* ) malloc ( ldb * colsB * sizeof(double));
        h_B = (double* ) malloc ( ldb * colsB * sizeof(double));

        /* reset host workspace */
        memset ( h_A, 0xFF,  lda * colsA * sizeof(double));
        memset ( h_X, 0xFF,  ldb * colsB * sizeof(double));
        memset ( h_B, 0xFF,  ldb * colsB * sizeof(double));

        /* pointers to the first valid entry of A, B and X */
        double *_A = &h_A[ (IA-1) + (JA-1) * lda ];
        double *_B = &h_B[ (IB-1) + (JB-1) * ldb ];

        /* Set B[IB:IB+N, JB] = 1 */
        for(int64_t i=0; i < N; i++) { _B[i] = 1.0; }

        /* Set A[IA:IA+N, JA:JA+N] = permuted laplacian */
        gen_1d_laplacian_perm( N, _A, lda);

        /* print input matrices */
        print_host_matrix ( lda, colsA, h_A, lda, "Input matrix A" );
        print_host_matrix ( ldb, colsB, h_B, ldb, "Input matrix B" );
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
    const int64_t LLDB = cusolverMpNUMROC(ldb, MB, RSRCB, rankId % numRowDevices, numRowDevices);
    const int64_t localColsB = cusolverMpNUMROC(colsB, NB, CSRCB, rankId / numRowDevices, numColDevices);

    /* Allocate global d_A */
    cudaStat = cudaMalloc( (void**)&d_A, localColsA * LLDA * sizeof(double) );
    assert( cudaStat == cudaSuccess );

    /* Allocate global d_B */
    cudaStat = cudaMalloc( (void**)&d_B, localColsB * LLDB * sizeof(double) );
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

    cusolverStat = cusolverMpCreateDeviceGrid(
            cusolverMpHandle,
            &gridB,
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
            (IA-1) + N,
            (JA-1) + N,
            MA,
            NA,
            RSRCA,
            CSRCA,
            LLDA);
            
    assert ( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    cusolverStat = cusolverMpCreateMatrixDesc(
            &descrB,
            gridB,
            CUDA_R_64F,
            (IB -1) + N,
            (JB -1) + 1,
            MB,
            NB,
            RSRCB,
            CSRCB,
            LLDB);
    assert ( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    /* Allocate global d_ipiv */
    if ( enable_pivoting == true )
    {
        /* REMARK : ipiv overlaps A[IA, JA:JA+N] as in Netlib's ScaLAPACK */
        cudaStat = cudaMalloc( (void**)&d_ipiv, localColsA * sizeof(int64_t) );
        assert( cudaStat == cudaSuccess );
    }


    /* =========================================== */
    /*             ALLOCATE D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMalloc( &d_info_getrf, sizeof(int));
    assert( cudaStat == cudaSuccess  );

    cudaStat = cudaMalloc( &d_info_getrs, sizeof(int));
    assert( cudaStat == cudaSuccess  );


    /* =========================================== */
    /*                RESET D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMemset( d_info_getrf, 0, sizeof(int));
    assert( cudaStat == cudaSuccess  );

    cudaStat = cudaMemset( d_info_getrs, 0, sizeof(int));
    assert( cudaStat == cudaSuccess  );

    /* =========================================== */
    /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
    /* =========================================== */

    cusolverStat = cusolverMpGetrf_bufferSize(
            cusolverMpHandle,
            N,
            N,
            d_A,
            IA,
            JA,
            descrA,
            d_ipiv,
            CUDA_R_64F,
            &workspaceInBytesOnDevice_getrf,
            &workspaceInBytesOnHost_getrf);
    assert ( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    cusolverStat = cusolverMpGetrs_bufferSize(
            cusolverMpHandle,
            CUBLAS_OP_N, /* only non-transposed is supported */
            N,
            NRHS,
            (const void*) d_A,
            IA,
            JA,
            descrA,
            (const int64_t*) d_ipiv,
            d_B,
            IB,
            JB,
            descrB,
            CUDA_R_64F,
            &workspaceInBytesOnDevice_getrs,
            &workspaceInBytesOnHost_getrs);
    assert ( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    /* =========================================== */
    /*         ALLOCATE PGETRF WORKSPACE            */
    /* =========================================== */

    cudaStat = cudaMalloc( (void**)&d_work_getrf, workspaceInBytesOnDevice_getrf );
    assert( cudaStat == cudaSuccess  );

    h_work_getrf = (void*) malloc( workspaceInBytesOnHost_getrf );
    assert( h_work_getrf != nullptr );


    /* =========================================== */
    /*         ALLOCATE PGETRS WORKSPACE            */
    /* =========================================== */

    cudaStat = cudaMalloc( (void**)&d_work_getrs, workspaceInBytesOnDevice_getrs );
    assert( cudaStat == cudaSuccess  );

    h_work_getrs = (void*) malloc( workspaceInBytesOnHost_getrs );
    assert( h_work_getrs != nullptr );

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

    cusolverStat = cusolverMpMatrixScatterH2D (
            cusolverMpHandle,
            ldb,
            colsB,
            (void*) d_B, /* routine requires void** */
            1,
            1,
            descrB,
            0, /* root rank */
            (void*) h_B,
            ldb );
    assert( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    /* sync wait for data to arrive to device */ 
    calStat = cal_stream_sync( cal_comm, localStream );
    assert( calStat == CAL_OK );


    /* =========================================== */
    /*                   CALL PGETRF               */
    /* =========================================== */

    cusolverStat = cusolverMpGetrf (
            cusolverMpHandle,
            N,
            N,
            d_A,
            IA,
            JA,
            descrA,
            d_ipiv,
            CUDA_R_64F,
            d_work_getrf,
            workspaceInBytesOnDevice_getrf,
            h_work_getrf,
            workspaceInBytesOnHost_getrf,
            d_info_getrf);

    /* sync after cusolverMpGetrf */
    calStat = cal_stream_sync( cal_comm, localStream );
    assert( calStat == CAL_OK );


    /* copy d_info_getrf to host */
    cudaStat = cudaMemcpyAsync(
            &h_info_getrf,
            d_info_getrf,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            localStream );
    assert( cudaStat == cudaSuccess );

    /* wait for d_info_getrf copy */
    cudaStat = cudaStreamSynchronize( localStream );
    assert( cudaStat == cudaSuccess );

    /* check return value of cusolverMpGetrf */
    assert ( h_info_getrf == 0 );


    /* =========================================== */
    /*                   CALL PGETRS               */
    /* =========================================== */

    cusolverStat = cusolverMpGetrs(
            cusolverMpHandle,
            CUBLAS_OP_N, /* only non-transposed is supported */
            N,
            NRHS,
            (const void*) d_A,
            IA,
            JA,
            descrA,
            (const int64_t*) d_ipiv,
            d_B,
            IB,
            JB,
            descrB,
            CUDA_R_64F,
            d_work_getrs,
            workspaceInBytesOnDevice_getrs,
            h_work_getrs,
            workspaceInBytesOnHost_getrs,
            d_info_getrs);

    /* sync after cusolverMpGetrs */
    calStat = cal_stream_sync( cal_comm, localStream );
    assert( calStat == CAL_OK );

    /* copy d_info_getrs to host */
    cudaStat = cudaMemcpyAsync(
            &h_info_getrs,
            d_info_getrs,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            localStream );
    assert( cudaStat == cudaSuccess );

    /* wait for d_info_getrs copy */
    cudaStat = cudaStreamSynchronize( localStream );
    assert( cudaStat == cudaSuccess );

    /* check return value of cusolverMpGetrf */
    assert( h_info_getrs == 0 );

    /* =========================================== */
    /*      GATHER MATRICES A AND B FROM MASTER    */
    /* =========================================== */

    /* Copy solution to h_X */
    cusolverStat = cusolverMpMatrixGatherD2H (
            cusolverMpHandle,
            ldb,
            colsB,
            (void*) d_B,
            1,
            1,
            descrB,
            0, /* master rank, destination */
            (void*) h_X,
            ldb );
    assert( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    /* sync wait for data to arrive to host */ 
    calStat = cal_stream_sync( cal_comm, localStream );
    assert( calStat == CAL_OK );


    /* =========================================== */
    /*            CHECK RESIDUAL ON MASTER         */ 
    /* =========================================== */
    if ( rankId == 0 )
    {
        /* print input matrices */
        print_host_matrix ( ldb, colsB, h_X, ldb, "Output matrix X" );

        /* pointers to the first valid entry of A, B and X */
        double *_A = &h_A[ (IA-1) + (JA-1) * lda ];
        double *_X = &h_X[ (IB-1) + (JB-1) * ldb ];
        double *_B = &h_B[ (IB-1) + (JB-1) * ldb ];

        /* measure residual error |b - A*x| */
        double max_err = 0;
        for(int row = 0; row < N ; row++){
            double sum = 0.0;
            for(int col = 0; col < N ; col++){
                double Aij = _A[ row + col * lda ];
                double  xj = _X[ col ];
                sum += Aij*xj;
            }
            double bi = _B[ row ];
            double err = fabs( bi - sum );

            max_err = ( max_err > err )? max_err : err;
        }

        double x_nrm_inf = vec_nrm_inf(N, _X);
        double b_nrm_inf = vec_nrm_inf(N, _B);
        double A_nrm_inf = 4.0;
        double rel_err = max_err/(A_nrm_inf * x_nrm_inf + b_nrm_inf);

        printf("\n|b - A*x|_inf = %E\n", max_err);
        printf("|x|_inf = %E\n", x_nrm_inf);
        printf("|b|_inf = %E\n", b_nrm_inf);
        printf("|A|_inf = %E\n", A_nrm_inf);

        /* relative error is around machine zero  */
        /* the user can use |b - A*x|/(N*|A|*|x|+|b|) as well */
        printf("|b - A*x|/(|A|*|x|+|b|) = %E\n\n", rel_err);
    }

    /* =========================================== */
    /*        CLEAN UP HOST WORKSPACE ON MASTER    */ 
    /* =========================================== */
    if ( rankId == 0 )
    {
        if ( h_A ) { free( h_A ); h_A = nullptr; }
        if ( h_X ) { free( h_X ); h_X = nullptr; }
        if ( h_B ) { free( h_B ); h_B = nullptr; }
    }

    /* =========================================== */
    /*           DESTROY MATRIX DESCRIPTORS        */ 
    /* =========================================== */

    cusolverStat = cusolverMpDestroyMatrixDesc( descrA );
    assert( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    cusolverStat = cusolverMpDestroyMatrixDesc( descrB );
    assert( cusolverStat == CUSOLVER_STATUS_SUCCESS );


    /* =========================================== */
    /*             DESTROY MATRIX GRIDS            */ 
    /* =========================================== */

    cusolverStat = cusolverMpDestroyGrid( gridA );
    assert( cusolverStat == CUSOLVER_STATUS_SUCCESS );

    cusolverStat = cusolverMpDestroyGrid( gridB );
    assert( cusolverStat == CUSOLVER_STATUS_SUCCESS );


    /* =========================================== */
    /*          DEALLOCATE DEVICE WORKSPACE        */ 
    /* =========================================== */

    if ( d_A != nullptr ) {
        cudaStat = cudaFree( d_A );
        assert( cudaStat == cudaSuccess );
        d_A = nullptr;
    }

    if ( d_B != nullptr ) {
        cudaStat = cudaFree( d_B );
        assert( cudaStat == cudaSuccess );
        d_B = nullptr;
    }

    if ( d_ipiv != nullptr ) {
        cudaStat = cudaFree( d_ipiv );
        assert( cudaStat == cudaSuccess );
        d_ipiv = nullptr;
    }

    if ( d_work_getrf != nullptr ) {
        cudaStat = cudaFree( d_work_getrf );
        assert( cudaStat == cudaSuccess );
        d_work_getrf = nullptr;
    }

    if ( d_work_getrs != nullptr ) {
        cudaStat = cudaFree( d_work_getrs );
        assert( cudaStat == cudaSuccess );
        d_work_getrs = nullptr;
    }

    if ( d_info_getrf != nullptr ) {
        cudaStat = cudaFree( d_info_getrf );
        assert( cudaStat == cudaSuccess );
        d_info_getrf = nullptr;
    }

    if ( d_info_getrs != nullptr ) {
        cudaStat = cudaFree( d_info_getrs );
        assert( cudaStat == cudaSuccess );
        d_info_getrs = nullptr;
    }

    /* =========================================== */
    /*         DEALLOCATE HOST WORKSPACE           */ 
    /* =========================================== */
    if ( h_work_getrf ) { free( h_work_getrf ); h_work_getrf = nullptr; }
    if ( h_work_getrs ) { free( h_work_getrs ); h_work_getrs = nullptr; }

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
