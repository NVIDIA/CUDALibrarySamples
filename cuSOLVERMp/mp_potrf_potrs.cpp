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
#include <time.h>

#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

/* compute |x|_inf */
static double normI(
    int64_t m,
    int64_t n,
    int64_t lda,
    const double *A)
{
    double max_nrm = 0;

    for(auto col = 0; col < n ; col++)
    {
        double err = 0;
        for(auto row = 0; row < m ; row++)
        {
            err += std::fabs(A[row + col * lda]);
        }

        max_nrm = std::fmax(max_nrm, err);
    }

    return max_nrm;
};

/* A is 1D laplacian, return A[n:-1:1, :] */
static void generate_diagonal_dominant_symmetric_matrix(
    int64_t n,
    double *A,
    int64_t lda)
{
    time(NULL);

    /* set A[0:n, 0:n] = 0 */
    for(auto j = 0; j < n; j++)
    {
        double sum = 0;
        for(auto i = 0; i < n; i++)
        {
            if (i < j)
            {
                A[ i + j * lda ] = A[ j + i * lda ];
            }
            else
            {
                A[ i + j * lda ] = double(rand()) / RAND_MAX;
                sum += A[ i + j * lda ];
            }
        }

        A[ j + j * lda ] = 2 * sum;
    }
}

/* Print matrix */
static void print_host_matrix (
    int64_t m,
    int64_t n,
    double *A,
    int64_t lda,
    const char *msg)
{
    printf("print_host_matrix : %s\n", msg);

    for(auto i = 0; i < m; i++)
    {
        for(auto j = 0; j < n; j++)
        {
            printf("%.2lf  ", A[i + j * lda]);
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
    const int64_t n = 10;
    const int64_t nrhs = 1;

    /* Offsets of A and B matrices (base-1) */
    const int64_t ia = 3;
    const int64_t ja = 3;
    const int64_t ib = 3;
    const int64_t jb = 1;

    /* Tile sizes */
    const int64_t mbA = 2;
    const int64_t nbA = 2;
    const int64_t mbB = 2;
    const int64_t nbB = 2;

    /* Define grid of processors */
    const int numRowDevices = 2;
    const int numColDevices = 1;

    /* Current implementation only allows RSRC,CSRC=(0,0) */
    const uint32_t rsrca = 0;
    const uint32_t csrca = 0;
    const uint32_t rsrcb = 0;
    const uint32_t csrcb = 0;

    /* Get rank id and rank size of the com. */
    int rankSize, rankId;
    MPI_Comm_size(MPI_COMM_WORLD, &rankSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);


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
    calStat = cal_comm_create_distr(&mpi_global_comm, localDeviceId, &cal_comm);
    assert(calStat == CAL_OK);

    /* Create local stream */
    cudaStat = cudaStreamCreate(&localStream);
    assert(cudaStat == cudaSuccess);

    /* Initialize cusolverMp library handle */
    cusolverStat = cusolverMpCreate(
        &cusolverMpHandle,
        localDeviceId,
        localStream);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* cudaLigMg grids */
    cudaLibMpGrid_t gridA = nullptr; 
    cudaLibMpGrid_t gridB = nullptr; 

    /* cudaLib matrix descriptors */
    cudaLibMpMatrixDesc_t descA = nullptr;
    cudaLibMpMatrixDesc_t descB = nullptr;

    /* Distributed matrices */
    void *d_A = nullptr;
    void *d_B = nullptr;

    /* Distributed device workspace */
    void *d_potrfWork = nullptr;
    void *d_potrsWork = nullptr;

    /* Distributed host workspace */
    void *h_potrfWork = nullptr;
    void *h_potrsWork = nullptr;

    /* size of workspace on device */
    size_t potrfWorkspaceInBytesOnDevice = 0;
    size_t potrsWorkspaceInBytesOnDevice = 0;

    /* size of workspace on host */
    size_t potrfWorkspaceInBytesOnHost = 0;
    size_t potrsWorkspaceInBytesOnHost = 0;

    /* error codes from cusolverMp (device) */
    int* d_potrfInfo = nullptr;
    int* d_potrsInfo = nullptr;

    /* error codes from cusolverMp (host) */
    int  h_potrfInfo = 0;
    int  h_potrsInfo = 0;

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    /* cusolverMppotrs only supports nrhs == 1 at this point. */
    assert(nrhs == 1);

    /* Single process per device */
    assert((numRowDevices * numColDevices) <= rankSize);

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    const int64_t lda = (ia -1) + n;
    const int64_t colsA = (ja -1) + n;
    const int64_t ldb = (ib -1) + n;
    const int64_t colsB = (jb -1) + nrhs;

    double *h_A = nullptr;
    double *h_B = nullptr;
    double *h_X = nullptr;

    if (rankId == 0)
    {
        /* allocate host workspace */
        h_A = (double*)malloc(lda * colsA * sizeof(double));
        h_X = (double*)malloc(ldb * colsB * sizeof(double));
        h_B = (double*)malloc(ldb * colsB * sizeof(double));

        /* reset host workspace */
        memset(h_A, 0xFF,  lda * colsA * sizeof(double));
        memset(h_X, 0xFF,  ldb * colsB * sizeof(double));
        memset(h_B, 0xFF,  ldb * colsB * sizeof(double));

        /* pointers to the first valid entry of A and B */
        double *_A = &h_A[ (ia-1) + (ja-1) * lda ];
        double *_B = &h_B[ (ib-1) + (jb-1) * ldb ];

        /* Set A[ia:ia+n, ja:ja+n] = diagonal dominant lower triangular matrix */
        generate_diagonal_dominant_symmetric_matrix(n, _A, lda);

        /* Set B[ib:ib+n, jb] = 1 */
        for(int64_t i = 0; i < n; i++) { _B[i] = 1.0; }

        /* print input matrices */
        print_host_matrix(lda, colsA, h_A, lda, "Input matrix A");
        print_host_matrix(ldb, colsB, h_B, ldb, "Input matrix B");
    }

    /* compute the load leading dimension of the device buffers */
    const int64_t llda = cusolverMpNUMROC(lda, mbA, rsrca, rankId % numRowDevices, numRowDevices);
    const int64_t localColsA = cusolverMpNUMROC(colsA, nbA, csrca, rankId / numRowDevices, numColDevices);

    const int64_t lldb = cusolverMpNUMROC(ldb, mbB, rsrcb, rankId % numRowDevices, numRowDevices);
    const int64_t localColsB = cusolverMpNUMROC(colsB, nbB, csrcb, rankId / numRowDevices, numColDevices);

    /* Allocate global d_A */
    cudaStat = cudaMalloc((void**)&d_A, llda * localColsA * sizeof(double));
    assert(cudaStat == cudaSuccess);

    /* Allocate global d_B */
    cudaStat = cudaMalloc((void**)&d_B, lldb * localColsB * sizeof(double));
    assert(cudaStat == cudaSuccess);

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
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpCreateDeviceGrid(
        cusolverMpHandle,
        &gridB,
        cal_comm,
        numRowDevices,
        numColDevices,
        CUDALIBMP_GRID_MAPPING_COL_MAJOR);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*        CREATE MATRIX DESCRIPTORS            */
    /* =========================================== */
    cusolverStat = cusolverMpCreateMatrixDesc(
        &descA,
        gridA,
        CUDA_R_64F,
        lda,
        colsA,
        mbA,
        nbA,
        rsrca,
        csrca,
        llda);

    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpCreateMatrixDesc(
        &descB,
        gridB,
        CUDA_R_64F,
        ldb,
        colsB,
        mbB,
        nbB,
        rsrcb,
        csrcb,
        lldb);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*             ALLOCATE D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMalloc(&d_potrfInfo, sizeof(int));
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc(&d_potrsInfo, sizeof(int));
    assert(cudaStat == cudaSuccess);


    /* =========================================== */
    /*                RESET D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMemset(d_potrfInfo, 0, sizeof(int));
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMemset(d_potrsInfo, 0, sizeof(int));
    assert(cudaStat == cudaSuccess);

    /* =========================================== */
    /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
    /* =========================================== */

    cusolverStat = cusolverMpPotrf_bufferSize(
        cusolverMpHandle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_A,
        ia,
        ja,
        descA,
        CUDA_R_64F,
        &potrfWorkspaceInBytesOnDevice,
        &potrfWorkspaceInBytesOnHost);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpPotrs_bufferSize(
        cusolverMpHandle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        nrhs,
        (const void*) d_A,
        ia,
        ja,
        descA,
        d_B,
        ib,
        jb,
        descB,
        CUDA_R_64F,
        &potrsWorkspaceInBytesOnDevice,
        &potrsWorkspaceInBytesOnHost);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*         ALLOCATE Ppotrf WORKSPACE            */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_potrfWork, potrfWorkspaceInBytesOnDevice);
    assert(cudaStat == cudaSuccess);

    h_potrfWork = (void*)malloc(potrfWorkspaceInBytesOnHost);
    assert(h_potrfWork != nullptr);

    /* =========================================== */
    /*         ALLOCATE Ppotrs WORKSPACE            */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_potrsWork, potrsWorkspaceInBytesOnDevice);
    assert(cudaStat == cudaSuccess);

    h_potrsWork = (void*)malloc(potrsWorkspaceInBytesOnHost);
    assert(h_potrsWork != nullptr);

    /* =========================================== */
    /*      SCATTER MATRICES A AND B FROM MASTER   */
    /* =========================================== */
    cusolverStat = cusolverMpMatrixScatterH2D(
        cusolverMpHandle,
        lda,
        colsA,
        (void*) d_A,
        1,
        1,
        descA,
        0, /* root rank */
        (void*) h_A,
        lda);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpMatrixScatterH2D(
        cusolverMpHandle,
        ldb,
        colsB,
        (void*) d_B,
        1,
        1,
        descB,
        0, /* root rank */
        (void*) h_B,
        ldb);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync wait for data to arrive to device */ 
    calStat = cal_stream_sync(cal_comm, localStream);
    assert(calStat == CAL_OK);


    /* =========================================== */
    /*                   CALL Ppotrf               */
    /* =========================================== */

    cusolverStat = cusolverMpPotrf(
        cusolverMpHandle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_A,
        ia,
        ja,
        descA,
        CUDA_R_64F,
        d_potrfWork,
        potrfWorkspaceInBytesOnDevice,
        h_potrfWork,
        potrfWorkspaceInBytesOnHost,
        d_potrfInfo);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync after cusolverMppotrf */
    calStat = cal_stream_sync(cal_comm, localStream);
    assert(calStat == CAL_OK);

    /* copy d_potrfInfo to host */
    cudaStat = cudaMemcpyAsync(
        &h_potrfInfo,
        d_potrfInfo,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        localStream);
    assert(cudaStat == cudaSuccess);

    /* wait for d_potrfInfo copy */
    cudaStat = cudaStreamSynchronize(localStream);
    assert(cudaStat == cudaSuccess);

    /* check return value of cusolverMppotrf */
    assert(h_potrfInfo == 0);

    /* =========================================== */
    /*                   CALL Ppotrs               */
    /* =========================================== */

    cusolverStat = cusolverMpPotrs(
        cusolverMpHandle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        nrhs,
        (const void*) d_A,
        ia,
        ja,
        descA,
        d_B,
        ib,
        jb,
        descB,
        CUDA_R_64F,
        d_potrsWork,
        potrsWorkspaceInBytesOnDevice,
        h_potrsWork,
        potrsWorkspaceInBytesOnHost,
        d_potrsInfo);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync after cusolverMppotrs */
    calStat = cal_stream_sync(cal_comm, localStream);
    assert(calStat == CAL_OK);

    /* copy d_potrsInfo to host */
    cudaStat = cudaMemcpyAsync(
        &h_potrsInfo,
        d_potrsInfo,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        localStream);
    assert(cudaStat == cudaSuccess);

    /* wait for d_potrsInfo copy */
    cudaStat = cudaStreamSynchronize(localStream);
    assert(cudaStat == cudaSuccess);

    /* check return value of cusolverMppotrf */
    assert(h_potrsInfo == 0);

    /* =========================================== */
    /*      GATHER MATRICES A AND B FROM MASTER    */
    /* =========================================== */

    /* Copy solution to h_X */
    cusolverStat = cusolverMpMatrixGatherD2H(
        cusolverMpHandle,
        ldb,
        colsB,
        (void*) d_B,
        1,
        1,
        descB,
        0, /* master rank, destination */
        (void*) h_X,
        ldb);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync wait for data to arrive to host */ 
    calStat = cal_stream_sync(cal_comm, localStream);
    assert(calStat == CAL_OK);


    /* =========================================== */
    /*            CHECK RESIDUAL ON MASTER         */ 
    /* =========================================== */
    if (rankId == 0)
    {
        /* print input matrices */
        print_host_matrix(ldb, colsB, h_X, ldb, "Output matrix X");

        /* pointers to the first valid entry of A, B and X */
        double *_A = &h_A[ (ia-1) + (ja-1) * lda ];
        double *_X = &h_X[ (ib-1) + (jb-1) * ldb ];
        double *_B = &h_B[ (ib-1) + (jb-1) * ldb ];

        /* measure residual error |b - A*x| */
        double max_err = 0;
        for(int row = 0; row < n ; row++)
        {
            double sum = 0.0;
            for(int col = 0; col < n ; col++)
            {
                double Aij = _A[ row + col * lda ];
                double  xj = _X[ col ];
                sum += Aij*xj;
            }
            double bi = _B[ row ];
            double err = std::fabs(bi - sum);

            max_err = std::fmax(max_err, err);
        }

        double x_nrm_inf = normI(n, 1, ldb, _X);
        double b_nrm_inf = normI(n, 1, ldb, _B);
        double A_nrm_inf = normI(n, n, lda, _A);
        double rel_err = max_err/(A_nrm_inf * x_nrm_inf + b_nrm_inf);

        printf("\n|b - A*x|_inf = %E\n", max_err);
        printf("|x|_inf = %E\n", x_nrm_inf);
        printf("|b|_inf = %E\n", b_nrm_inf);
        printf("|A|_inf = %E\n", A_nrm_inf);

        /* relative error is around machine zero  */
        /* the user can use |b - A*x|/(n*|A|*|x|+|b|) as well */
        printf("|b - A*x|/(|A|*|x|+|b|) = %E\n\n", rel_err);
    }

    /* =========================================== */
    /*        CLEAN UP HOST WORKSPACE ON MASTER    */ 
    /* =========================================== */
    if (rankId == 0)
    {
        if (h_A) { free(h_A); h_A = nullptr; }
        if (h_X) { free(h_X); h_X = nullptr; }
        if (h_B) { free(h_B); h_B = nullptr; }
    }

    /* =========================================== */
    /*           DESTROY MATRIX DESCRIPTORS        */ 
    /* =========================================== */

    cusolverStat = cusolverMpDestroyMatrixDesc(descA);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpDestroyMatrixDesc(descB);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*             DESTROY MATRIX GRIDS            */ 
    /* =========================================== */

    cusolverStat = cusolverMpDestroyGrid(gridA);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpDestroyGrid(gridB);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*          DEALLOCATE DEVICE WORKSPACE        */ 
    /* =========================================== */

    if (d_A)
    {
        cudaStat = cudaFree(d_A);
        assert(cudaStat == cudaSuccess);
        d_A = nullptr;
    }

    if (d_B)
    {
        cudaStat = cudaFree(d_B);
        assert(cudaStat == cudaSuccess);
        d_B = nullptr;
    }

    if (d_potrfWork)
    {
        cudaStat = cudaFree(d_potrfWork);
        assert(cudaStat == cudaSuccess);
        d_potrfWork = nullptr;
    }

    if (d_potrsWork)
    {
        cudaStat = cudaFree(d_potrsWork);
        assert(cudaStat == cudaSuccess);
        d_potrsWork = nullptr;
    }

    if (d_potrfInfo)
    {
        cudaStat = cudaFree(d_potrfInfo);
        assert(cudaStat == cudaSuccess);
        d_potrfInfo = nullptr;
    }

    if (d_potrsInfo)
    {
        cudaStat = cudaFree(d_potrsInfo);
        assert(cudaStat == cudaSuccess);
        d_potrsInfo = nullptr;
    }

    /* =========================================== */
    /*         DEALLOCATE HOST WORKSPACE           */ 
    /* =========================================== */
    if (h_potrfWork) { free(h_potrfWork); h_potrfWork = nullptr; }
    if (h_potrsWork) { free(h_potrsWork); h_potrsWork = nullptr; }

    /* =========================================== */
    /*                      CLEANUP                */ 
    /* =========================================== */

    /* Destroy cusolverMp handle */
    cusolverStat = cusolverMpDestroy(cusolverMpHandle);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync before cal_comm_destroy */
    calStat = cal_comm_barrier(cal_comm, localStream);
    assert(calStat == CAL_OK);

    /* destroy CAL communicator */
    calStat = cal_comm_destroy(cal_comm);
    assert(calStat == CAL_OK);

    /* destroy user stream */
    cudaStat = cudaStreamDestroy(localStream);
    assert(cudaStat == cudaSuccess);

    /* MPI barrier before MPI_Finalize */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Finalize MPI environment */
    MPI_Finalize();

    return 0;
};
