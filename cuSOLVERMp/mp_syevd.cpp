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
#include <time.h>

#include <cmath>
#include <vector>

#include <omp.h>
#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

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
    Options opts
    {
        .m = 1,
        .n = 128,
        .nrhs = 1,
        .mbA = 32,
        .nbA = 32,
        .mbB = 32,
        .nbB = 32,
        .mbQ = 32,
        .nbQ = 32,
        .ia = 1,
        .ja = 1,
        .ib = 1,
        .jb = 1,
        .iq = 1,
        .jq = 1,
        .p = 2,
        .q = 1,
        .verbose = false
    };

    opts.parse(argc, argv);
    opts.validate();
    opts.print();

    /* Initialize MPI  library */
    MPI_Init(NULL, NULL);

    char compz = 'Z';

    /* Define dimensions, block sizes and offsets of A and B matrices */
    const int64_t n = opts.n;

    /* Offsets of A and B matrices (base-1) */
    const int64_t ia = opts.ia;
    const int64_t ja = opts.jb;
    const int64_t iq = opts.iq;
    const int64_t jq = opts.jq;

    /* Tile sizes */
    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;
    const int64_t mbQ = opts.mbQ;
    const int64_t nbQ = opts.nbQ;

    /* Define grid of processors */
    const int numRowDevices = opts.p;
    const int numColDevices = opts.q;

    /* Current implementation only allows RSRC,CSRC=(0,0) */
    const uint32_t rsrca = 0;
    const uint32_t csrca = 0;
    const uint32_t rsrcq = 0;
    const uint32_t csrcq = 0;

    /* Get rank id and rank size of the com. */
    int mpiCommSize, mpiRank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpiCommSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    /* 
     * Initialize device context for this process
     */
    int localRank = getLocalRank();
    cudaError_t cudaStat = cudaSetDevice(localRank);
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaFree(0);
    assert(cudaStat == cudaSuccess);

    {
        const int rank = mpiRank;
        const int commSize = mpiCommSize;

        /* Error codes */
        cusolverStatus_t cusolverStat = CUSOLVER_STATUS_SUCCESS;
        calError_t  calStat = CAL_OK;
        cudaError_t cudaStat = cudaSuccess;

        cudaStat = cudaSetDevice(localRank);
        assert(cudaStat == cudaSuccess);

        /* Create communicator */
        cal_comm_create_params_t params;
        params.allgather = allgather;
        params.req_test = request_test;
        params.req_free = request_free;
        params.data = reinterpret_cast<void*>(MPI_COMM_WORLD);
        params.rank = rank;
        params.nranks = commSize;
        params.local_device = localRank;

        cal_comm_t cal_comm = nullptr;
        calStat = cal_comm_create(params, &cal_comm);
        assert(calStat == CAL_OK);

        /* Create local stream */
        cudaStream_t localStream = nullptr;
        cudaStat = cudaStreamCreate(&localStream);
        assert(cudaStat == cudaSuccess);

        /* Initialize cusolverMp library handle */
        cusolverMpHandle_t cusolverMpHandle = nullptr;
        cusolverStat = cusolverMpCreate(
            &cusolverMpHandle,
            localRank,
            localStream);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* cudaLigMg grids */
        cudaLibMpGrid_t gridA = nullptr; 
        cudaLibMpGrid_t gridQ = nullptr; 

        /* cudaLib matrix descriptors */
        cudaLibMpMatrixDesc_t descA = nullptr;
        cudaLibMpMatrixDesc_t descQ = nullptr;

        /* Distributed matrices */
        void *d_A = nullptr;
        void *d_D = nullptr;
        void *d_Q = nullptr;

        /* Distributed device workspace */
        void *d_syevdWork = nullptr;

        /* Distributed host workspace */
        void *h_syevdWork = nullptr;

        /* size of workspace on device */
        size_t syevdWorkspaceInBytesOnDevice = 0;

        /* size of workspace on host */
        size_t syevdWorkspaceInBytesOnHost = 0;

        /* error codes from cusolverMp (device) */
        int* d_syevdInfo = nullptr;

        /* error codes from cusolverMp (host) */
        int  h_syevdInfo = 0;

        /* =========================================== */
        /*          Create inputs on master rank       */
        /* =========================================== */

        /* Single process per device */
        assert((numRowDevices * numColDevices) <= commSize);

        /* =========================================== */
        /*          Create inputs on master rank       */
        /* =========================================== */

        const int64_t lda = (ia -1) + n;
        const int64_t colsA = (ja -1) + n;
        const int64_t ldq = (iq -1) + n;
        const int64_t colsQ = (jq -1) + n;

        double *h_A = nullptr;
        double *h_Q = nullptr;
        double *h_D = nullptr;

        if (rank == 0)
        {
            /* allocate host workspace */
            h_A = (double*)malloc(lda * colsA * sizeof(double));
            h_Q = (double*)malloc(ldq * colsQ * sizeof(double));
            h_D = (double*)malloc(n   * 1     * sizeof(double));

            /* reset host workspace */
            memset(h_A, 0xFF,  lda * colsA * sizeof(double));
            memset(h_Q, 0xFF,  ldq * colsQ * sizeof(double));
            memset(h_D, 0xFF,  n   * 1     * sizeof(double));

            double *_A = &h_A[ (ia-1) + (ja-1) * lda ];

            /* Set A[ia:ia+n, ja:ja+n] = diagonal dominant lower triangular matrix */
            generate_diagonal_dominant_symmetric_matrix(n, _A, lda);

            /* print input matrices */
            if (opts.verbose)
            {
                print_host_matrix(lda, colsA, h_A, lda, "Input matrix A");
            }
        }

        /* compute the load leading dimension of the device buffers */
        const int64_t llda = cusolverMpNUMROC(lda, mbA, rsrca, rank % numRowDevices, numRowDevices);
        const int64_t localColsA = cusolverMpNUMROC(colsA, nbA, csrca, rank / numRowDevices, numColDevices);

        const int64_t lldq = cusolverMpNUMROC(ldq, mbQ, rsrcq, rank % numRowDevices, numRowDevices);
        const int64_t localColsQ = cusolverMpNUMROC(colsQ, nbQ, csrcq, rank / numRowDevices, numColDevices);

        /* Allocate global d_A */
        cudaStat = cudaMalloc((void**)&d_A, llda * localColsA * sizeof(double));
        assert(cudaStat == cudaSuccess);

        /* Allocate global d_B */
        cudaStat = cudaMalloc((void**)&d_Q, lldq * localColsQ * sizeof(double));
        assert(cudaStat == cudaSuccess);

        /* Allocate global d_D */
        cudaStat = cudaMalloc((void**)&d_D, n * 1 * sizeof(double));
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
            &gridQ,
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
            &descQ,
            gridQ,
            CUDA_R_64F,
            ldq,
            colsQ,
            mbQ,
            nbQ,
            rsrcq,
            csrcq,
            lldq);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*             ALLOCATE D_INFO                 */
        /* =========================================== */

        cudaStat = cudaMalloc(&d_syevdInfo, sizeof(int));
        assert(cudaStat == cudaSuccess);

        /* =========================================== */
        /*                RESET D_INFO                 */
        /* =========================================== */

        cudaStat = cudaMemset(d_syevdInfo, 0, sizeof(int));
        assert(cudaStat == cudaSuccess);

        /* =========================================== */
        /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
        /* =========================================== */

        cusolverStat = cusolverMpSyevd_bufferSize(
            cusolverMpHandle,
            &compz,
            CUBLAS_FILL_MODE_LOWER,
            n,
            d_A,
            ia,
            ja,
            descA,
            d_D,
            d_Q,
            iq,
            jq,
            descQ,
            CUDA_R_64F,
            &syevdWorkspaceInBytesOnDevice,
            &syevdWorkspaceInBytesOnHost);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*         ALLOCATE Psyevd WORKSPACE            */
        /* =========================================== */

        cudaStat = cudaMalloc((void**)&d_syevdWork, syevdWorkspaceInBytesOnDevice);
        assert(cudaStat == cudaSuccess);

        h_syevdWork = (void*)malloc(syevdWorkspaceInBytesOnHost);
        assert(h_syevdWork != nullptr);

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
            ldq,
            colsQ,
            (void*) d_Q,
            1,
            1,
            descQ,
            0, /* root rank */
            (void*) h_Q,
            ldq);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync wait for data to arrive to device */ 
        calStat = cal_stream_sync(cal_comm, localStream);
        assert(calStat == CAL_OK);

        /* =========================================== */
        /*                   CALL psyevd               */
        /* =========================================== */

        cusolverStat = cusolverMpSyevd(
            cusolverMpHandle,
            &compz,
            CUBLAS_FILL_MODE_LOWER,
            n,
            d_A,
            ia,
            ja,
            descA,
            d_D,
            d_Q,
            iq,
            jq,
            descQ,
            CUDA_R_64F,
            d_syevdWork,
            syevdWorkspaceInBytesOnDevice,
            h_syevdWork,
            syevdWorkspaceInBytesOnHost,
            d_syevdInfo);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync after cusolverMpsyevd */
        calStat = cal_stream_sync(cal_comm, localStream);
        assert(calStat == CAL_OK);

        /* copy d_syevdInfo to host */
        cudaStat = cudaMemcpyAsync(
            &h_syevdInfo,
            d_syevdInfo,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            localStream);
        assert(cudaStat == cudaSuccess);

        /* wait for d_syevdInfo copy */
        cudaStat = cudaStreamSynchronize(localStream);
        assert(cudaStat == cudaSuccess);

        /* check return value of cusolverMpsyevd */
        assert(h_syevdInfo == 0);

        /* =========================================== */
        /*      GATHER MATRICES A AND Q FROM MASTER    */
        /* =========================================== */

        /* Copy solution to h_Q */
        cusolverStat = cusolverMpMatrixGatherD2H(
            cusolverMpHandle,
            ldq,
            colsQ,
            (void*) d_Q,
            1,
            1,
            descQ,
            0, /* master rank, destination */
            (void*) h_Q,
            ldq);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync wait for data to arrive to host */ 
        calStat = cal_stream_sync(cal_comm, localStream);
        assert(calStat == CAL_OK);

        /* =========================================== */
        /*        CLEAN UP HOST WORKSPACE ON MASTER    */ 
        /* =========================================== */
        if (rank == 0)
        {
            if (h_A)
            {
                free(h_A);
                h_A = nullptr;
            }

            if (h_Q)
            {
                free(h_Q);
                h_Q = nullptr;
            }

            if (h_D)
            {
                free(h_D);
                h_D = nullptr;
            }
        }

        /* =========================================== */
        /*           DESTROY MATRIX DESCRIPTORS        */ 
        /* =========================================== */

        cusolverStat = cusolverMpDestroyMatrixDesc(descA);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverMpDestroyMatrixDesc(descQ);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*             DESTROY MATRIX GRIDS            */ 
        /* =========================================== */

        cusolverStat = cusolverMpDestroyGrid(gridA);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverMpDestroyGrid(gridQ);
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

        if (d_Q)
        {
            cudaStat = cudaFree(d_Q);
            assert(cudaStat == cudaSuccess);
            d_Q = nullptr;
        }

        if (d_D)
        {
            cudaStat = cudaFree(d_D);
            assert(cudaStat == cudaSuccess);
            d_D = nullptr;
        }

        if (d_syevdWork)
        {
            cudaStat = cudaFree(d_syevdWork);
            assert(cudaStat == cudaSuccess);
            d_syevdWork = nullptr;
        }

        if (d_syevdInfo)
        {
            cudaStat = cudaFree(d_syevdInfo);
            assert(cudaStat == cudaSuccess);
            d_syevdInfo = nullptr;
        }

        /* =========================================== */
        /*         DEALLOCATE HOST WORKSPACE           */ 
        /* =========================================== */
        if (h_syevdWork)
        {
            free(h_syevdWork);
            h_syevdWork = nullptr;
        }

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
    }

    /* MPI barrier before MPI_Finalize */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Finalize MPI environment */
    MPI_Finalize();

    return 0;
};
