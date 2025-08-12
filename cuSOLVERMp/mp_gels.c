/*
 * Copyright 2025 NVIDIA Corporation.  All rights reserved.
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


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#include <omp.h>
#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

/* print matrix */
static void print_host_matrix(const int64_t m, const int64_t n, const double* A, const int64_t lda, const char* msg)
{
    printf("print_host_matrix : %s\n", msg);

    for (int64_t i = 0; i < m; i++)
    {
        for (int64_t j = 0; j < n; j++)
            printf("%.6e  ", A[i + j * lda]);
        printf("\n");
    }
}

/* set random matrix A, dim(A) = [m,n] */
static void generate_random_matrix(const int64_t m, const int64_t n, double* A, const int64_t lda)
{
    for (int64_t j = 0; j < n; ++j)
    {
        for (int64_t i = 0; i < m; ++i)
        {
            A[i + j * lda] = (double)(rand()) / RAND_MAX;
        }
    }
}

/* compute error for A X = B, dim(A) = [m,n], dim(B) = [m,nrhs], dim(X) = [n,nrhs] */
static void compute_rhs(const int64_t m,
                        const int64_t n,
                        const int64_t nrhs,
                        const double* A,
                        const int64_t lda,
                        const double* X,
                        const int64_t ldx,
                        double*       B,
                        const int64_t ldb)
{
    for (int64_t j = 0; j < nrhs; ++j)
    {
        for (int64_t i = 0; i < m; ++i)
        {
            double tmp = 0;
            for (int64_t k = 0; k < n; ++k)
            {
                tmp += A[i + k * lda] * X[k + j * ldx];
            }
            B[i + j * ldb] = tmp;
        }
    }
}

/* compute two norm of matrix A */
static double compute_norm2(const int64_t m, const int64_t n, const double* A, const int64_t lda)
{
    double norm_A = 0;
    for (int64_t j = 0; j < n; ++j)
    {
        for (int64_t i = 0; i < m; ++i)
        {
            const double val = A[i + j * lda];
            norm_A += val * val;
        }
    }
    return sqrt(norm_A);
}

/* compute error for A X = B, dim(A) = [m,n], dim(B) = [m,nrhs], dim(X) = [n,nrhs] */
static double compute_error(const int64_t m,
                            const int64_t n,
                            const int64_t nrhs,
                            const double* A,
                            const int64_t lda,
                            const double* X,
                            const int64_t ldx,
                            double*       B,
                            const int64_t ldb)
{
    /// | AX-B | / max(m,n) |A| |X|
    const double norm_A = compute_norm2(m, n, A, lda);
    const double norm_X = compute_norm2(n, nrhs, X, ldx);

    for (int64_t j = 0; j < nrhs; ++j)
    {
        for (int64_t i = 0; i < m; ++i)
        {
            double tmp = 0;
            for (int64_t k = 0; k < n; ++k)
            {
                tmp += A[i + k * lda] * X[k + j * ldx];
            }
            B[i + j * ldb] -= tmp;
        }
    }
    const double norm_R = compute_norm2(m, nrhs, B, ldb);

    return (norm_R / fmax(m, n) / norm_A / norm_X);
}


int main(int argc, char* argv[])
{
    Options opts = { .m           = 20,
                     .n           = 8,
                     .nrhs        = 1,
                     .mbA         = 2,
                     .nbA         = 2,
                     .mbB         = 2,
                     .nbB         = 2,
                     .mbQ         = 1,
                     .nbQ         = 1,
                     .ia          = 3,
                     .ja          = 1,
                     .ib          = 3,
                     .jb          = 1,
                     .iq          = 1,
                     .jq          = 1,
                     .p           = 1,
                     .q           = 2,
                     .grid_layout = 'C',
                     .verbose     = false };

    parse(&opts, argc, argv);
    validate(&opts);

    /* Initialize MPI library */
    MPI_Init(NULL, NULL);

    /* Get MPI global comm */
    MPI_Comm mpi_global_comm = MPI_COMM_WORLD;

    /* Get rank id and rank size of the comm. */
    int mpiCommSize, mpiRank;
    MPI_Comm_size(mpi_global_comm, &mpiCommSize);
    MPI_Comm_rank(mpi_global_comm, &mpiRank);

    if (mpiRank == 0) print(&opts);

    /* Define dimensions, block sizes and offsets of A and B matrices */
    const int64_t m    = opts.m;
    const int64_t n    = opts.n;
    const int64_t nrhs = opts.nrhs;

    /* Current implementation supports over-determined system without transpose (m >= n) case */
    assert(m >= n);

    /* Offsets of A and B matrices (base-1) */
    const int64_t ia = opts.ia;
    const int64_t ja = opts.ja;
    const int64_t ib = opts.ib;
    const int64_t jb = opts.jb;

    /* Tile sizes */
    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;
    const int64_t mbB = opts.mbB;
    const int64_t nbB = opts.nbB;

    /* Define grid of processors */
    const int numRowDevices = opts.p;
    const int numColDevices = opts.q;

    /* Convert grid layout to cusolverMp grid mapping */
    const cusolverMpGridMapping_t gridLayout =
            (opts.grid_layout == 'C' || opts.grid_layout == 'c' ? CUSOLVERMP_GRID_MAPPING_COL_MAJOR
                                                                : CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);

    /* Current implementation only allows RSRC,CSRC=(0,0) */
    const uint32_t rsrca = 0;
    const uint32_t csrca = 0;
    const uint32_t rsrcb = 0;
    const uint32_t csrcb = 0;

    /*
     * Initialize device context for this process
     */
    int         localRank = getLocalRank();
    cudaError_t cudaStat  = cudaSetDevice(localRank);
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaFree(0);
    assert(cudaStat == cudaSuccess);

    {
        const int rank     = mpiRank;
        const int commSize = mpiCommSize;

        /* Error codes */
        cusolverStatus_t cusolverStat = CUSOLVER_STATUS_SUCCESS;
        ncclResult_t     ncclStat     = ncclSuccess;
        cudaError_t      cudaStat     = cudaSuccess;

        cudaStat = cudaSetDevice(localRank);
        assert(cudaStat == cudaSuccess);

        /* Create communicator */
        ncclUniqueId id;

        if (rank == 0)
        {
            ncclGetUniqueId(&id);
        }

        MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, mpi_global_comm);

        ncclComm_t comm;
        ncclStat = ncclCommInitRank(&comm, commSize, id, rank);
        assert(ncclStat == ncclSuccess);

        /* Create local stream */
        cudaStream_t localStream = NULL;
        cudaStat                 = cudaStreamCreate(&localStream);
        assert(cudaStat == cudaSuccess);

        /* Initialize cusolverMp library handle */
        cusolverMpHandle_t cusolverMpHandle = NULL;
        cusolverStat                        = cusolverMpCreate(&cusolverMpHandle, localRank, localStream);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* cusolverMp grids */
        cusolverMpGrid_t gridA = NULL;
        cusolverMpGrid_t gridB = NULL;

        /* cusolverMp matrix descriptors */
        cusolverMpMatrixDescriptor_t descA = NULL;
        cusolverMpMatrixDescriptor_t descB = NULL;

        /* Distributed matrices A X = B */
        void* d_A = NULL;
        void* d_B = NULL;

        /* Distributed device workspace */
        void* d_gelsWork = NULL;

        /* Distributed host workspace */
        void* h_gelsWork = NULL;

        /* size of workspace on device */
        size_t gelsWorkspaceInBytesOnDevice = 0;

        /* size of workspace on host */
        size_t gelsWorkspaceInBytesOnHost = 0;

        /* error codes from cusolverMp (device) */
        int* d_gelsInfo = NULL;

        /* error codes from cusolverMp (host) */
        int h_gelsInfo = 0;

        /* =========================================== */
        /*          Create inputs on master rank       */
        /* =========================================== */

        /* Single process per device */
        assert((numRowDevices * numColDevices) <= commSize);

        /* =========================================== */
        /*          Create inputs on master rank       */
        /* =========================================== */

        const int64_t m_global_A = (ia - 1) + m;
        const int64_t n_global_A = (ja - 1) + n;
        const int64_t m_global_B = (ib - 1) + m;
        const int64_t n_global_B = (jb - 1) + nrhs;

        double* h_A = NULL;
        double* h_B = NULL;
        double* h_X = NULL;

        if (rank == 0)
        {
            /* allocate host workspace */
            h_A = (double*)malloc(m_global_A * n_global_A * sizeof(double));
            h_B = (double*)malloc(m_global_B * n_global_B * sizeof(double));
            h_X = (double*)malloc(m_global_B * n_global_B * sizeof(double));

            /* reset host workspace */
            memset(h_A, 0xFF, m_global_A * n_global_A * sizeof(double));
            memset(h_B, 0xFF, m_global_B * n_global_B * sizeof(double));
            memset(h_X, 0xFF, m_global_B * n_global_B * sizeof(double));

            /* Set A random */
            double* ptr_A = &h_A[(ia - 1) + (ja - 1) * m_global_A];
            double* ptr_B = &h_B[(ib - 1) + (jb - 1) * m_global_B];
            double* ptr_X = &h_X[(ib - 1) + (jb - 1) * m_global_B];

            generate_random_matrix(m, n, ptr_A, m_global_A);
            generate_random_matrix(n, nrhs, ptr_X, m_global_B);
            compute_rhs(m, n, nrhs, ptr_A, m_global_A, ptr_X, m_global_B, ptr_B, m_global_B);

            /* print input matrices */
            if (opts.verbose)
            {
                print_host_matrix(m_global_A, n_global_A, h_A, m_global_A, "Input matrix A");
                print_host_matrix(m_global_B, n_global_B, h_B, m_global_B, "Input matrix B");
            }
        }

        /* compute process grid index */
        int myRowRank, myColRank;
        if (gridLayout == CUSOLVERMP_GRID_MAPPING_COL_MAJOR)
        {
            myRowRank = rank % numRowDevices;
            myColRank = rank / numRowDevices;
        }
        else
        {
            myRowRank = rank / numColDevices;
            myColRank = rank % numColDevices;
        }

        /* compute the load leading dimension of the device buffers */
        const int64_t m_local_A = cusolverMpNUMROC(m_global_A, mbA, myRowRank, rsrca, numRowDevices);
        const int64_t n_local_A = cusolverMpNUMROC(n_global_A, nbA, myColRank, csrca, numColDevices);

        const int64_t m_local_B = cusolverMpNUMROC(m_global_B, mbB, myRowRank, rsrcb, numRowDevices);
        const int64_t n_local_B = cusolverMpNUMROC(n_global_B, nbB, myColRank, csrcb, numColDevices);

        /* Allocate global d_A */
        cudaStat = cudaMalloc((void**)&d_A, m_local_A * n_local_A * sizeof(double));
        assert(cudaStat == cudaSuccess);

        /* Allocate global d_B */
        cudaStat = cudaMalloc((void**)&d_B, m_local_B * n_local_B * sizeof(double));
        assert(cudaStat == cudaSuccess);

        /* =========================================== */
        /*          CREATE GRID DESCRIPTORS            */
        /* =========================================== */
        cusolverStat =
                cusolverMpCreateDeviceGrid(cusolverMpHandle, &gridA, comm, numRowDevices, numColDevices, gridLayout);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat =
                cusolverMpCreateDeviceGrid(cusolverMpHandle, &gridB, comm, numRowDevices, numColDevices, gridLayout);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*        CREATE MATRIX DESCRIPTORS            */
        /* =========================================== */
        cusolverStat = cusolverMpCreateMatrixDesc(
                &descA, gridA, CUDA_R_64F, m_global_A, n_global_A, mbA, nbA, rsrca, csrca, m_local_A);

        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverMpCreateMatrixDesc(
                &descB, gridB, CUDA_R_64F, m_global_B, n_global_B, mbB, nbB, rsrcb, csrcb, m_local_B);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*             ALLOCATE D_INFO                 */
        /* =========================================== */

        cudaStat = cudaMalloc((void**)&d_gelsInfo, sizeof(int));
        assert(cudaStat == cudaSuccess);

        /* =========================================== */
        /*                RESET D_INFO                 */
        /* =========================================== */

        cudaStat = cudaMemset(d_gelsInfo, 0, sizeof(int));
        assert(cudaStat == cudaSuccess);

        /* =========================================== */
        /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
        /* =========================================== */

        cusolverStat = cusolverMpGels_bufferSize(cusolverMpHandle,
                                                 CUBLAS_OP_N,
                                                 m,
                                                 n,
                                                 nrhs,
                                                 d_A,
                                                 ia,
                                                 ja,
                                                 descA,
                                                 d_B,
                                                 ib,
                                                 jb,
                                                 descB,
                                                 CUDA_R_64F,
                                                 &gelsWorkspaceInBytesOnDevice,
                                                 &gelsWorkspaceInBytesOnHost);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*         ALLOCATE GELS WORKSPACE             */
        /* =========================================== */

        cudaStat = cudaMalloc((void**)&d_gelsWork, gelsWorkspaceInBytesOnDevice);
        assert(cudaStat == cudaSuccess);

        h_gelsWork = (void*)malloc(gelsWorkspaceInBytesOnHost);
        assert(h_gelsWork != NULL);

        /* =========================================== */
        /*      SCATTER MATRICES A AND B FROM MASTER   */
        /* =========================================== */
        cusolverStat = cusolverMpMatrixScatterH2D(cusolverMpHandle,
                                                  m_global_A,
                                                  n_global_A,
                                                  (void*)d_A,
                                                  1,
                                                  1,
                                                  descA,
                                                  0, /* root rank */
                                                  (void*)h_A,
                                                  m_global_A);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverMpMatrixScatterH2D(cusolverMpHandle,
                                                  m_global_B,
                                                  n_global_B,
                                                  (void*)d_B,
                                                  1,
                                                  1,
                                                  descB,
                                                  0, /* root rank */
                                                  (void*)h_B,
                                                  m_global_B);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync wait for data to arrive to device */
        cudaStat = cudaStreamSynchronize(localStream);
        assert(cudaStat == cudaSuccess);

        /* =========================================== */
        /*                   CALL PGELS                */
        /* =========================================== */

        cusolverStat = cusolverMpGels(cusolverMpHandle,
                                      CUBLAS_OP_N,
                                      m,
                                      n,
                                      nrhs,
                                      d_A,
                                      ia,
                                      ja,
                                      descA,
                                      d_B,
                                      ib,
                                      jb,
                                      descB,
                                      CUDA_R_64F,
                                      d_gelsWork,
                                      gelsWorkspaceInBytesOnDevice,
                                      h_gelsWork,
                                      gelsWorkspaceInBytesOnHost,
                                      d_gelsInfo);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync after PGELS */
        cudaStat = cudaStreamSynchronize(localStream);
        assert(cudaStat == cudaSuccess);

        /* copy d_gelsInfo to host */
        cudaStat = cudaMemcpyAsync(&h_gelsInfo, d_gelsInfo, sizeof(int), cudaMemcpyDeviceToHost, localStream);
        assert(cudaStat == cudaSuccess);

        /* wait for d_gelsInfo copy */
        cudaStat = cudaStreamSynchronize(localStream);
        assert(cudaStat == cudaSuccess);

        /* check return value of cusolverMpgels */
        assert(h_gelsInfo == 0);

        /* =================================== */
        /*      GATHER MATRICES B TO MASTER    */
        /* =================================== */

        /* Copy solution to h_X */
        cusolverStat = cusolverMpMatrixGatherD2H(cusolverMpHandle,
                                                 m_global_B,
                                                 n_global_B,
                                                 (void*)d_B,
                                                 1,
                                                 1,
                                                 descB,
                                                 0, /* master rank, destination */
                                                 (void*)h_X,
                                                 m_global_B);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync wait for data to arrive to host */
        cudaStat = cudaStreamSynchronize(localStream);
        assert(cudaStat == cudaSuccess);

        /* =========================================== */
        /*            CHECK RESIDUAL ON MASTER         */
        /* =========================================== */
        if (rank == 0)
        {
            /* print input matrices */
            if (opts.verbose)
            {
                print_host_matrix(m_global_B, n_global_B, h_X, m_global_B, "Output matrix X");
            }

            /* pointers to matrices A, B, and X */
            double* ptr_A = &h_A[(ia - 1) + (ja - 1) * m_global_A];
            double* ptr_B = &h_B[(ib - 1) + (jb - 1) * m_global_B];
            double* ptr_X = &h_X[(ib - 1) + (jb - 1) * m_global_B];

            /* compute error, B is overwritten with residual, B = B - AX */
            const double rel_err = compute_error(m, n, nrhs, ptr_A, m_global_A, ptr_X, m_global_B, ptr_B, m_global_B);

            /* relative error is around machine zero  */
            printf("|b - A*x|/(max(m,n)*|A|*|x|) = %E\n\n", rel_err);
        }


        /* =========================================== */
        /*        CLEAN UP HOST WORKSPACE ON MASTER    */
        /* =========================================== */
        if (rank == 0)
        {
            if (h_A)
            {
                free(h_A);
                h_A = NULL;
            }

            if (h_B)
            {
                free(h_B);
                h_B = NULL;
            }

            if (h_X)
            {
                free(h_X);
                h_X = NULL;
            }
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
            d_A = NULL;
        }

        if (d_B)
        {
            cudaStat = cudaFree(d_B);
            assert(cudaStat == cudaSuccess);
            d_B = NULL;
        }

        if (d_gelsWork)
        {
            cudaStat = cudaFree(d_gelsWork);
            assert(cudaStat == cudaSuccess);
            d_gelsWork = NULL;
        }

        if (d_gelsInfo)
        {
            cudaStat = cudaFree(d_gelsInfo);
            assert(cudaStat == cudaSuccess);
            d_gelsInfo = NULL;
        }

        /* =========================================== */
        /*         DEALLOCATE HOST WORKSPACE           */
        /* =========================================== */
        if (h_gelsWork)
        {
            free(h_gelsWork);
            h_gelsWork = NULL;
        }

        /* =========================================== */
        /*                      CLEANUP                */
        /* =========================================== */

        /* Destroy cusolverMp handle */
        cusolverStat = cusolverMpDestroy(cusolverMpHandle);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync before ncclCommDestroy */
        cudaStat = cudaStreamSynchronize(localStream);
        assert(cudaStat == cudaSuccess);

        /* destroy nccl communicator */
        ncclStat = ncclCommDestroy(comm);
        assert(ncclStat == ncclSuccess);

        /* destroy user stream */
        cudaStat = cudaStreamDestroy(localStream);
        assert(cudaStat == cudaSuccess);
    }

    /* MPI barrier before MPI_Finalize */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Finalize MPI environment */
    MPI_Finalize();

    if (mpiRank == 0)
    {
        printf("[SUCCEEDED]\n");
    }

    return 0;
}
