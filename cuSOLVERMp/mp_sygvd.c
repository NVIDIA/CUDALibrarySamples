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
#include <math.h>

#include <omp.h>
#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

/* A is a diagonal weighted matrix */
static void generate_diagonal_dominant_symmetric_matrix(int64_t n, double* A, int64_t lda)
{
    time(NULL);

    /* set A[0:n, 0:n] = 0 */
    for (int64_t j = 0; j < n; j++)
    {
        double sum = 0;
        for (int64_t i = 0; i < n; i++)
        {
            if (i < j)
            {
                A[i + j * lda] = A[j + i * lda];
            }
            else
            {
                A[i + j * lda] = (double)(rand()) / RAND_MAX;
            }
            sum += A[i + j * lda];
        }

        A[j + j * lda] = 10 * sum;
    }
}

/* Print matrix */
static void print_host_matrix(int64_t m, int64_t n, double* A, int64_t lda, const char* msg)
{
    printf("print_host_matrix : %s\n", msg);

    for (int64_t i = 0; i < m; i++)
    {
        for (int64_t j = 0; j < n; j++)
        {
            printf("%.6e ", A[i + j * lda]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    /* Options used in this sample are marked */
    Options opts = { .m           = 256,
                     .mbA         = 32,
                     .nbA         = 32,
                     .mbB         = 1,
                     .nbB         = 1,
                     .mbZ         = 1,
                     .nbZ         = 1,
                     .ia          = 1,
                     .ja          = 1,
                     .ib          = 1,
                     .jb          = 1,
                     .iz          = 1,
                     .jz          = 1,
                     .p           = 2,
                     .q           = 1,
                     .grid_layout = 'C',
                     .verbose     = false };

    parse(&opts, argc, argv);
    validate(&opts);

    /* Initialize MPI library */
    MPI_Init(NULL, NULL);

    /* Define dimensions, block sizes and offsets of A and B matrices */
    const int64_t m = opts.m;

    /* Offsets of A and B matrices (base-1) */
    const int64_t ia = opts.ia;
    const int64_t ja = opts.ja;
    const int64_t ib = opts.ib;
    const int64_t jb = opts.jb;
    const int64_t iz = opts.iz;
    const int64_t jz = opts.jz;

    /* Tile sizes */
    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;
    const int64_t mbB = opts.mbB == 1 ? mbA : opts.mbB;
    const int64_t nbB = opts.nbB == 1 ? nbA : opts.nbB;
    const int64_t mbZ = opts.mbZ == 1 ? mbA : opts.mbZ;
    const int64_t nbZ = opts.nbZ == 1 ? nbA : opts.nbZ;

    /* Define grid of processors */
    const int                     numRowDevices = opts.p;
    const int                     numColDevices = opts.q;
    const cusolverMpGridMapping_t gridLayout =
            (opts.grid_layout == 'C' || opts.grid_layout == 'c' ? CUSOLVERMP_GRID_MAPPING_COL_MAJOR
                                                                : CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);

    /* Current implementation only allows RSRC,CSRC=(0,0) */
    const uint32_t rsrca = 0;
    const uint32_t csrca = 0;
    const uint32_t rsrcb = 0;
    const uint32_t csrcb = 0;
    const uint32_t rsrcz = 0;
    const uint32_t csrcz = 0;

    /* Get rank id and rank size of the comm. */
    int mpiCommSize, mpiRank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpiCommSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    const int verbose = opts.verbose;

    /* input argument used in this sample */
    if (mpiRank == 0)
    {
        print(&opts);

        fprintf(stdout, "input arguments used:\n");
        fprintf(stdout,
                "m=%d, "
                "mbA=%d, nbA=%d, ia=%d, ja=%d, "
                "mbB=%d, nbB=%d, ib=%d, jb=%d, "
                "mbZ=%d, nbZ=%d, iz=%d, jz=%d, "
                "p=%d, q=%d, grid_layout=%s verbose=%d\n",
                (int)(m),
                (int)mbA,
                (int)nbA,
                (int)ia,
                (int)ja,
                (int)mbB,
                (int)nbB,
                (int)ib,
                (int)jb,
                (int)mbZ,
                (int)nbZ,
                (int)iz,
                (int)jz,
                numRowDevices,
                numColDevices,
                gridLayout == CUSOLVERMP_GRID_MAPPING_COL_MAJOR ? "CUSOLVERMP_GRID_MAPPING_COL_MAJOR"
                                                                : "CUSOLVERMP_GRID_MAPPING_ROW_MAJOR",
                verbose);
    }

    /* error check of implementation restrictions */
    {
        /* check using the same square block sizes for all */
        const bool use_same_square_blocksize =
                ((mbA == nbA) && (mbA == mbB && mbA == nbB) && (mbA == mbZ && mbA == nbZ));
        if (!use_same_square_blocksize)
        {
            fprintf(stderr, "Error: blocksizes are not the same square\n");
            exit(1);
        }

        /* current implementation constraint using ia=1, ja=1 */
        const bool use_unit_global_offsets = ((ia == 1 && ja == 1) && (ib == 1 && jb == 1) && (iz == 1 && jz == 1));
        if (!use_unit_global_offsets)
        {
            fprintf(stderr, "Error: current implementation does not support non-unit offsets i.e., ia, ja, etc.\n");
            exit(1);
        }
    }

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

        /* Error codes */
        cusolverStatus_t cusolverStat = CUSOLVER_STATUS_SUCCESS;
        calError_t       calStat      = CAL_OK;
        cudaError_t      cudaStat     = cudaSuccess;

        cudaStat = cudaSetDevice(localRank);
        assert(cudaStat == cudaSuccess);

        /* Create communicator */
        cal_comm_t cal_comm = NULL;
#ifdef USE_CAL_MPI
        calStat = cal_comm_create_mpi(MPI_COMM_WORLD, rank, commSize, localRank, &cal_comm);
#else
        cal_comm_create_params_t params;
        params.allgather    = allgather;
        params.req_test     = request_test;
        params.req_free     = request_free;
        params.data         = (void*)(MPI_COMM_WORLD);
        params.rank         = rank;
        params.nranks       = commSize;
        params.local_device = localRank;

        calStat = cal_comm_create(params, &cal_comm);
#endif
        assert(calStat == CAL_OK);

        /* Create local stream */
        cudaStream_t localStream = NULL;

        cudaStat = cudaStreamCreate(&localStream);
        assert(cudaStat == cudaSuccess);

        /* Initialize cusolverMp library handle */
        cusolverMpHandle_t cusolverMpHandle = NULL;

        cusolverStat = cusolverMpCreate(&cusolverMpHandle, localRank, localStream);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* cusolverMp grid */
        cusolverMpGrid_t grid = NULL;

        /* cusolverMp matrix descriptors */
        cusolverMpMatrixDescriptor_t descA = NULL;
        cusolverMpMatrixDescriptor_t descB = NULL;
        cusolverMpMatrixDescriptor_t descZ = NULL;

        /* Distributed matrices */
        void* d_A = NULL;
        void* d_B = NULL;
        void* d_D = NULL;
        void* d_Z = NULL;

        /* Distributed device workspace */
        void* d_sygvdWork = NULL;

        /* Distributed host workspace */
        void* h_sygvdWork = NULL;

        /* size of workspace on device */
        size_t sygvdWorkspaceInBytesOnDevice = 0;

        /* size of workspace on host */
        size_t sygvdWorkspaceInBytesOnHost = 0;

        /* error codes from cusolverMp (device) */
        int* d_sygvdInfo = NULL;

        /* error codes from cusolverMp (host) */
        int h_sygvdInfo = 0;

        /* =========================================== */
        /*          Create inputs on master rank       */
        /* =========================================== */

        /* Single process per device */
        assert((numRowDevices * numColDevices) <= commSize);

        /* =========================================== */
        /*          Create inputs on master rank       */
        /* =========================================== */

        /* current implementation requires A, B, and Z matrices are aligned each other i.e., use the same ia, ja */
        const int64_t m_global = (ia - 1) + m;
        const int64_t n_global = (ja - 1) + m;

        double* h_A = NULL;
        double* h_B = NULL;
        double* h_D = NULL;
        double* h_Z = NULL;

        if (rank == 0)
        {
            /* allocate host workspace */
            h_A = (double*)malloc(m_global * n_global * sizeof(double));
            h_B = (double*)malloc(m_global * n_global * sizeof(double));
            h_D = (double*)malloc(m_global * sizeof(double));
            h_Z = (double*)malloc(m_global * n_global * sizeof(double));

            /* clean the allocated memory */
            memset(h_A, 0, m_global * n_global * sizeof(double));
            memset(h_B, 0, m_global * n_global * sizeof(double));
            memset(h_D, 0, m_global * sizeof(double));
            memset(h_Z, 0, m_global * n_global * sizeof(double));

            /* pointer offsets */
            double* AA = &h_A[(ia - 1) + (ja - 1) * m_global];
            double* BB = &h_B[(ib - 1) + (jb - 1) * m_global];

            /* set A[ia:ia+n, ja:ja+n] = diagonal dominant random lower triangular matrix */
            generate_diagonal_dominant_symmetric_matrix(m, AA, m_global);
            generate_diagonal_dominant_symmetric_matrix(m, BB, m_global);

            /* print input matrices */
            if (verbose)
            {
                print_host_matrix(m_global, n_global, h_A, m_global, "Input matrix A");
                print_host_matrix(m_global, n_global, h_B, m_global, "Input matrix B");
            }
        }

        /* compute the local dimensions device buffers */
        const int64_t m_local_A = cusolverMpNUMROC(m_global, mbA, myRowRank, rsrca, numRowDevices);
        const int64_t n_local_A = cusolverMpNUMROC(n_global, nbA, myColRank, csrca, numColDevices);

        const int64_t m_local_B = cusolverMpNUMROC(m_global, mbB, myRowRank, rsrcb, numRowDevices);
        const int64_t n_local_B = cusolverMpNUMROC(n_global, nbB, myColRank, csrcb, numColDevices);

        const int64_t m_local_Z = cusolverMpNUMROC(m_global, mbZ, myRowRank, rsrcz, numRowDevices);
        const int64_t n_local_Z = cusolverMpNUMROC(n_global, nbZ, myColRank, csrcz, numColDevices);

        /* Allocate local d_A, d_B, d_D, d_Z */
        cudaStat = cudaMalloc((void**)&d_A, m_local_A * n_local_A * sizeof(double));
        assert(cudaStat == cudaSuccess);

        cudaStat = cudaMalloc((void**)&d_B, m_local_B * n_local_B * sizeof(double));
        assert(cudaStat == cudaSuccess);

        cudaStat = cudaMalloc((void**)&d_D, m_global * sizeof(double));
        assert(cudaStat == cudaSuccess);

        cudaStat = cudaMalloc((void**)&d_Z, m_local_Z * n_local_Z * sizeof(double));
        assert(cudaStat == cudaSuccess);

        /* =========================================== */
        /*          CREATE GRID DESCRIPTORS            */
        /* =========================================== */
        cusolverStat =
                cusolverMpCreateDeviceGrid(cusolverMpHandle, &grid, cal_comm, numRowDevices, numColDevices, gridLayout);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*        CREATE MATRIX DESCRIPTORS            */
        /* =========================================== */
        cusolverStat = cusolverMpCreateMatrixDesc(
                &descA, grid, CUDA_R_64F, m_global, n_global, mbA, nbA, rsrca, csrca, m_local_A);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverMpCreateMatrixDesc(
                &descB, grid, CUDA_R_64F, m_global, n_global, mbB, nbB, rsrcb, csrcb, m_local_B);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverMpCreateMatrixDesc(
                &descZ, grid, CUDA_R_64F, m_global, n_global, mbZ, nbZ, rsrcz, csrcz, m_local_Z);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*             ALLOCATE D_INFO                 */
        /* =========================================== */

        cudaStat = cudaMalloc((void**)&d_sygvdInfo, sizeof(int));
        assert(cudaStat == cudaSuccess);

        /* =========================================== */
        /*                RESET D_INFO                 */
        /* =========================================== */

        cudaStat = cudaMemset(d_sygvdInfo, 0, sizeof(int));
        assert(cudaStat == cudaSuccess);

        /* =========================================== */
        /*      SCATTER MATRICES A AND B FROM MASTER   */
        /* =========================================== */
        cusolverStat = cusolverMpMatrixScatterH2D(cusolverMpHandle,
                                                  m_global,
                                                  n_global,
                                                  (void*)d_A,
                                                  ia,
                                                  ja,
                                                  descA,
                                                  0, /* root rank */
                                                  (void*)h_A,
                                                  m_global);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync wait for data to arrive to device */
        calStat = cal_stream_sync(cal_comm, localStream);
        assert(calStat == CAL_OK);

        cusolverStat = cusolverMpMatrixScatterH2D(cusolverMpHandle,
                                                  m_global,
                                                  n_global,
                                                  (void*)d_B,
                                                  ib,
                                                  jb,
                                                  descB,
                                                  0, /* root rank */
                                                  (void*)h_B,
                                                  m_global);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync wait for data to arrive to device */
        calStat = cal_stream_sync(cal_comm, localStream);
        assert(calStat == CAL_OK);

        /* =========================================== */
        /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
        /* =========================================== */

        cusolverStat = cusolverMpSygvd_bufferSize(cusolverMpHandle,
                                                  CUSOLVER_EIG_TYPE_1,
                                                  CUSOLVER_EIG_MODE_VECTOR,
                                                  CUBLAS_FILL_MODE_LOWER,
                                                  m,
                                                  ia,
                                                  ja,
                                                  descA,
                                                  ib,
                                                  jb,
                                                  descB,
                                                  iz,
                                                  jz,
                                                  descZ,
                                                  CUDA_R_64F,
                                                  &sygvdWorkspaceInBytesOnDevice,
                                                  &sygvdWorkspaceInBytesOnHost);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*         ALLOCATE Psygvd WORKSPACE            */
        /* =========================================== */

        cudaStat = cudaMalloc((void**)&d_sygvdWork, sygvdWorkspaceInBytesOnDevice);
        assert(cudaStat == cudaSuccess);

        h_sygvdWork = (void*)malloc(sygvdWorkspaceInBytesOnHost);
        assert(h_sygvdWork != NULL);

        /* sync wait for data to arrive to device */
        calStat = cal_stream_sync(cal_comm, localStream);
        assert(calStat == CAL_OK);

        /* =========================================== */
        /*                   CALL psygvd               */
        /* =========================================== */

        cusolverStat = cusolverMpSygvd(cusolverMpHandle,
                                       CUSOLVER_EIG_TYPE_1,
                                       CUSOLVER_EIG_MODE_VECTOR,
                                       CUBLAS_FILL_MODE_LOWER,
                                       m,
                                       d_A,
                                       ia,
                                       ja,
                                       descA,
                                       d_B,
                                       ib,
                                       jb,
                                       descB,
                                       d_D,
                                       d_Z,
                                       iz,
                                       jz,
                                       descZ,
                                       CUDA_R_64F,
                                       d_sygvdWork,
                                       sygvdWorkspaceInBytesOnDevice,
                                       h_sygvdWork,
                                       sygvdWorkspaceInBytesOnHost,
                                       d_sygvdInfo);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync after cusolverMpsygvd */
        calStat = cal_stream_sync(cal_comm, localStream);
        assert(calStat == CAL_OK);

        /* copy d_sygvdInfo to host */
        cudaStat = cudaMemcpyAsync(&h_sygvdInfo, d_sygvdInfo, sizeof(int), cudaMemcpyDeviceToHost, localStream);
        assert(cudaStat == cudaSuccess);

        /* wait for d_sygvdInfo copy */
        cudaStat = cudaStreamSynchronize(localStream);
        assert(cudaStat == cudaSuccess);

        /* check return value of cusolverMpsygvd */
        assert(h_sygvdInfo == 0);

        /* =================================== */
        /*      GATHER MATRICES Z TO MASTER    */
        /* =================================== */

        /* copy eigen vectors to h_Z */
        cusolverStat = cusolverMpMatrixGatherD2H(cusolverMpHandle,
                                                 m_global,
                                                 n_global,
                                                 (void*)d_Z,
                                                 iz,
                                                 jz,
                                                 descZ,
                                                 0, /* master rank, destination */
                                                 (void*)h_Z,
                                                 m_global);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        if (rank == 0)
        {
            /* copy d_D to host */
            cudaStat = cudaMemcpyAsync(h_D, d_D, m_global * sizeof(double), cudaMemcpyDeviceToHost, localStream);
            assert(cudaStat == cudaSuccess);
        }

        /* sync wait for data to arrive to host */
        calStat = cal_stream_sync(cal_comm, localStream);
        assert(calStat == CAL_OK);

        if (rank == 0)
        {
            if (verbose)
            {
                print_host_matrix(m_global, 1, h_D, m_global, "Output eigen values");
                print_host_matrix(m_global, n_global, h_Z, m_global, "Output matrix eigen vectors");
            }
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

            if (h_D)
            {
                free(h_D);
                h_D = NULL;
            }

            if (h_Z)
            {
                free(h_Z);
                h_Z = NULL;
            }

            if (h_sygvdWork)
            {
                free(h_sygvdWork);
                h_sygvdWork = NULL;
            }
        }

        /* =========================================== */
        /*           DESTROY MATRIX DESCRIPTORS        */
        /* =========================================== */

        cusolverStat = cusolverMpDestroyMatrixDesc(descA);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverMpDestroyMatrixDesc(descB);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverMpDestroyMatrixDesc(descZ);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*             DESTROY MATRIX GRIDS            */
        /* =========================================== */

        cusolverStat = cusolverMpDestroyGrid(grid);
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

        if (d_D)
        {
            cudaStat = cudaFree(d_D);
            assert(cudaStat == cudaSuccess);
            d_D = NULL;
        }

        if (d_Z)
        {
            cudaStat = cudaFree(d_Z);
            assert(cudaStat == cudaSuccess);
            d_Z = NULL;
        }

        if (d_sygvdWork)
        {
            cudaStat = cudaFree(d_sygvdWork);
            assert(cudaStat == cudaSuccess);
            d_sygvdWork = NULL;
        }

        if (d_sygvdInfo)
        {
            cudaStat = cudaFree(d_sygvdInfo);
            assert(cudaStat == cudaSuccess);
            d_sygvdInfo = NULL;
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
}
