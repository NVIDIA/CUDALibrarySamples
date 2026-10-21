! SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: Apache-2.0
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
! http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.
!
program main
  ! Fortran MPI variant of multi-GPU / distributed cuDSS usage: assigns devices
  ! per rank and wires MPI communicators via cudssSetCommLayer (OpenMPI comm layer).
  ! Conceptually extends examples/cudss_mg_mode.cpp (multiple devices) with
  ! multi-node communicator support; no separate C++ twin in examples/.
  use iso_c_binding
  use cudautils
  use cudss
  use cudss_mpi, only: cudssDataSetComm
  use cudssutils
  use validation
  use mpi

  implicit none

  ! 5x5 matrix
  integer, parameter :: n=5
  integer, parameter :: nnz=8
  integer, parameter :: nrhs=1

  integer, parameter :: nrows = n, ncols = n
  integer, parameter :: ldb = nrows, ldx = ncols
  
  integer, allocatable :: csr_offsets(:), csr_columns(:)
  real(8), allocatable :: csr_values(:), x_values(:), b_values(:)
  
  type(c_ptr) :: csr_offsets_d, csr_columns_d, csr_values_d, x_values_d, b_values_d

  type(cudssHandle) :: h
  type(cudssConfig) :: solverConfig
  type(cudssData) :: solverData
  type(cudssStatus) :: statusDSS
  type(cudssMatrix) :: b, x

  type(cudssMatrix) :: A
  type(cudssMatrixType) :: mtype = CUDSS_MTYPE_SPD
  type(cudssMatrixViewType) :: mview = CUDSS_MVIEW_UPPER
  type(cudssIndexBase) :: base = CUDSS_BASE_ONE

  integer :: status, i
  integer(cuda_stream_kind) :: stream
  integer :: c
  real(8), parameter :: tol = 1.d-11

  integer :: myrank, nprocs, ierr, namelength
  integer :: devID
  integer, target :: mpi_comm_fint

  integer :: numArgs, argLen
  character(len=MPI_MAX_PROCESSOR_NAME) :: hostname
  character (len=:), allocatable :: commLayerLib

  csr_offsets_d = C_NULL_PTR
  csr_columns_d = C_NULL_PTR
  csr_values_d = C_NULL_PTR
  x_values_d = C_NULL_PTR
  b_values_d = C_NULL_PTR

  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD, myrank, ierr)
  call mpi_comm_size(MPI_COMM_WORLD, nprocs, ierr)

  if (myrank == 0) then
     print "('cuDSS mgmn example using ', i0, ' processes')", nprocs
  end if

  ! assign GPUs to MPI ranks

  block
    integer :: local_comm
    devID = 0
    call mpi_comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, &
         MPI_INFO_NULL, local_comm, ierr)
    call mpi_comm_rank(local_comm, devID, ierr)
    status = cudaSetDevice(devID)
    if (status /= cudaSuccess) print *, 'cudaSetDevice: ', status
  end block

  call MPI_GET_PROCESSOR_NAME(hostname, namelength, ierr)
  do i=0, nprocs-1
     call MPI_BARRIER(MPI_COMM_WORLD, ierr)
     if (i == myrank) print "('[',i0,'] host: ', a, '  device: ', i0)", myrank, trim(hostname), devID
  end do
  call MPI_BARRIER(MPI_COMM_WORLD, ierr)

  ! get comm layer library from command-line arguments or
  ! from CUDSS_COMM_LIB environment variable
  ! (same handling as cudss_multithreaded_mode.cuf for CUDSS_THREADING_LIB).

  numArgs = command_argument_count()
  if (numArgs > 0) then
     call get_command_argument(1, length = argLen)
     allocate(character(len=argLen) :: commLayerLib)
     call get_command_argument(1, commLayerLib)
     print *, 'Comm layer library: ', commLayerLib
     print *
  else
     commLayerLib = C_NULL_CHAR
     print *, 'Since the comm layer library was not provided as a command line argument,'
     print *, 'it will be taken from the CUDSS_COMM_LIB environment variable'
     print *
  end if

  ! property types

  if (myrank == 0) print "('Maj/Min/Patch:', 3(1x, i0))", CUDSS_VERSION_MAJOR, CUDSS_VERSION_MINOR, CUDSS_VERSION_PATCH

  if (myrank == 0) then
     allocate(&
          csr_offsets(n+1), &
          csr_columns(nnz), &
          csr_values(nnz), &
          x_values(nrhs*n), &
          b_values(nrhs*n))

     ! Symmetric matrix with upper triangular elements stored
     !
     ! Full matrix problem is:
     ! 
     !  4 0 1 0 0       7
     !  0 3 2 0 0      12
     !  1 2 5 0 1  X = 25
     !  0 0 0 1 0       4
     !  0 0 1 0 2      13
     
     csr_offsets = [1,3,5,7,8,9]
     csr_columns = [1,3,2,3,3,5,4,5]
     csr_values = [4.0, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0]
     b_values = [7.0, 12.0, 25.0, 4.0, 13.0]
     x_values = 0.0d0
     
     call dev_malloc_i4(csr_offsets_d, csr_offsets)
     call dev_malloc_i4(csr_columns_d, csr_columns)
     call dev_malloc_r8(csr_values_d, csr_values)
     call dev_malloc_r8(b_values_d, b_values)
     call dev_malloc_r8(x_values_d, x_values)
  end if

  status = cudaStreamCreate(stream)
  if (status /= cudaSuccess) print *, 'cudaStreamCreate: ', status

  statusDSS = cudssCreate(h)
  if (statusDSS /= cudss_status_success) print *, 'cudssCreate: ', statusDSS

  statusDSS = cudssSetCommLayer(h, commLayerLib)
  if (statusDSS /= cudss_status_success) then
     print *, 'cudssSetCommLayer: ', statusDSS
     print *, 'The full path of the comm library must be specified as a'
     print *, 'command-line argument or as the CUDSS_COMM_LIB environment variable.'
     call MPI_FINALIZE(ierr)
     error stop 1
  end if

  statusDSS = cudssSetStream(h, stream)
  if (statusDSS /= cudss_status_success) print *, 'cudssSetStream: ', statusDSS

  statusDSS = cudssConfigCreate(solverConfig)
  if (statusDSS /= cudss_status_success) print *, 'cudssConfigCreate: ', statusDSS

  statusDSS = cudssDataCreate(h, solverData)
  if (statusDSS /= cudss_status_success) print *, 'cudssDataCreate: ', statusDSS

  statusDSS = cudssMatrixCreateDn(b, int(nrows,8), int(nrhs,8), int(ldb,8), &
       b_values_d, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixCreate for b:', statusDSS

  if (myrank == 0) then
     print *, 'Printing b ***'
     call printMatrix(b)
  end if

  statusDSS = cudssMatrixCreateDn(x, int(ncols,8), int(nrhs,8), int(ldx,8), &
       x_values_d, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixCreateDn for x: ', statusDSS
  
  statusDSS = cudssMatrixCreateCsr(A, &
       int(nrows,8), int(ncols,8), int(nnz,8), &
       csr_offsets_d, csr_columns_d, csr_values_d, &
       CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, &
       mtype, mview, base)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixCreateCsr for A: ', statusDSS
  
  if (myrank == 0) then
     print *, 'Printing A ***'
     call printMatrix(A)
  end if

  ! set the communicator for MGMN mode
  mpi_comm_fint = MPI_COMM_WORLD
  statusDSS = cudssDataSetComm(h, solverData, CUDSS_DATA_COMM_HOST, mpi_comm_fint, c_sizeof(mpi_comm_fint))
  if (statusDSS /= cudss_status_success) print *, 'cudssDataSetComm: ', statusDSS

  statusDSS = cudssDataSetComm(h, solverData, CUDSS_DATA_COMM_DEVICE, mpi_comm_fint, c_sizeof(mpi_comm_fint))
  if (statusDSS /= cudss_status_success) print *, 'cudssDataSetComm: ', statusDSS
  
  statusDSS = cudssExecute(h, CUDSS_PHASE_ANALYSIS, &
       solverConfig, solverData, A, x, b)
  if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase analysis: ', statusDSS

  statusDSS = cudssExecute(h, CUDSS_PHASE_FACTORIZATION, &
       solverConfig, solverData, A, x, b)
  if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase factorization: ', statusDSS

  statusDSS = cudssExecute(h, CUDSS_PHASE_SOLVE, &
       solverConfig, solverData, A, x, b)
  if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase solve: ', statusDSS

  status = cudaStreamSynchronize(stream)
  if (status /= cudaSuccess) print *, 'cudaStreamSynchronize: ', status

  if (myrank == 0) call dev_copy_d2h_r8(x_values, x_values_d, size(x_values))

  if (myrank==0) then
     print *, 'x: '
     do c = 1, ncols
        print *, c, x_values(c)
     end do

     if (.not. validate_spd_upper_solution_r64(n, csr_offsets, csr_columns, csr_values, x_values, b_values, tol)) &
          error stop 1
  end if

  ! cleanup

  statusDSS = cudssMatrixDestroy(A)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixDestroy(A): ', statusDSS
  statusDSS = cudssMatrixDestroy(b)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixDestroy(b): ', statusDSS
  statusDSS = cudssMatrixDestroy(x)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixDestroy(x): ', statusDSS
  statusDSS = cudssDataDestroy(h, solverData)
  if (statusDSS /= cudss_status_success) print *, 'cudssDataDestroy(h,solverData): ', statusDSS  
  statusDSS = cudssConfigDestroy(solverConfig)
  if (statusDSS /= cudss_status_success) print *, 'cudssConfigDestroy(solverConfig): ', statusDSS
  statusDSS = cudssDestroy(h)
  if (statusDSS /= cudss_status_success) print *, 'cudssDestroy: ', statusDSS
  status = cudaStreamDestroy(stream)
  if (status /= cudaSuccess) print *, 'cudaStreamDestroy: ', status

  if (myrank==0) then
     call dev_free(csr_offsets_d)
     call dev_free(csr_columns_d)
     call dev_free(csr_values_d)
     call dev_free(x_values_d)
     call dev_free(b_values_d)
     deallocate(csr_offsets, csr_columns, csr_values, x_values, b_values)
  end if

  if (myrank == 0) print *, 'Example cudss_mgmn_mode PASSED'

  call MPI_FINALIZE(ierr)
end program main
  
