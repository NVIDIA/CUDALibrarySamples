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
! This example demonstrates basic usage of batched cuDSS APIs for solving
! two systems of linear algebraic equations with sparse matrices.
! Each of them can be written in the form:
!                             Ax = b,
! where:
!     A is the sparse input matrix,
!     b is the (dense) right-hand side vector (or a matrix),
!     x is the (dense) solution vector (or a matrix).
! Matches examples/cudss_batch.cpp.
module matrices
  use iso_c_binding
  use cudautils
  
  type csrMatrix
     integer :: n
     integer :: nnz
     integer, allocatable :: offsets(:)
     integer, allocatable :: columns(:)
     real(8), allocatable :: values(:)
     type(c_ptr) :: offsets_d
     type(c_ptr) :: columns_d
     type(c_ptr) :: values_d
  end type csrMatrix

  type denseMatrix
     integer :: nrows
     integer :: ncols
     real(8), allocatable :: values(:)
     type(c_ptr) :: values_d
  end type denseMatrix

  interface allocateMatrix
     module procedure :: allocateCSRMatrix, allocateDenseMatrix
  end interface allocateMatrix

  interface deallocateMatrix
     module procedure :: deallocateCSRMatrix, deallocateDenseMatrix
  end interface deallocateMatrix

contains

  subroutine allocateCSRMatrix(m)
    type(csrMatrix) :: m
    allocate(m%offsets(m%n+1), m%columns(m%nnz), m%values(m%nnz))
    m%offsets_d = c_null_ptr
    m%columns_d = c_null_ptr
    m%values_d = c_null_ptr
  end subroutine allocateCSRMatrix

  subroutine allocateDenseMatrix(m)
    type(denseMatrix) :: m
    allocate(m%values(m%nrows*m%ncols))
    m%values_d = c_null_ptr
  end subroutine allocateDenseMatrix

  subroutine deallocateCSRMatrix(m)
    type(csrMatrix) :: m
    call dev_free(m%offsets_d)
    call dev_free(m%columns_d)
    call dev_free(m%values_d)
    deallocate(m%offsets, m%columns, m%values)
  end subroutine deallocateCSRMatrix

  subroutine deallocateDenseMatrix(m)
    type(denseMatrix) :: m
    call dev_free(m%values_d)
    deallocate(m%values)
  end subroutine deallocateDenseMatrix

end module matrices


program main
  use iso_c_binding
  use cudautils
  use cudss
  use cudssutils
  use matrices
  use validation
  
  implicit none

  integer(8), parameter :: batchCount = 2
  integer, parameter :: n(batchCount) = [5, 6]
  integer, parameter :: nnz(batchCount) = [8, 11]
  integer, parameter :: nrhs(batchCount) = [1, 1]

  type(csrMatrix) :: As(batchCount)  
  type(denseMatrix) :: xs(batchCount), bs(batchCount)

  type(cudssHandle) :: h
  type(cudssMatrix) :: A, b, x
  type(cudssConfig) :: solverConfig
  type(cudssData) :: solverData
  type(cudssStatus) :: statusDSS
  type(cudssMatrixType) :: mtype = CUDSS_MTYPE_SPD
  type(cudssMatrixViewType) :: mview = CUDSS_MVIEW_UPPER
  type(cudssIndexBase) :: base = CUDSS_BASE_ONE

  type(cuda_devptr) :: &
       devptr_A_offsets(batchCount), &
       devptr_A_columns(batchCount), &
       devptr_A_values(batchCount), &
       devptr_b_values(batchCount), &
       devptr_x_values(batchCount)
  type(c_ptr) :: &
       devptr_A_offsets_d, &
       devptr_A_columns_d, &
       devptr_A_values_d, &
       devptr_b_values_d, &
       devptr_x_values_d

  integer, target :: b_nrows(batchCount), b_ncols(batchCount), b_ld(batchCount)
  integer, target :: x_nrows(batchCount), x_ncols(batchCount), x_ld(batchCount)
  integer, target :: a_nrows(batchCount), a_ncols(batchCount), a_nnz(batchCount)
  integer :: status
  integer(cuda_stream_kind) :: stream
  integer :: c, i
  real(8), parameter :: tol = 1.d-11
  
  print *, '----------------------------------------------------------'
  print *, 'cuDSS example: solving two real linear systems of size 5x5 and 6x6'
  print *, 'with symmetric positive-definite matrices'
  print *, '----------------------------------------------------------'

  ! property types

  print "('Maj/Min/Patch:', 3(1x, i0))", CUDSS_VERSION_MAJOR, CUDSS_VERSION_MINOR, CUDSS_VERSION_PATCH

  do i = 1, batchCount
     As(i)%n = n(i)
     As(i)%nnz = nnz(i)
     call allocateMatrix(As(i))

     xs(i)%nrows = n(i)
     xs(i)%ncols = nrhs(i)
     call allocateMatrix(xs(i))
        
     bs(i)%nrows = n(i)
     bs(i)%ncols = nrhs(i)
     call allocateMatrix(bs(i))

     if (i==1) then
        As(i)%offsets = [1,3,5,7,8,9]
        As(i)%columns = [1,3,2,3,3,5,4,5]
        As(i)%values = [4.0, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0]        
        bs(i)%values = [7.0, 12.0, 25.0, 4.0, 13.0]
     else if (i==2) then
        As(i)%offsets = [1,3,5,8,9,11,12]
        As(i)%columns = [1,6,2,5,3,5,6,4,5,6,6]
        As(i)%values = [3.0, 1.0, 2.0, 1.0, 6.0, 2.0, 2.0, 5.0, 7.0, 3.0, 8.0]
        bs(i)%values = [9.0, 9.0, 40.0, 20.0, 61.0, 70.0]
     end if

     call dev_malloc_i4(As(i)%offsets_d, As(i)%offsets)
     call dev_malloc_i4(As(i)%columns_d, As(i)%columns)
     call dev_malloc_r8(As(i)%values_d, As(i)%values)
     call dev_malloc_r8(bs(i)%values_d, bs(i)%values)
     xs(i)%values = 0.0d0
     call dev_malloc_r8(xs(i)%values_d, xs(i)%values)

     devptr_A_offsets(i) = c_devloc(As(i)%offsets_d)
     devptr_A_columns(i) = c_devloc(As(i)%columns_d)
     devptr_A_values(i) = c_devloc(As(i)%values_d)
     devptr_b_values(i) = c_devloc(bs(i)%values_d)
     devptr_x_values(i) = c_devloc(xs(i)%values_d)
  end do

  call dev_upload_ptr_array(devptr_A_offsets_d, devptr_A_offsets, int(batchCount))
  call dev_upload_ptr_array(devptr_A_columns_d, devptr_A_columns, int(batchCount))
  call dev_upload_ptr_array(devptr_A_values_d, devptr_A_values, int(batchCount))
  call dev_upload_ptr_array(devptr_b_values_d, devptr_b_values, int(batchCount))
  call dev_upload_ptr_array(devptr_x_values_d, devptr_x_values, int(batchCount))
  
  
  status = cudaStreamCreate(stream)
  if (status /= cudaSuccess) print *, 'cudaStreamCreate: ', status

  statusDSS = cudssCreate(h)
  if (statusDSS /= cudss_status_success) print *, 'cudssCreate: ', statusDSS

  statusDSS = cudssSetStream(h, stream)
  if (statusDSS /= cudss_status_success) print *, 'cudssSetStream: ', statusDSS

  statusDSS = cudssConfigCreate(solverConfig)
  if (statusDSS /= cudss_status_success) print *, 'cudssConfigCreate: ', statusDSS

  statusDSS = cudssDataCreate(h, solverData)
  if (statusDSS /= cudss_status_success) print *, 'cudssDataCreate: ', statusDSS

  b_nrows = n
  b_ncols = nrhs
  b_ld = b_nrows
  statusDSS = cudssMatrixCreateBatchDn(b, batchCount, b_nrows, b_ncols, b_ld, devptr_b_values_d, &
       CUDSS_R_32I, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixCreateBatchDn for bs:', statusDSS

  print *, 'Printing b ***'
  call printMatrix(b)

  x_nrows = n
  x_ncols = nrhs
  x_ld = x_nrows
  statusDSS = cudssMatrixCreateBatchDn(x, batchCount, x_nrows, x_ncols, x_ld, devptr_x_values_d, &
       CUDSS_R_32I, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixCreateBatchDn for x: ', statusDSS

  a_nrows = n
  a_ncols = n
  a_nnz = nnz
  statusDSS = cudssMatrixCreateBatchCsr(A, batchCount, &
       a_nrows, a_ncols, a_nnz, &
       devptr_A_offsets_d, devptr_A_columns_d, devptr_A_values_d, &
       CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, &
       mtype, mview, base)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixCreateBatchCsr for A: ', statusDSS

  print *, 'Printing A ***'
  call printMatrix(A)
  
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

  do i = 1, batchCount
    call dev_copy_d2h_r8(xs(i)%values, xs(i)%values_d, size(xs(i)%values))
  end do
  do i = 1, batchCount
    if (.not. validate_spd_upper_solution_r64(As(i)%n, As(i)%offsets, As(i)%columns, As(i)%values, &
         xs(i)%values, bs(i)%values, tol)) error stop 1
  end do

  print *, 'Printing x ***'
  call printMatrix(x)
  

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

  call dev_free(devptr_A_offsets_d)
  call dev_free(devptr_A_columns_d)
  call dev_free(devptr_A_values_d)
  call dev_free(devptr_b_values_d)
  call dev_free(devptr_x_values_d)

  do i = 1, batchCount
     call deallocateMatrix(As(i))
     call deallocateMatrix(xs(i))
     call deallocateMatrix(bs(i))
  end do

  print *, 'Example cudss_batch PASSED'
end program main
