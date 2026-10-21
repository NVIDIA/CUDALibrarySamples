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
  ! This example demonstrates basic usage of cuDSS APIs for solving
  ! two systems of linear algebraic equations with the same sparsity pattern:
  !                             A1 x1 = b1,
  !                             A2 x2 = b2
  ! where:
  !     A1, A2 are the sparse input matrices (with the same non-zero pattern),
  !     b1, b2 are dense right-hand side vectors (or matrices),
  !     x1, x2 are dense solution vectors (or matrices).
  ! Matches examples/cudss_uniform_batch.cpp.
  use iso_c_binding
  use cudautils
  use cudss
  use cudssutils
  use validation
  
  implicit none

  integer, parameter :: uniformBatchSize = 2
  integer, parameter :: n = 5
  integer, parameter :: nnz = 8
  integer, parameter :: nrhs = 1
  integer, parameter :: nrows = n, ncols = n, ldx = ncols, ldb = nrows

  type(cudssHandle) :: h
  type(cudssMatrix) :: A, b, x
  type(cudssConfig) :: solverConfig
  type(cudssData) :: solverData
  type(cudssStatus) :: statusDSS
  type(cudssMatrixType) :: mtype = CUDSS_MTYPE_SPD
  type(cudssMatrixViewType) :: mview = CUDSS_MVIEW_UPPER
  type(cudssIndexBase) :: base = CUDSS_BASE_ONE

  integer, allocatable :: csr_offsets(:), csr_columns(:)
  real(8), allocatable :: csr_values(:,:), x_values(:,:), b_values(:,:)

  type(c_ptr) :: csr_offsets_d, csr_columns_d, csr_values_d, x_values_d, b_values_d

  integer :: status
  integer(cuda_stream_kind) :: stream
  integer :: c, i
  real(8), parameter :: tol = 1.d-11
  
  ! property types

  print "('Maj/Min/Patch:', 3(1x, i0))", CUDSS_VERSION_MAJOR, CUDSS_VERSION_MINOR, CUDSS_VERSION_PATCH

  allocate(&
       csr_offsets(n+1), &
       csr_columns(nnz), &
       csr_values(nnz,2), &
       x_values(nrhs*n,2), &
       b_values(nrhs*n,2))

  
  csr_offsets(:) = [1,3,5,7,8,9]
  csr_columns(:) = [1,3,2,3,3,5,4,5]

  csr_values(:,1) = [4.0, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0]        
  csr_values(:,2) = [2.0, 1.0, 3.0, 1.0, 6.0, 2.0, 4.0, 8.0]

  b_values(:,1) = [7.0,  12.0, 25.0, 4.0, 13.0]
  b_values(:,2) = [13.0, 15.0, 29.0, 8.0, 14.0]
  x_values = 0.0d0

  call dev_malloc_i4(csr_offsets_d, csr_offsets)
  call dev_malloc_i4(csr_columns_d, csr_columns)
  block
    real(8), target :: flat(size(csr_values))
    flat = reshape(csr_values, [size(csr_values)])
    call dev_malloc_r8(csr_values_d, flat)
  end block
  block
    real(8), target :: flat(size(b_values))
    flat = reshape(b_values, [size(b_values)])
    call dev_malloc_r8(b_values_d, flat)
  end block
  block
    real(8), target :: flat(size(x_values))
    flat = reshape(x_values, [size(x_values)])
    call dev_malloc_r8(x_values_d, flat)
  end block

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

  statusDSS = cudssMatrixCreateDn(b, int(nrows,8), int(nrhs,8), int(ldb,8), &
       b_values_d, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixCreate for b:', statusDSS
  
  print *, 'Printing b ***'
  call printMatrix(b, solverConfig)

  statusDSS = cudssMatrixCreateDn(x, int(ncols,8), int(nrhs,8), int(ldx,8), &
       x_values_d, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixCreateDn for x: ', statusDSS

  statusDSS = cudssMatrixCreateCsr(A, &
       int(nrows,8), int(ncols,8), int(nnz,8), &
       csr_offsets_d, csr_columns_d, csr_values_d, &
       CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, &
       mtype, mview, base)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixCreateCsr for A: ', statusDSS


  print *, 'Printing A ***'
  call printMatrix(A, solverConfig)

  ! cuDSS will factorize and solve all matrices in the batch simultaneously
  statusDSS = cudssConfigSet(solverConfig, CUDSS_CONFIG_UBATCH_SIZE, &
       uniformBatchSize, c_sizeof(uniformBatchSize))
  if (statusDSS /= cudss_status_success) print *, 'cudssConfigSet uniform batch size: ', statusDSS

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

  block
    real(8), target :: flat(size(x_values))
    call dev_copy_d2h_r8(flat, x_values_d, size(flat))
    x_values = reshape(flat, shape(x_values))
  end block
  do i = 1, uniformBatchSize
    if (.not. validate_spd_upper_solution_r64(n, csr_offsets, csr_columns, csr_values(:,i), &
         x_values(:,i), b_values(:,i), tol)) error stop 1
  end do

  print *, 'Printing x ***'
  call printMatrix(x, solverConfig)

  ! Optionally ubatch_index can be set to factorize one specific matrix of the
  ! batch or use a specific factorized matrix on solve. That allows to keep
  ! in memory multiple factorized matrices.
  block
    real(8), pointer :: csr_values_p(:), b_values_p(:)
    real(8) :: flat(n*nrhs)
    integer :: ubatch_index
    ubatch_index = 1

    call c_f_pointer(csr_values_d, csr_values_p, [nnz*uniformBatchSize])
    call c_f_pointer(b_values_d, b_values_p, [n*nrhs*uniformBatchSize])

    statusDSS = cudssConfigSet(solverConfig, CUDSS_CONFIG_UBATCH_INDEX, &
         ubatch_index, c_sizeof(ubatch_index))
    if (statusDSS /= cudss_status_success) print *, 'cudssConfigSet ubatch index: ', statusDSS

    statusDSS = cudssMatrixSetValues(A, c_loc(csr_values_p(ubatch_index*nnz + 1)))
    if (statusDSS /= cudss_status_success) print *, 'cudssMatrixSetValues for A: ', statusDSS
    statusDSS = cudssMatrixSetValues(b, c_loc(b_values_p(ubatch_index*n*nrhs + 1)))
    if (statusDSS /= cudss_status_success) print *, 'cudssMatrixSetValues for b: ', statusDSS

    statusDSS = cudssExecute(h, CUDSS_PHASE_FACTORIZATION, &
         solverConfig, solverData, A, x, b)
    if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase factorization: ', statusDSS

    statusDSS = cudssExecute(h, CUDSS_PHASE_SOLVE, &
         solverConfig, solverData, A, x, b)
    if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase solve: ', statusDSS

    status = cudaStreamSynchronize(stream)
    if (status /= cudaSuccess) print *, 'cudaStreamSynchronize: ', status

    call dev_copy_d2h_r8(flat, x_values_d, size(flat))
    if (.not. validate_spd_upper_solution_r64(n, csr_offsets, csr_columns, csr_values(:,ubatch_index+1), &
         flat, b_values(:,ubatch_index+1), tol)) error stop 1
  end block

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

  call dev_free(csr_offsets_d)
  call dev_free(csr_columns_d)
  call dev_free(csr_values_d)
  call dev_free(x_values_d)
  call dev_free(b_values_d)
  deallocate(csr_offsets, csr_columns, csr_values, x_values, b_values)

  print *, 'Example cudss_uniform_batch PASSED'
end program main
  
