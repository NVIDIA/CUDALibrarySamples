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
  ! This example demonstrates usage of the host execution mode of cuDSS
  ! for solving a system of linear algebraic equations with a sparse matrix:
  !                             Ax = b,
  ! where:
  !     A is the sparse input matrix,
  !     b is the (dense) right-hand side vector (or a matrix),
  !     x is the (dense) solution vector (or a matrix).
  ! Matches examples/cudss_hybrid_execute_mode.cpp.
  use iso_c_binding
  use cudautils
  use cudss
  use validation
  
  implicit none

  ! 5x5 matrix
  integer, parameter :: n=5
  integer, parameter :: nnz=8
  integer, parameter :: nrhs=1

  integer, parameter :: nrows = n, ncols = n
  integer, parameter :: ldb = nrows, ldx = ncols
  
  integer, allocatable, target :: csr_offsets(:), csr_columns(:)
  real(8), allocatable, target :: csr_values(:), x_values(:), b_values(:)

  type(cudssHandle) :: h
  type(cudssConfig) :: solverConfig
  type(cudssData) :: solverData
  type(cudssStatus) :: statusDSS
  type(cudssMatrix) :: b, x

  type(cudssMatrix) :: A
  type(cudssMatrixType) :: mtype = CUDSS_MTYPE_SPD
  type(cudssMatrixViewType) :: mview = CUDSS_MVIEW_UPPER
  type(cudssIndexBase) :: base = CUDSS_BASE_ONE

  integer :: status

  integer(cuda_stream_kind) :: stream
  integer :: c
  real(8), parameter :: tol = 1.d-11
  
  print *, '---------------------------------------------------------'
  print *, 'cuDSS example: solving a real linear 5x5 system'
  print *, 'with a symmetric positive-definite matrix'
  print *, '---------------------------------------------------------'

  ! property types

  print "('Maj/Min/Patch:', 3(1x, i0))", CUDSS_VERSION_MAJOR, CUDSS_VERSION_MINOR, CUDSS_VERSION_PATCH
  
  allocate(&
       csr_offsets(n+1), &
       csr_columns(nnz), &
       csr_values(nnz), &
       x_values(nrhs*n), &
       b_values(nrhs*n))

  ! Symmetric matrix with upper triangular elements stored
  ! full matrix problem is:
  ! 
  !  4 0 1 0 0       7
  !  0 3 2 0 0      12
  !  1 0 5 0 1  X = 25
  !  0 0 0 1 0       4
  !  0 0 1 0 2      13

  csr_offsets = [1,3,5,7,8,9]
  csr_columns = [1,3,2,3,3,5,4,5]
  csr_values = [4.0, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0]
  b_values = [7.0, 12.0, 25.0, 4.0, 13.0]
  x_values = 0.0d0

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

  ! Note: (Optional) Since Hybrid Execute Mode is used, the righthand side (b)
  !       and the solution (x) can be passed as host memory.
  statusDSS = cudssMatrixCreateDn(b, int(nrows,8), int(nrhs,8), int(ldb,8), &
       c_loc(b_values(1)), CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixCreate for b:', statusDSS

  statusDSS = cudssMatrixCreateDn(x, int(ncols,8), int(nrhs,8), int(ldx,8), &
       c_loc(x_values(1)), CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixCreateDn for x: ', statusDSS

  ! Note: (Optional) Since Hybrid Execute Mode is used, the input matrix
  !       (csr_offsets, csr_columns and csr_values) can be passed as host memory.
  statusDSS = cudssMatrixCreateCsr(A, &
       int(nrows,8), int(ncols,8), int(nnz,8), &
       c_loc(csr_offsets(1)), c_loc(csr_columns(1)), c_loc(csr_values(1)), &
       CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, &
       mtype, mview, base)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixCreateCsr for A: ', statusDSS
  
  ! enable hybrid execute mode allowing cuDSS to use both CPU and GPU
  
  block
    integer(4) :: hybridExecuteMode

    hybridExecuteMode = 1
    statusDSS = cudssConfigSet(solverConfig, CUDSS_CONFIG_HYBRID_EXECUTE_MODE, hybridExecuteMode, c_sizeof(hybridExecuteMode))
    if (statusDSS /= cudss_status_success) print *, 'cudssConfigSet enabling hybrid execute mode: ', statusDSS
  end block

  statusDSS = cudssExecute(h, CUDSS_PHASE_ANALYSIS, &
       solverConfig, solverData, A, x, b)
  if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase analysis: ', statusDSS

  statusDSS = cudssExecute(h, CUDSS_PHASE_FACTORIZATION, &
       solverConfig, solverData, A, x, b)
  if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase factorization: ', statusDSS

  statusDSS = cudssExecute(h, CUDSS_PHASE_SOLVE, &
       solverConfig, solverData, A, x, b)
  if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase solve: ', statusDSS

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

  status = cudaStreamSynchronize(stream)
  if (status /= cudaSuccess) print *, 'cudaStreamSynchronize: ', status

  print *, 'x: '
  do c = 1, ncols
     print *, c, x_values(c)
  end do

  if (.not. validate_spd_upper_solution_r64(n, csr_offsets, csr_columns, csr_values, x_values, b_values, tol)) &
       error stop 1

  status = cudaStreamDestroy(stream)
  if (status /= cudaSuccess) print *, 'cudaStreamDestroy: ', status

  deallocate(csr_offsets, csr_columns, csr_values, x_values, b_values)

  print *, 'Example cudss_hybrid_execute_mode PASSED'
end program main
  
