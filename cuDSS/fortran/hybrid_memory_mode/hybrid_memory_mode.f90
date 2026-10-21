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
  ! This example demonstrates usage of hybrid memory mode in cuDSS for solving
  ! a system of linear algebraic equations with a sparse matrix:
  !                             Ax = b,
  ! where:
  !     A is the sparse input matrix,
  !     b is the (dense) right-hand side vector (or a matrix),
  !     x is the (dense) solution vector (or a matrix).
  ! Note: hybrid memory mode is intended to be used for solving systems with
  ! matrices whose factors do not fit into the device memory (the mode
  ! uses host memory to store the full factors + a temporary device buffer).
  ! As it has a nontrivial overhead relative to the default (device) memory mode,
  ! it is recommended to enable this feature only for solving large systems.
  ! Matches examples/cudss_hybrid_memory_mode.cpp.
  use iso_c_binding
  use cudautils
  use cudss
  use cudssutils
  use validation
  
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

  integer :: status

  integer(cuda_stream_kind) :: stream
  integer :: c
  real(8), parameter :: tol = 1.d-11
  
  print *, '---------------------------------------------------------'
  print *, 'cuDSS example: solving a real linear 5x5 system'
  print *, 'with a symmetric positive-definite matrix using'
  print *, 'the hybrid memory mode.'
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

  call dev_malloc_i4(csr_offsets_d, csr_offsets)
  call dev_malloc_i4(csr_columns_d, csr_columns)
  call dev_malloc_r8(csr_values_d, csr_values)
  call dev_malloc_r8(b_values_d, b_values)
  call dev_malloc_r8(x_values_d, x_values)
  
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
  call printMatrix(b)

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
  call printMatrix(A)
  
  ! enable hybrid mode where factors are stored in host memory
  ! N.B. Must be set before the analysis phase
  
  block
    integer(4) :: hybridMode = 1

    statusDSS = cudssConfigSet(solverConfig, CUDSS_CONFIG_HYBRID_MEMORY_MODE, hybridMode, c_sizeof(hybridMode))
    if (statusDSS /= cudss_status_success) print *, 'cudssConfigSet enabling hybrid mode: ', statusDSS
  end block

  ! The x and b shapes passed to analysis determine the hybrid memory minimum.
  statusDSS = cudssExecute(h, CUDSS_PHASE_ANALYSIS, &
       solverConfig, solverData, A, x, b)
  if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase analysis: ', statusDSS

  ! query the minimal amount of device memory sufficient for the hybrid memory mode
  block
    integer(c_size_t) :: sizeWritten
    integer(c_int64_t) :: deviceMemoryMin, deviceMemoryLimit

    statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN, &
         deviceMemoryMin, c_sizeof(deviceMemoryMin), sizeWritten)
    if (statusDSS /= cudss_status_success) print *, 'cudssDataGet: ', statusDSS
    print "('minimum device memory required for hybrid mode (bytes): ', i0)", deviceMemoryMin

    ! The exact minimum has no headroom for input or configuration changes.
    deviceMemoryLimit = deviceMemoryMin
    statusDSS = cudssConfigSet(solverConfig, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT, &
         deviceMemoryLimit, c_sizeof(deviceMemoryLimit))
    if (statusDSS /= cudss_status_success) print *, 'cudssConfigSet: ', statusDSS
    print "('set the upper limit on device memory for hybrid mode to (bytes): ', i0)", deviceMemoryLimit
  end block

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

  call dev_copy_d2h_r8(x_values, x_values_d, size(x_values))

  print *, 'x: '
  do c = 1, ncols
     print *, c, x_values(c)
  end do

  if (.not. validate_spd_upper_solution_r64(n, csr_offsets, csr_columns, csr_values, x_values, b_values, tol)) &
       error stop 1

  status = cudaStreamDestroy(stream)
  if (status /= cudaSuccess) print *, 'cudaStreamDestroy: ', status
  
  call dev_free(csr_offsets_d)
  call dev_free(csr_columns_d)
  call dev_free(csr_values_d)
  call dev_free(x_values_d)
  call dev_free(b_values_d)
  deallocate(csr_offsets, csr_columns, csr_values, x_values, b_values)

  print *, 'Example cudss_hybrid_memory_mode PASSED'
end program main
  
