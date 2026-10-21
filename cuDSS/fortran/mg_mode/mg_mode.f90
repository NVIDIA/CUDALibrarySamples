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
  ! a system of linear algebraic equations with a sparse matrix:
  !                             Ax = b,
  ! using multiple devices, where:
  !     A is the sparse input matrix,
  !     b is the (dense) right-hand side vector (or a matrix),
  !     x is the (dense) solution vector (or a matrix).
  ! Matches examples/cudss_mg_mode.cpp.
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

  integer :: deviceCount
  integer, allocatable, target :: deviceIndices(:)
  integer(cuda_stream_kind), allocatable, target :: deviceStreams(:)
  integer :: currentDevice

  integer :: status
  integer :: c, i
  real(8), parameter :: tol = 1.d-11
  
  ! property types

  print "('Maj/Min/Patch:', 3(1x, i0))", CUDSS_VERSION_MAJOR, CUDSS_VERSION_MINOR, CUDSS_VERSION_PATCH

  ! device stuff
  status = cudaGetDeviceCount(deviceCount)
  if (status /= cudaSuccess) print *, 'cudaGetDeviceCount: ', status 

  allocate(deviceIndices(deviceCount))
  do i = 1, deviceCount
     deviceIndices(i) = i-1
  enddo

  print "('running on ', i0, ' device(s)')", deviceCount
  
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
  
  ! (optional) Creating one custom stream per device, to be set on the library handle
  status = cudaGetDevice(currentDevice)
  if (status /= cudaSuccess) print *, 'cudaGetDevice: ', status

  allocate(deviceStreams(deviceCount))
  do i = 1, deviceCount
     status = cudaSetDevice(deviceIndices(i))
     if (status /= cudaSuccess) print *, 'cudaSetDevice: ', status
     status = cudaStreamCreate(deviceStreams(i))
     if (status /= cudaSuccess) print *, 'cudaStreamCreate: ', status
  enddo
  status = cudaSetDevice(currentDevice)
  if (status /= cudaSuccess) print *, 'cudaSetDevice restore: ', status

  statusDSS = cudssCreateMg(h, deviceCount, deviceIndices)
  if (statusDSS /= cudss_status_success) print *, 'cudssCreateMg: ', statusDSS

  statusDSS = cudssSetMgStreams(h, c_loc(deviceStreams(1)), deviceCount)
  if (statusDSS /= cudss_status_success) print *, 'cudssSetMgStreams: ', statusDSS

  statusDSS = cudssConfigCreate(solverConfig)
  if (statusDSS /= cudss_status_success) print *, 'cudssConfigCreate: ', statusDSS


  statusDSS = cudssConfigSet(solverConfig, CUDSS_CONFIG_DEVICE_COUNT, &
       deviceCount, c_sizeof(deviceCount))
  if (statusDSS /= cudss_status_success) print *, 'cudssConfigSet deviceCount: ', statusDSS

  statusDSS = cudssConfigSet(solverConfig, CUDSS_CONFIG_DEVICE_INDICES, &
       deviceIndices, deviceCount * c_sizeof(deviceCount))
  if (statusDSS /= cudss_status_success) print *, 'cudssConfigSet deviceIndices: ', statusDSS

  
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
  
  statusDSS = cudssExecute(h, CUDSS_PHASE_ANALYSIS, &
       solverConfig, solverData, A, x, b)
  if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase analysis: ', statusDSS

  block
    integer :: defaultDevice = 0
    integer(8) :: hybridDeviceMemoryLimit = 0
    integer(c_size_t) :: sizeWritten
    
    status = cudaGetDevice(defaultDevice)
    if (status /= cudaSuccess) print *, 'cudaGetDevice: ', status

    do i = 1, deviceCount
       status = cudaSetDevice(deviceIndices(i))
       if (status /= cudaSuccess) print *, 'cudaSetDevice: ', status
       
       statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN, &
            hybridDeviceMemoryLimit, c_sizeof(hybridDeviceMemoryLimit), sizeWritten)
       if (statusDSS /= cudss_status_success) print *, 'cudssDataGet memory min: ', statusDSS

       print "('device: ', i0, ', CUDSS_DATA_HYBRID_MEMORY_MIN: ', i0)", &
            deviceIndices(i), hybridDeviceMemoryLimit
    end do

    status = cudaSetDevice(defaultDevice)
    if (status /= cudaSuccess) print *, 'cudaSetDevice default: ', status

  end block


  statusDSS = cudssExecute(h, CUDSS_PHASE_FACTORIZATION, &
       solverConfig, solverData, A, x, b)
  if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase factorization: ', statusDSS

  statusDSS = cudssExecute(h, CUDSS_PHASE_SOLVE, &
       solverConfig, solverData, A, x, b)
  if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase solve: ', statusDSS

  call dev_copy_d2h_r8(x_values, x_values_d, size(x_values))

  print *, 'x: '
  do c = 1, ncols
     print *, c, x_values(c)
  end do

  if (.not. validate_spd_upper_solution_r64(n, csr_offsets, csr_columns, csr_values, x_values, b_values, tol)) &
       error stop 1

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

  status = cudaGetDevice(currentDevice)
  if (status /= cudaSuccess) print *, 'cudaGetDevice: ', status
  do i = 1, deviceCount
     status = cudaSetDevice(deviceIndices(i))
     if (status /= cudaSuccess) print *, 'cudaSetDevice: ', status
     status = cudaStreamDestroy(deviceStreams(i))
     if (status /= cudaSuccess) print *, 'cudaStreamDestroy: ', status
  enddo
  status = cudaSetDevice(currentDevice)
  if (status /= cudaSuccess) print *, 'cudaSetDevice restore: ', status

  call dev_free(csr_offsets_d)
  call dev_free(csr_columns_d)
  call dev_free(csr_values_d)
  call dev_free(x_values_d)
  call dev_free(b_values_d)
  deallocate(csr_offsets, csr_columns, csr_values, x_values, b_values, deviceIndices, deviceStreams)

  print *, 'Example cudss_mg_mode PASSED'
end program main
  
