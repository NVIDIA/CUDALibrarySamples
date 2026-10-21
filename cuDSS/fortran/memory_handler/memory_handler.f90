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
! This example demonstrates usage of cuDSS APIs for user-defined memory allocators.
! The APIs are shown for an example of solving a system of linear algebraic
! equations with a sparse matrix:
!                             Ax = b,
! where:
!     A is the sparse input matrix,
!     b is the (dense) right-hand side vector (or a matrix),
!     x is the (dense) solution vector (or a matrix).
! Matches examples/cudss_memory_handler.cpp.
!
! Module memHandler implements callbacks (cudaMallocAsync/cudaFreeAsync) analogous to the
! C++ ExampleDeviceMemPool and example_device_alloc/example_device_dealloc wrappers.
module memHandler
  use iso_c_binding
  use cudautils

  type memoryPool
     type(cuda_devptr) :: pool
  end type memoryPool

contains

  integer(c_int) function poolAlloc(ctx, ptr, size, stream) bind(c)
    implicit none
    type(c_ptr), value :: ctx
    type(cuda_devptr) :: ptr
    integer(c_size_t), value :: size
    integer(cuda_stream_kind), value :: stream
    print "('allocating ', i0, ' bytes of device memory')", size
    if (size == 0) then
       ptr = C_NULL_CUDA_DEVPTR
       poolAlloc = cudaSuccess
    else
       poolAlloc = cudaMallocAsync(ptr, size, stream)
    end if
  end function poolAlloc

  integer(c_int) function poolFree(ctx, ptr, size, stream) bind(c)
    implicit none
    type(c_ptr), value :: ctx
    type(cuda_devptr), value :: ptr
    integer(c_size_t), value :: size
    integer(cuda_stream_kind), value :: stream
    print "('freeing ', i0, ' bytes of device memory')", size
    poolFree = cudaFreeAsync(ptr, stream)
  end function poolFree

end module memHandler


program main
  use iso_c_binding
  use cudautils
  use cudss
  use cudssutils
  use memHandler
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

  type(cudssDeviceMemHandler), target :: devMemHandler
  type(memoryPool), target :: pool
  
  integer :: status
  integer(cuda_stream_kind) :: stream
  integer :: c
  real(8), parameter :: tol = 1.d-11
  
  ! property types

  print "('Maj/Min/Patch:', 3(1x, i0))", CUDSS_VERSION_MAJOR, CUDSS_VERSION_MINOR, CUDSS_VERSION_PATCH
  
  allocate(csr_offsets(n+1), csr_columns(nnz), csr_values(nnz), &
       x_values(nrhs*n), b_values(nrhs*n))

  ! Symmetric matrix with upper triangular elements stored
  ! full matrix problem is:
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
  
  status = cudaStreamCreate(stream)
  if (status /= cudaSuccess) print *, 'cudaStreamCreate: ', status

  statusDSS = cudssCreate(h)
  if (statusDSS /= cudss_status_success) print *, 'cudssCreate: ', statusDSS

  statusDSS = cudssSetStream(h, stream)
  if (statusDSS /= cudss_status_success) print *, 'cudssSetStream: ', statusDSS

  ! the ctx/pool isn't used for this trivial memory handler
  devMemHandler%ctx = c_loc(pool)
  devMemHandler%device_alloc = c_funloc(poolAlloc)
  devMemHandler%device_free = c_funloc(poolFree)
  devMemHandler%name = f2cString('trivial device memory manager', CUDSS_ALLOCATOR_NAME_LEN)

  ! not really needed here, but can be used to detach the current handler
  statusDSS = cudssSetDeviceMemHandler(h, c_null_ptr)
  if (statusDSS /= cudss_status_success) print *, 'cudssSetDeviceMemHandler detach: ', statusDSS

  statusDSS = cudssSetDeviceMemHandler(h, devMemHandler)
  if (statusDSS /= cudss_status_success) print *, 'cudssSetDeviceMemHandler: ', statusDSS
  
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
  
  statusDSS = cudssExecute(h, CUDSS_PHASE_ANALYSIS, &
       solverConfig, solverData, A, x, b)
  if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase analysis: ', statusDSS

  statusDSS = cudssExecute(h, CUDSS_PHASE_FACTORIZATION, &
       solverConfig, solverData, A, x, b)
  if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase factorization: ', statusDSS

  statusDSS = cudssExecute(h, CUDSS_PHASE_SOLVE, &
       solverConfig, solverData, A, x, b)
  if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase solve: ', statusDSS

  block
    type(cudssDeviceMemHandler) :: returnedHandler
    character(len=CUDSS_ALLOCATOR_NAME_LEN) :: handlerName

    statusDSS = cudssGetDeviceMemHandler(h, returnedHandler)
    if (statusDSS /= cudss_status_success) print *, 'cudssGetDeviceMemHandler: ', statusDSS
    handlerName = transfer(returnedHandler%name, handlerName)
    print "('Device memory handler name: ', a)", handlerName(1:index(handlerName, C_NULL_CHAR) - 1)
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

  print *, 'Example cudss_memory_handler PASSED'
end program main
