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
! This example demonstrates usage of cuDSS helper APIs for matrix objects.
! Matches examples/cudss_sparse_matrix_helpers.cpp (CSR helpers; 0-based indices in this port).

program main
  use iso_fortran_env, only: wp => real64
  use iso_c_binding, only: c_int, c_int64_t, c_size_t, c_sizeof, c_loc
  use cudautils
  use cudss
  implicit none

  integer, parameter :: nrows = 5, ncols = 5, nnz = 9, nnz2 = 5

  integer, target :: csr_offsets(0:nrows), csr_columns(0:nnz-1), csr_offsets2(0:nrows), csr_columns2(0:nnz2-1)
  real(wp), target :: csr_values(0:nnz-1), csr_values2(0:nnz2-1)

  type(c_ptr) :: csr_offsets_d, csr_columns_d, csr_values_d
  type(c_ptr) :: csr_offsets_d2, csr_columns_d2, csr_values_d2

  type(cudssHandle) :: handle
  type(cudssMatrix) :: A
  type(cudssStatus) :: st
  type(cudssMatrixType) :: mtype = CUDSS_MTYPE_SYMMETRIC
  type(cudssMatrixViewType) :: mview = CUDSS_MVIEW_UPPER
  type(cudssIndexBase) :: base = CUDSS_BASE_ZERO

  integer(c_int64_t) :: nr_o, nc_o, nz_o, nz_new
  type(c_ptr), target :: ro, co, vo
  integer(c_int) :: ot, it, vt
  type(cudssMatrixType) :: mt_o
  type(cudssMatrixViewType) :: mv_o
  type(cudssIndexBase) :: ib_o

  real(wp) :: newv
  type(cudssMatrixFormat) :: mf
  integer :: i, status
  integer(c_size_t) :: nbytes

  ! Note: this example uses 0-based indexing (CUDSS_BASE_ZERO) with zero-offset
  ! Fortran arrays (a(0:n)) for demonstration. One-based indexing (CUDSS_BASE_ONE)
  ! with natural unit-offset arrays (a(n+1)) is usually more idiomatic in Fortran;
  ! see cudss_simple / cudss_high_precision for that style.
  print *, 'CUDSS Version (Major,Minor,PatchLevel): ', CUDSS_VERSION_MAJOR, CUDSS_VERSION_MINOR, CUDSS_VERSION_PATCH

  csr_offsets = [0, 2, 4, 6, 8, 9]
  csr_columns = [0, 1, 1, 2, 2, 3, 3, 4, 4]
  do i = 0, nnz-1
    csr_values(i) = real(2*i, kind=wp)
  end do

  csr_offsets2 = [0, 1, 2, 3, 4, 5]
  do i = 0, nrows-1
    csr_columns2(i) = i
  end do
  do i = 0, nnz2-1
    csr_values2(i) = real(4*i, kind=wp)
  end do

  call dev_malloc_i4_lb(csr_offsets_d, csr_offsets, 0, nrows)
  call dev_malloc_i4_lb(csr_columns_d, csr_columns, 0, nnz-1)
  call dev_malloc_r8_lb(csr_values_d, csr_values, 0, nnz-1)
  call dev_malloc_i4_lb(csr_offsets_d2, csr_offsets2, 0, nrows)
  call dev_malloc_i4_lb(csr_columns_d2, csr_columns2, 0, nnz2-1)
  block
    real(wp), target :: buf(0:max(nnz, nnz2)-1)
    buf(0:nnz2-1) = csr_values2
    call dev_malloc_r8_lb(csr_values_d2, buf, 0, max(nnz, nnz2)-1)
  end block

  st = cudssCreate(handle)

  st = cudssMatrixCreateCsr(A, int(nrows,8), int(ncols,8), int(nnz,8), &
       csr_offsets_d, csr_columns_d, csr_values_d, &
       CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, mtype, mview, base)

  ! rowEnd sits between rowStart and colIndices; use keywords so rowEnd is omitted (NULL in C).
  st = cudssMatrixGetCsr(A, nrows=nr_o, ncols=nc_o, nnz=nz_o, rowStart=ro, colIndices=co, values=vo, &
       offsetType=ot, indexType=it, valueType=vt, mtype=mt_o, mview=mv_o, indexBase=ib_o)

  if (nr_o /= nrows .or. nc_o /= ncols .or. nz_o /= nnz .or. ot /= CUDSS_R_32I .or. it /= CUDSS_R_32I &
       .or. vt /= CUDSS_R_64F .or. mt_o /= mtype .or. mv_o /= mview .or. ib_o /= base &
       .or. dev_ptr_value(ro) /= dev_ptr_value(csr_offsets_d) .or. dev_ptr_value(co) /= dev_ptr_value(csr_columns_d) &
       .or. dev_ptr_value(vo) /= dev_ptr_value(csr_values_d)) then
     print *, 'FAILED GetCsr initial'
     error stop 2
  end if

  ! Modifying values of the matrix A in-place
  newv = -1.0_wp
  csr_values(0) = newv
  call dev_copy_h2d_r8_lb(csr_values_d, csr_values, 0, nnz-1)
  st = cudssMatrixGetCsr(A, values=vo)
  nbytes = int(nnz, c_size_t) * c_sizeof(csr_values(0))
  status = cudaMemcpy(c_loc(csr_values(0)), vo, nbytes, cudaMemcpyDeviceToHost)
  if (status /= cudaSuccess) error stop 'cudaMemcpy D2H failed (csr_values)'
  if (abs(csr_values(0) - newv) > epsilon(1.0_wp)) then
     print *, 'FAILED in-place val'
     error stop 3
  end if

  ! Replacing values of matrix A with new data
  newv = -4.0_wp
  block
    real(wp), target :: buf(0:max(nnz, nnz2)-1)
    nbytes = int(max(nnz, nnz2), c_size_t) * c_sizeof(buf(0))
    status = cudaMemcpy(c_loc(buf(0)), csr_values_d2, nbytes, cudaMemcpyDeviceToHost)
    if (status /= cudaSuccess) error stop 'cudaMemcpy D2H failed (csr_values_d2)'
    buf(1) = newv
    status = cudaMemcpy(csr_values_d2, c_loc(buf(0)), nbytes, cudaMemcpyHostToDevice)
    if (status /= cudaSuccess) error stop 'cudaMemcpy H2D failed (csr_values_d2)'
  end block
  st = cudssMatrixSetValues(A, csr_values_d2)
  st = cudssMatrixGetCsr(A, values=vo)
  nbytes = int(nnz, c_size_t) * c_sizeof(csr_values(0))
  status = cudaMemcpy(c_loc(csr_values(0)), vo, nbytes, cudaMemcpyDeviceToHost)
  if (status /= cudaSuccess) error stop 'cudaMemcpy D2H failed (csr_values after SetValues)'
  if (abs(csr_values(1) - newv) > epsilon(1.0_wp)) then
     print *, 'FAILED SetValues'
     error stop 4
  end if

  ! Changing the matrix structure (here: removing the last entry)
  st = cudssMatrixSetCsrPointers(A, csr_offsets_d2, csr_columns_d2, csr_values_d2)
  st = cudssMatrixGetCsr(A, nnz=nz_new, rowStart=ro, colIndices=co, values=vo)

  ! Note: changing the matrix structure does not change the nnz returned by
  ! cudssMatrixGetCsr; the original value passed to cudssMatrixCreateCsr is returned.
  if (nz_new /= nz_o) then
     print *, 'FAILED nnz after SetCsrPointers', nz_new, nz_o
     error stop 6
  end if

  ! Demonstrating cudssMatrixGetFormat
  st = cudssMatrixGetFormat(A, mf)
  if (mf%val /= CUDSS_MFORMAT_CSR%val) then
     print *, 'FAILED format', mf%val
     error stop 5
  end if

  ! Destroying opaque objects, matrix wrappers and the cuDSS library handle
  st = cudssMatrixDestroy(A)
  st = cudssDestroy(handle)

  call dev_free(csr_offsets_d)
  call dev_free(csr_columns_d)
  call dev_free(csr_values_d)
  call dev_free(csr_offsets_d2)
  call dev_free(csr_columns_d2)
  call dev_free(csr_values_d2)

  print *, 'Example cudss_sparse_matrix_helpers PASSED'
end program main
