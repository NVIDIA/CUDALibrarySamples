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
! Matches examples/cudss_dense_matrix_helpers.cpp (dense helpers only; no full linear solve).

program main
  use iso_fortran_env, only: wp => real64
  use iso_c_binding, only: c_int, c_int64_t
  use cudautils
  use cudss
  use cudssutils
  implicit none

  integer(c_int64_t) :: nrows, ncols, ld
  integer :: i, j, lin
  real(wp), allocatable, target :: dense_h(:), dense_h2(:)
  type(c_ptr) :: dense_d, dense_d2
  type(cudssHandle) :: handle
  type(cudssMatrix) :: B
  type(cudssStatus) :: st
  type(cudssMatrixFormat) :: mf
  integer(c_int64_t) :: nr_o, nc_o, ldb_o
  type(c_ptr), target :: vals_o
  integer(c_int) :: dt_o
  type(cudssLayout) :: lay_o

  real(wp) :: newv
  integer(c_int) :: fm

  print "('Maj/Min/Patch:', 3(1x, i0))", CUDSS_VERSION_MAJOR, CUDSS_VERSION_MINOR, CUDSS_VERSION_PATCH

  nrows = 6
  ncols = nrows/2
  ld = 2*nrows

  allocate(dense_h(ld*ncols), dense_h2(ld*ncols))

  lin = 0
  do j = 1, ncols
    do i = 1, ld
      lin = lin + 1
      dense_h(lin) = real(lin, kind=wp)
    end do
  end do

  dense_h2 = dense_h
  call dev_malloc_r8(dense_d, dense_h)
  call dev_malloc_r8(dense_d2, dense_h2)

  st = cudssCreate(handle)
  if (st /= cudss_status_success) print *, 'cudssCreate ', st

  st = cudssMatrixCreateDn(B, nrows, ncols, ld, dense_d, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR)
  if (st /= cudss_status_success) print *, 'cudssMatrixCreateDn ', st

  st = cudssMatrixGetDn(B, nr_o, nc_o, ldb_o, vals_o, dt_o, lay_o)
  if (st /= cudss_status_success) print *, 'cudssMatrixGetDn ', st

  if (nr_o /= nrows .or. nc_o /= ncols .or. ldb_o /= ld .or. dt_o /= CUDSS_R_64F &
       .or. lay_o /= CUDSS_LAYOUT_COL_MAJOR .or. dev_ptr_value(vals_o) /= dev_ptr_value(dense_d)) then
     print *, 'FAILED: cudssMatrixGetDn consistency'
     error stop 2
  end if
  call printMatrix(B)

  newv = -1.0_wp
  dense_h(1) = newv
  call dev_copy_h2d(dense_d, dense_h)
  st = cudssMatrixGetDn(B, values=vals_o)
  call dev_copy_d2h_r8(dense_h, vals_o, size(dense_h))
  if (abs(dense_h(1) - newv) > epsilon(1.0_wp)) then
     print *, 'FAILED: in-place value'
     error stop 3
  end if
  call printMatrix(B)

  newv = -4.0_wp
  dense_h2(2) = newv
  call dev_copy_h2d(dense_d2, dense_h2)
  st = cudssMatrixSetValues(B, dense_d2)
  if (st /= cudss_status_success) print *, 'cudssMatrixSetValues ', st
  st = cudssMatrixGetDn(B, values=vals_o)
  call dev_copy_d2h_r8(dense_h, vals_o, size(dense_h))
  if (abs(dense_h(2) - newv) > epsilon(1.0_wp)) then
     print *, 'FAILED: SetValues'
     error stop 4
  end if
  call printMatrix(B)

  st = cudssMatrixGetFormat(B, mf)
  fm = mf%val
  if (iand(fm, CUDSS_MFORMAT_DENSE%val) == 0) then
     print *, 'FAILED: format', fm
     error stop 5
  end if

  st = cudssMatrixDestroy(B)
  st = cudssDestroy(handle)

  call dev_free(dense_d)
  call dev_free(dense_d2)
  deallocate(dense_h, dense_h2)

  print *, 'Example cudss_dense_matrix_helpers PASSED'
end program main
