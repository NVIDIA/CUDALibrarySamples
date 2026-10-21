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
! This example demonstrates basic usage of cuDSS APIs for solving
! a high-precision linear system with a sparse matrix:
!                             Ax = b,
! where:
!     A is the sparse input matrix,
!     b is the (dense) right-hand side vector (or a matrix),
!     x is the (dense) solution vector (or a matrix),
! and A, b, and x use cudss_fp64mp2_t values.
! Note: This path requires device compute capability 9.0 or higher.
! Matches examples/cudss_high_precision.cpp.
program main
  use iso_fortran_env, only: wp => real64
  use iso_c_binding, only: c_double, c_int, c_loc, c_ptr, c_size_t, c_sizeof
  use cudautils
  use cudss
  implicit none

  type, bind(C) :: cudss_fp64mp2_t
    real(c_double) :: hi
    real(c_double) :: lo
  end type cudss_fp64mp2_t

  integer, parameter :: n = 5, nnz = 8, nrhs = 1
  integer :: csr_offsets(n+1), csr_columns(nnz)
  type(cudss_fp64mp2_t) :: csr_values(nnz), x_values(n), b_values(n)
  type(c_ptr) :: csr_offsets_d, csr_columns_d, csr_values_d, x_values_d, b_values_d

  type(cudssHandle) :: h
  type(cudssConfig) :: cfg
  type(cudssData) :: dat
  type(cudssMatrix) :: A, b, x
  type(cudssStatus) :: st
  type(cudssMatrixType) :: mtype = CUDSS_MTYPE_SPD
  type(cudssMatrixViewType) :: mview = CUDSS_MVIEW_UPPER
  type(cudssIndexBase) :: base = CUDSS_BASE_ONE
  integer(cuda_stream_kind) :: stream
  integer :: status, i, cc
  integer(c_int) :: det0 = 0, major, minor

  status = cudaDeviceGetAttribute(major, cudaDevAttrComputeCapabilityMajor, 0)
  if (status /= cudaSuccess) print *, 'cudaDeviceGetAttribute major', status
  status = cudaDeviceGetAttribute(minor, cudaDevAttrComputeCapabilityMinor, 0)
  if (status /= cudaSuccess) print *, 'cudaDeviceGetAttribute minor', status
  cc = major * 10 + minor
  if (cc < 90) then
    print *, 'Example SKIPPED: compute capability < 9.0 (', cc, ')'
    stop 0
  end if

  csr_offsets = [1, 3, 5, 7, 8, 9]
  csr_columns = [1, 3, 2, 3, 3, 5, 4, 5]
  csr_values = [to_mp2(4.0_wp), to_mp2(1.0_wp), to_mp2(3.0_wp), to_mp2(2.0_wp), &
       to_mp2(5.0_wp), to_mp2(1.0_wp), to_mp2(1.0_wp), to_mp2(2.0_wp)]
  b_values(1) = mp2(7.0_c_double, 4.0e-20_c_double)
  b_values(2) = mp2(12.0_c_double, -3.0e-23_c_double)
  b_values(3) = mp2(25.0_c_double, 1.0e-20_c_double - 2.0e-23_c_double)
  b_values(4) = mp2(4.0_c_double, 0.0_c_double)
  b_values(5) = mp2(13.0_c_double, 0.0_c_double)
  x_values = to_mp2(0.0_wp)

  call dev_malloc_i4(csr_offsets_d, csr_offsets)
  call dev_malloc_i4(csr_columns_d, csr_columns)
  call dev_malloc_mp2(csr_values_d, csr_values)
  call dev_malloc_mp2(b_values_d, b_values)
  call dev_malloc_mp2(x_values_d, x_values)

  status = cudaStreamCreate(stream)
  st = cudssCreate(h)
  st = cudssSetStream(h, stream)
  st = cudssConfigCreate(cfg)
  st = cudssDataCreate(h, dat)
  st = cudssConfigSet(cfg, CUDSS_CONFIG_DETERMINISTIC_MODE, det0, c_sizeof(det0))

  st = cudssMatrixCreateDn(b, int(n, 8), int(nrhs, 8), int(n, 8), b_values_d, &
       CUDSS_R_64F_MP2, CUDSS_LAYOUT_COL_MAJOR)
  st = cudssMatrixCreateDn(x, int(n, 8), int(nrhs, 8), int(n, 8), x_values_d, &
       CUDSS_R_64F_MP2, CUDSS_LAYOUT_COL_MAJOR)
  st = cudssMatrixCreateCsr(A, int(n, 8), int(n, 8), int(nnz, 8), csr_offsets_d, csr_columns_d, csr_values_d, &
       CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F_MP2, mtype, mview, base)

  st = cudssExecute(h, CUDSS_PHASE_ANALYSIS, cfg, dat, A, x, b)
  st = cudssExecute(h, CUDSS_PHASE_FACTORIZATION, cfg, dat, A, x, b)
  st = cudssExecute(h, CUDSS_PHASE_SOLVE, cfg, dat, A, x, b)

  status = cudaStreamSynchronize(stream)
  call dev_copy_d2h_mp2(x_values, x_values_d, n)

  st = cudssMatrixDestroy(A)
  st = cudssMatrixDestroy(b)
  st = cudssMatrixDestroy(x)
  st = cudssDataDestroy(h, dat)
  st = cudssConfigDestroy(cfg)
  st = cudssDestroy(h)

  status = cudaStreamDestroy(stream)

  block
    type(cudss_fp64mp2_t) :: exp_i
    do i = 1, n
       exp_i = expected(i)
       if (delta_mp2(x_values(i), exp_i) > 1.0e-30_wp) then
          error stop 2
       end if
    end do
  end block

  call dev_free(csr_offsets_d)
  call dev_free(csr_columns_d)
  call dev_free(csr_values_d)
  call dev_free(b_values_d)
  call dev_free(x_values_d)

  print *, 'Example cudss_high_precision PASSED'

contains

  subroutine dev_malloc_mp2(dev, host)
    type(cudss_fp64mp2_t), intent(in), target :: host(:)
    type(c_ptr), intent(out) :: dev
    integer(c_size_t) :: nbytes
    integer :: istat
    nbytes = c_sizeof(host(1)) * size(host)
    istat = cudaMalloc(dev, nbytes)
    if (istat /= cudaSuccess) error stop 'cudaMalloc failed (mp2)'
    istat = cudaMemcpy(dev, c_loc(host(1)), nbytes, cudaMemcpyHostToDevice)
    if (istat /= cudaSuccess) error stop 'cudaMemcpy H2D failed (mp2)'
  end subroutine dev_malloc_mp2

  subroutine dev_copy_d2h_mp2(host, dev, n)
    integer, intent(in) :: n
    type(cudss_fp64mp2_t), intent(out), target :: host(n)
    type(c_ptr), intent(in) :: dev
    integer(c_size_t) :: nbytes
    integer :: istat
    nbytes = c_sizeof(host(1)) * n
    istat = cudaMemcpy(c_loc(host(1)), dev, nbytes, cudaMemcpyDeviceToHost)
    if (istat /= cudaSuccess) error stop 'cudaMemcpy D2H failed (mp2)'
  end subroutine dev_copy_d2h_mp2

  pure function to_mp2(x) result(y)
    real(wp), intent(in) :: x
    type(cudss_fp64mp2_t) :: y
    y%hi = real(x, c_double)
    y%lo = 0.0_c_double
  end function to_mp2

  pure function mp2(hi, lo) result(y)
    real(c_double), intent(in) :: hi, lo
    type(cudss_fp64mp2_t) :: y
    y%hi = hi
    y%lo = lo
  end function mp2

  pure function expected(k) result(y)
    integer, intent(in) :: k
    type(cudss_fp64mp2_t) :: y
    select case (k)
    case (1); y = mp2(1.0_c_double, 1.0e-20_c_double)
    case (2); y = mp2(2.0_c_double, -1.0e-23_c_double)
    case (3); y = mp2(3.0_c_double, 0.0_c_double)
    case (4); y = mp2(4.0_c_double, 0.0_c_double)
    case (5); y = mp2(5.0_c_double, 0.0_c_double)
    end select
  end function expected

  pure function delta_mp2(a, e) result(d)
    type(cudss_fp64mp2_t), intent(in) :: a, e
    real(wp) :: d
    d = max(abs(a%hi - e%hi), abs(a%lo - e%lo))
  end function delta_mp2

end program main
