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
! This example demonstrates usage of cuDSS APIs for computing
! a (non-factorized) Schur complement matrix.
! The subset of rows and columns which define the Schur complement
! is provided as an integer vector with binary values.
! As in the main example, a system of linear algebraic equations
! with a sparse matrix is considered:
!                             Ax = b,
! where:
!     A is the sparse input matrix,
!     b is the (dense) right-hand side vector (or matrix),
!     x is the (dense) solution vector (or matrix).
! The C++ sample comment notes the Schur block may be solved outside cuDSS (e.g. cuSOLVER);
! this Fortran port solves the dense 2x2 Schur system on the host after partial solves.
! Matches examples/cudss_schur_complement.cpp.
program main
  use iso_fortran_env, only: wp => real64
  use iso_c_binding, only: c_int, c_int64_t, c_ptr, c_size_t, c_sizeof, c_loc
  use cudautils
  use cudss
  implicit none

  integer, parameter :: n = 5, nnz = 8, nrhs = 1
  integer, allocatable, target :: schur_idx(:)
  integer :: csr_offsets(0:n), csr_columns(0:nnz-1)
  real(wp) :: csr_values(0:nnz-1), x_values(n), b_values(n), br(2)
  type(c_ptr) :: csr_offsets_d, csr_columns_d, csr_values_d, x_values_d, b_values_d

  type(cudssHandle) :: h
  type(cudssConfig) :: cfg
  type(cudssData) :: dat
  type(cudssMatrix) :: A, b, x
  type(cudssMatrix), target :: S
  type(cudssStatus) :: st
  type(cudssMatrixType) :: mtype = CUDSS_MTYPE_SYMMETRIC
  type(cudssMatrixViewType) :: mview = CUDSS_MVIEW_UPPER
  type(cudssIndexBase) :: base = CUDSS_BASE_ZERO
  integer(cuda_stream_kind) :: stream
  integer :: status, i, j
  integer(c_int), target :: compute_schur
  integer(c_int64_t), target :: schur_shape(3)
  integer(c_size_t), target :: written
  integer(c_int64_t) :: snr, snc, sld
  type(c_ptr) :: schur_dev
  real(wp), allocatable, target :: schur_host(:)
  type(cudssPhase) :: ph
  integer(c_size_t) :: schur_nbytes
  logical :: schur_gpu_alloc

  integer(c_int64_t) :: ldb, ldx

  print *, '---------------------------------------------------------'
  print *, 'cuDSS example: computing a 2x2 Schur complement for a'
  print *, 'real linear 5x5 system with a symmetric matrix'
  print *, '---------------------------------------------------------'

  print *, 'CUDSS Version (Major,Minor,PatchLevel): ', CUDSS_VERSION_MAJOR, CUDSS_VERSION_MINOR, CUDSS_VERSION_PATCH

  ! Note: this example uses 0-based indexing (CUDSS_BASE_ZERO) with zero-offset
  ! Fortran arrays (a(0:n)) for demonstration. One-based indexing (CUDSS_BASE_ONE)
  ! with natural unit-offset arrays (a(n+1)) is usually more idiomatic in Fortran;
  ! see cudss_simple / cudss_high_precision for that style.
  csr_offsets = [0, 2, 4, 6, 7, 8]
  csr_columns = [0, 2, 1, 2, 2, 4, 3, 4]
  csr_values = [4.0_wp, 1.0_wp, 3.0_wp, 2.0_wp, 5.0_wp, 1.0_wp, 1.0_wp, 2.0_wp]
  b_values = [7.0_wp, 12.0_wp, 25.0_wp, 4.0_wp, 13.0_wp]

  allocate(schur_idx(n))
  schur_idx = [0, 1, 0, 0, 1]

  call dev_malloc_i4_lb(csr_offsets_d, csr_offsets, 0, n)
  call dev_malloc_i4_lb(csr_columns_d, csr_columns, 0, nnz-1)
  call dev_malloc_r8_lb(csr_values_d, csr_values, 0, nnz-1)
  call dev_malloc_r8(b_values_d, b_values)
  x_values = 0.0_wp
  call dev_malloc_r8(x_values_d, x_values)

  status = cudaStreamCreate(stream)
  st = cudssCreate(h)
  st = cudssSetStream(h, stream)
  st = cudssConfigCreate(cfg)
  st = cudssDataCreate(h, dat)

  compute_schur = 1
  st = cudssConfigSet(cfg, CUDSS_CONFIG_SCHUR_MODE, compute_schur, c_sizeof(compute_schur))
  st = cudssDataSet(h, dat, CUDSS_DATA_USER_SCHUR_INDICES, schur_idx, &
       int(n, kind=c_size_t) * c_sizeof(schur_idx(1)))

  ldb = int(n, 8)
  ldx = int(n, 8)
  st = cudssMatrixCreateDn(b, int(n,8), int(nrhs,8), ldb, b_values_d, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR)
  st = cudssMatrixCreateDn(x, int(n,8), int(nrhs,8), ldx, x_values_d, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR)
  st = cudssMatrixCreateCsr(A, int(n,8), int(n,8), int(nnz,8), csr_offsets_d, csr_columns_d, csr_values_d, &
       CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, mtype, mview, base)

  st = cudssExecute(h, CUDSS_PHASE_ANALYSIS, cfg, dat, A, x, b)
  st = cudssDataGet(h, dat, CUDSS_DATA_SCHUR_SHAPE, c_loc(schur_shape(1)), &
       int(c_sizeof(schur_shape), kind=c_size_t), c_loc(written))
  print *, 'schur shape: nrows = ', schur_shape(1), ', ncols = ', schur_shape(2), &
       ', sparse nnz (for a general matrix) = ', schur_shape(3)

  st = cudssExecute(h, CUDSS_PHASE_FACTORIZATION, cfg, dat, A, x, b)

  snr = schur_shape(1)
  snc = schur_shape(2)
  sld = snr

  schur_nbytes = int(snr * snc, kind=c_size_t) * int(c_sizeof(1.0_wp), kind=c_size_t)
  schur_gpu_alloc = .false.
  status = cudaMalloc(schur_dev, schur_nbytes)
  if (status /= cudaSuccess) then
     print *, 'cudaMalloc(schur_dev) failed: ', status
     error stop 3
  end if
  schur_gpu_alloc = .true.

  st = cudssMatrixCreateDn(S, snr, snc, sld, schur_dev, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR)
  st = cudssDataGet(h, dat, CUDSS_DATA_SCHUR_MATRIX, c_loc(S), int(c_sizeof(S), kind=c_size_t), c_loc(written))

  status = cudaStreamSynchronize(stream)

  allocate(schur_host(int(snr * snc)))
  call dev_copy_d2h_r8(schur_host, schur_dev, int(snr * snc))

  do i = 1, int(snr)
    do j = 1, int(snc)
      print *, 'schur_matrix_values(', i - 1, ',', j - 1, ') = ', &
           schur_host((j - 1) * int(snr) + i)
    end do
  end do

  ph%val = ior(CUDSS_PHASE_SOLVE_FWD_PERM%val, CUDSS_PHASE_SOLVE_FWD%val)
  st = cudssExecute(h, ph, cfg, dat, A, x, b)
  st = cudssExecute(h, CUDSS_PHASE_SOLVE_DIAG, cfg, dat, A, b, x)

  status = cudaStreamSynchronize(stream)
  call dev_copy_d2h_r8(b_values, b_values_d, n)
  associate (ns => int(snr))
    br(1) = b_values(n - ns + 1)
    br(2) = b_values(n - ns + 2)
    ! schur_host is column-major length ns*ns (same layout as cuDSS dense Schur)
    call solve_2x2(schur_host(1), schur_host(2), schur_host(3), schur_host(4), br(1), br(2))
    b_values(n - ns + 1) = br(1)
    b_values(n - ns + 2) = br(2)
    call dev_copy_h2d(b_values_d, b_values)
  end associate

  ph%val = ior(CUDSS_PHASE_SOLVE_BWD%val, CUDSS_PHASE_SOLVE_BWD_PERM%val)
  st = cudssExecute(h, ph, cfg, dat, A, x, b)

  status = cudaStreamSynchronize(stream)
  call dev_copy_d2h_r8(x_values, x_values_d, n)

  do i = 1, n
    if (abs(x_values(i) - real(i, kind=wp)) > 2.e-15_wp) then
       print *, 'FAILED x(', i, ')=', x_values(i), ' expected ', real(i, kind=wp)
       error stop 2
    end if
  end do

  st = cudssMatrixDestroy(A)
  st = cudssMatrixDestroy(b)
  st = cudssMatrixDestroy(x)
  st = cudssMatrixDestroy(S)

  st = cudssDataDestroy(h, dat)
  st = cudssConfigDestroy(cfg)
  st = cudssDestroy(h)

  status = cudaStreamDestroy(stream)
  status = cudaDeviceSynchronize()

  deallocate(schur_host)
  if (schur_gpu_alloc) then
     status = cudaFree(schur_dev)
     if (status /= cudaSuccess) print *, 'cudaFree(schur_dev): ', status
  end if
  call dev_free(csr_offsets_d)
  call dev_free(csr_columns_d)
  call dev_free(csr_values_d)
  call dev_free(x_values_d)
  call dev_free(b_values_d)
  deallocate(schur_idx)

  print *, 'Example cudss_schur_complement PASSED'

contains

  pure subroutine solve_2x2(a11, a21, a12, a22, rhs1, rhs2)
    real(wp), intent(in) :: a11, a21, a12, a22
    real(wp), intent(inout) :: rhs1, rhs2
    real(wp) :: det, x1, x2
    det = a11 * a22 - a12 * a21
    x1 = (rhs1 * a22 - rhs2 * a12) / det
    x2 = (a11 * rhs2 - a21 * rhs1) / det
    rhs1 = x1
    rhs2 = x2
  end subroutine solve_2x2

end program main
