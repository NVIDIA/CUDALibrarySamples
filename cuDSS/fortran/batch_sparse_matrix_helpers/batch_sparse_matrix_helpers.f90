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
! Matches examples/cudss_batch_sparse_matrix_helpers.cpp (batched CSR helpers; 1-based indices).

program main
  use iso_fortran_env, only: wp => real64
  use iso_c_binding, only: c_int64_t
  use cudautils
  use cudss
  implicit none

  integer, parameter :: batchCount = 2
  integer, parameter :: n1 = 5, n2 = 6, nnz1 = 8, nnz2 = 11
  integer, parameter :: nnz1b = 7, nnz2b = 10

  integer :: csr1_offsets(n1+1), csr1_columns(nnz1)
  integer :: csr2_offsets(n2+1), csr2_columns(nnz2)
  real(wp) :: csr1_values(nnz1), csr2_values(nnz2)

  integer :: csr1_offsets_b(n1+1), csr1_columns_b(nnz1b)
  integer :: csr2_offsets_b(n2+1), csr2_columns_b(nnz2b)
  real(wp) :: csr1_values_b(nnz1b), csr2_values_b(nnz2b)

  type(c_ptr) :: d1_off, d1_col, d2_off, d2_col
  type(c_ptr) :: d1_val, d2_val
  type(c_ptr) :: d1_off_b, d1_col_b, d2_off_b, d2_col_b
  type(c_ptr) :: d1_val_b, d2_val_b
  type(c_ptr) :: b1_d, b2_d, b1_b, b2_b

  real(wp) :: h1_val_b(max(nnz1, nnz1b)), h2_val_b(max(nnz2, nnz2b))
  real(wp) :: hb1(n1), hb2(n2), hb1_b(n1), hb2_b(n2)

  type(cuda_devptr) :: p_off(batchCount), p_col(batchCount), p_val(batchCount), p_b(batchCount)
  type(cuda_devptr) :: p_off_b(batchCount), p_col_b(batchCount), p_val_b(batchCount), p_b_b(batchCount)
  type(c_ptr) :: p_off_d, p_col_d, p_val_d, p_b_d
  type(c_ptr) :: p_off_db, p_col_db, p_val_db, p_b_db

  integer :: nrows(batchCount), ncols(batchCount), nnz(batchCount), nrhs(batchCount), ldb(batchCount)
  integer(c_int64_t) :: bc_out
  integer(c_int) :: offsetType_out, indexType_out, valueType_out
  type(cudssMatrixType) :: mtype_out
  type(cudssMatrixViewType) :: mview_out
  type(cudssIndexBase) :: indexBase_out
  integer(c_int) :: indexType_out_b, valueType_out_b
  type(cudssLayout) :: layout_out_b
  type(c_ptr) :: nnz_out
  type(c_ptr) :: off_out, col_out, val_out, b_val_out
  type(cudssHandle) :: handle
  type(cudssConfig) :: cfg
  type(cudssData) :: dat
  type(cudssMatrix) :: A, b
  type(cudssStatus) :: st
  type(cudssMatrixType) :: mtype = CUDSS_MTYPE_SPD
  type(cudssMatrixViewType) :: mview = CUDSS_MVIEW_UPPER
  type(cudssIndexBase) :: base = CUDSS_BASE_ONE
  type(cudssMatrixFormat) :: mf

  integer(cuda_stream_kind) :: stream
  integer :: status
  real(wp) :: new_csr(2), new_b(2), out_csr(2), out_b(2)

  csr1_offsets = [1, 3, 5, 7, 8, 9]
  csr1_columns = [1, 3, 2, 3, 3, 5, 4, 5]
  csr1_values = [4.0_wp, 1.0_wp, 3.0_wp, 2.0_wp, 5.0_wp, 1.0_wp, 1.0_wp, 2.0_wp]

  csr2_offsets = [1, 3, 5, 8, 9, 11, 12]
  csr2_columns = [1, 6, 2, 5, 3, 5, 4, 6, 5, 6, 6]
  csr2_values = [3.0_wp, 1.0_wp, 2.0_wp, 1.0_wp, 6.0_wp, 2.0_wp, 2.0_wp, 5.0_wp, 7.0_wp, 3.0_wp, 8.0_wp]

  hb1 = [7.0_wp, 12.0_wp, 25.0_wp, 4.0_wp, 13.0_wp]
  hb2 = [9.0_wp, 9.0_wp, 40.0_wp, 20.0_wp, 61.0_wp, 70.0_wp]

  call dev_malloc_i4(d1_off, csr1_offsets)
  call dev_malloc_i4(d1_col, csr1_columns)
  call dev_malloc_r8(d1_val, csr1_values)
  call dev_malloc_i4(d2_off, csr2_offsets)
  call dev_malloc_i4(d2_col, csr2_columns)
  call dev_malloc_r8(d2_val, csr2_values)
  call dev_malloc_r8(b1_d, hb1)
  call dev_malloc_r8(b2_d, hb2)

  p_off = [c_devloc(d1_off), c_devloc(d2_off)]
  p_col = [c_devloc(d1_col), c_devloc(d2_col)]
  p_val = [c_devloc(d1_val), c_devloc(d2_val)]
  p_b = [c_devloc(b1_d), c_devloc(b2_d)]

  h1_val_b(1:nnz1) = csr1_values
  h2_val_b(1:nnz2) = csr2_values
  call dev_malloc_r8(d1_val_b, h1_val_b(1:max(nnz1, nnz1b)))
  call dev_malloc_r8(d2_val_b, h2_val_b(1:max(nnz2, nnz2b)))
  hb1_b = hb1
  hb2_b = hb2
  call dev_malloc_r8(b1_b, hb1_b)
  call dev_malloc_r8(b2_b, hb2_b)
  p_off_b = p_off
  p_col_b = p_col
  p_val_b = [c_devloc(d1_val_b), c_devloc(d2_val_b)]
  p_b_b = [c_devloc(b1_b), c_devloc(b2_b)]

  call dev_upload_ptr_array(p_off_d, p_off, batchCount)
  call dev_upload_ptr_array(p_col_d, p_col, batchCount)
  call dev_upload_ptr_array(p_val_d, p_val, batchCount)
  call dev_upload_ptr_array(p_b_d, p_b, batchCount)
  call dev_upload_ptr_array(p_off_db, p_off_b, batchCount)
  call dev_upload_ptr_array(p_col_db, p_col_b, batchCount)
  call dev_upload_ptr_array(p_val_db, p_val_b, batchCount)
  call dev_upload_ptr_array(p_b_db, p_b_b, batchCount)

  nrows = [n1, n2]
  ncols = [n1, n2]
  nnz = [nnz1, nnz2]
  nrhs = [1, 1]
  ldb = nrows

  status = cudaStreamCreate(stream)
  st = cudssCreate(handle)
  st = cudssSetStream(handle, stream)
  st = cudssConfigCreate(cfg)
  st = cudssDataCreate(handle, dat)

  st = cudssMatrixCreateBatchDn(b, int(batchCount,8), nrows, nrhs, ldb, p_b_d, &
       CUDSS_R_32I, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR)
  st = cudssMatrixCreateBatchCsr(A, int(batchCount,8), nrows, ncols, nnz, &
       p_off_d, p_col_d, p_val_d, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, mtype, mview, base)

  ! Note: cuDSS matrix objects are lightweight wrappers around the user data.
  st = cudssMatrixGetBatchCsr(A, batchCount=bc_out, rowStart=off_out, colIndices=col_out, &
       values=val_out, offsetType=offsetType_out, indexType=indexType_out, &
       valueType=valueType_out, mtype=mtype_out, mview=mview_out, indexBase=indexBase_out)
  if (bc_out /= int(batchCount,8) .or. .not. c_associated(off_out, p_off_d) .or. &
       .not. c_associated(col_out, p_col_d) .or. .not. c_associated(val_out, p_val_d) .or. &
       offsetType_out /= CUDSS_R_32I .or. indexType_out /= CUDSS_R_32I .or. &
       valueType_out /= CUDSS_R_64F .or. mtype_out /= mtype .or. mview_out /= mview .or. &
       indexBase_out /= base) then
     print *, 'FAILED GetBatchCsr for A'
     error stop 2
  end if
  print *, 'Success: Check for matrix A from cudssMatrixGetBatchCsr passed'

  st = cudssMatrixGetBatchDn(b, batchCount=bc_out, values=b_val_out, integerType=indexType_out_b, &
       valueType=valueType_out_b, layout=layout_out_b)
  if (bc_out /= int(batchCount,8) .or. .not. c_associated(b_val_out, p_b_d) .or. &
       indexType_out_b /= CUDSS_R_32I .or. valueType_out_b /= CUDSS_R_64F &
       .or. layout_out_b /= CUDSS_LAYOUT_COL_MAJOR) then
     print *, 'FAILED GetBatchDn for b'
     error stop 2
  end if
  print *, 'Success: Check for matrix B from cudssMatrixGetBatchDn passed'

  new_csr = [-4.0_wp, -3.0_wp]
  new_b = [-2.0_wp, -5.0_wp]
  h1_val_b(1) = new_csr(1)
  h2_val_b(1) = new_csr(2)
  hb1_b(1) = new_b(1)
  hb2_b(1) = new_b(2)
  call dev_copy_h2d(d1_val_b, h1_val_b(1:1))
  call dev_copy_h2d(d2_val_b, h2_val_b(1:1))
  call dev_copy_h2d(b1_b, hb1_b(1:1))
  call dev_copy_h2d(b2_b, hb2_b(1:1))

  st = cudssMatrixSetBatchValues(A, p_val_db)
  st = cudssMatrixSetBatchValues(b, p_b_db)

  block
    real(wp) :: t1(1), t2(1), tb1(1), tb2(1)
    call dev_copy_d2h_r8(t1, d1_val_b, 1)
    call dev_copy_d2h_r8(t2, d2_val_b, 1)
    call dev_copy_d2h_r8(tb1, b1_b, 1)
    call dev_copy_d2h_r8(tb2, b2_b, 1)
    out_csr = [t1(1), t2(1)]
    out_b = [tb1(1), tb2(1)]
  end block
  if (any(abs(out_csr - new_csr) > epsilon(1.0_wp)) .or. any(abs(out_b - new_b) > epsilon(1.0_wp))) then
     print *, 'FAILED SetBatchValues'
     error stop 3
  end if

  csr1_offsets_b = [1, 3, 5, 6, 7, 8]
  csr1_columns_b = [1, 3, 2, 3, 3, 4, 5]
  csr1_values_b = [4.0_wp, 1.0_wp, 3.0_wp, 2.0_wp, 5.0_wp, 1.0_wp, 2.0_wp]

  csr2_offsets_b = [1, 3, 5, 7, 8, 10, 11]
  csr2_columns_b = [1, 6, 2, 5, 3, 5, 4, 5, 6, 6]
  csr2_values_b = [3.0_wp, 1.0_wp, 2.0_wp, 1.0_wp, 6.0_wp, 2.0_wp, 5.0_wp, 7.0_wp, 3.0_wp, 8.0_wp]

  call dev_malloc_i4(d1_off_b, csr1_offsets_b)
  call dev_malloc_i4(d1_col_b, csr1_columns_b)
  call dev_malloc_i4(d2_off_b, csr2_offsets_b)
  call dev_malloc_i4(d2_col_b, csr2_columns_b)
  h1_val_b(1:nnz1b) = csr1_values_b
  h2_val_b(1:nnz2b) = csr2_values_b
  call dev_copy_h2d(d1_val_b, h1_val_b(1:max(nnz1, nnz1b)))
  call dev_copy_h2d(d2_val_b, h2_val_b(1:max(nnz2, nnz2b)))

  p_off_b = [c_devloc(d1_off_b), c_devloc(d2_off_b)]
  p_col_b = [c_devloc(d1_col_b), c_devloc(d2_col_b)]
  p_val_b = [c_devloc(d1_val_b), c_devloc(d2_val_b)]
  call dev_upload_ptr_array(p_off_db, p_off_b, batchCount)
  call dev_upload_ptr_array(p_col_db, p_col_b, batchCount)
  call dev_upload_ptr_array(p_val_db, p_val_b, batchCount)

  st = cudssMatrixSetBatchCsrPointers(A, p_off_db, p_col_db, p_val_db)

  ! Note: changing the matrix structure does not change the nnz returned by
  ! cudssMatrixGetBatchCsr; the original nnz array passed to cudssMatrixCreateBatchCsr
  ! is still returned.
  st = cudssMatrixGetBatchCsr(A, nnz=nnz_out)
  block
    integer, pointer :: nnz_new(:)
    call c_f_pointer(nnz_out, nnz_new, [batchCount])
    if (any(nnz_new /= nnz)) then
       print *, 'FAILED nnz after SetBatchCsrPointers', nnz_new
       error stop 6
    end if
  end block

  st = cudssMatrixGetFormat(A, mf)
  if (iand(mf%val, CUDSS_MFORMAT_CSR%val) == 0 .or. iand(mf%val, CUDSS_MFORMAT_BATCH%val) == 0) then
     print *, 'FAILED format A', mf%val
     error stop 4
  end if
  st = cudssMatrixGetFormat(b, mf)
  if (iand(mf%val, CUDSS_MFORMAT_DENSE%val) == 0 .or. iand(mf%val, CUDSS_MFORMAT_BATCH%val) == 0) then
     print *, 'FAILED format b', mf%val
     error stop 5
  end if

  st = cudssMatrixDestroy(A)
  st = cudssMatrixDestroy(b)
  st = cudssDataDestroy(handle, dat)
  st = cudssConfigDestroy(cfg)
  st = cudssDestroy(handle)
  status = cudaStreamDestroy(stream)

  call dev_free(d1_off)
  call dev_free(d1_col)
  call dev_free(d2_off)
  call dev_free(d2_col)
  call dev_free(d1_val)
  call dev_free(d2_val)
  call dev_free(d1_off_b)
  call dev_free(d1_col_b)
  call dev_free(d2_off_b)
  call dev_free(d2_col_b)
  call dev_free(d1_val_b)
  call dev_free(d2_val_b)
  call dev_free(b1_d)
  call dev_free(b2_d)
  call dev_free(b1_b)
  call dev_free(b2_b)
  call dev_free(p_off_d)
  call dev_free(p_col_d)
  call dev_free(p_val_d)
  call dev_free(p_b_d)
  call dev_free(p_off_db)
  call dev_free(p_col_db)
  call dev_free(p_val_db)
  call dev_free(p_b_db)

  print *, 'Example cudss_batch_sparse_matrix_helpers PASSED'

end program main
