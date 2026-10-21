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
! Debug printing helpers for cuDSS matrix objects (dense and CSR).

module cudssutils
  use iso_c_binding
  use iso_fortran_env, only: wp => real64
  use cudss
  use cudautils
  implicit none

  private
  public :: printMatrix

contains

  subroutine printMatrix(A, solverConfig)
    type(cudssMatrix), intent(in) :: A
    type(cudssConfig), optional, intent(in) :: solverConfig
    type(cudssMatrixFormat) :: mformat
    type(cudssStatus) :: statusDSS
    logical :: isBatched, isUniformBatch = .false.
    integer :: uniformBatchSize

    statusDSS = cudssMatrixGetFormat(A, mformat)
    if (statusDSS /= cudss_status_success) print *, 'cudssMatrixGetFormat A: ', statusDSS
    isBatched = iand(mformat%val, CUDSS_MFORMAT_BATCH%val) /= 0

    if (present(solverConfig)) then
       block
          integer, target :: ubs
          integer(c_size_t) :: sizeWritten
          statusDSS = cudssConfigGet(solverConfig, CUDSS_CONFIG_UBATCH_SIZE, ubs, &
               c_sizeof(ubs), sizeWritten)
          if (statusDSS /= cudss_status_success) print *, 'cudssConfigGet ubatch size: ', statusDSS
          uniformBatchSize = ubs
          if (uniformBatchSize > 1) isUniformBatch = .true.
       end block
    end if

    if (isUniformBatch) then
       if (iand(mformat%val, CUDSS_MFORMAT_DENSE%val) /= 0) then
          call printUniformBatchDenseMatrix(A, uniformBatchSize)
       else if (iand(mformat%val, CUDSS_MFORMAT_CSR%val) /= 0) then
          call printUniformBatchCSRMatrix(A, uniformBatchSize)
       else
          print *, 'printMatrix(): unknown batched matrix type'
       end if
    else if (isBatched) then
       if (iand(mformat%val, CUDSS_MFORMAT_DENSE%val) /= 0) then
          call printBatchDenseMatrix(A)
       else if (iand(mformat%val, CUDSS_MFORMAT_CSR%val) /= 0) then
          call printBatchCSRMatrix(A)
       else
          print *, 'printMatrix(): unknown batched matrix type'
       end if
    else
       if (iand(mformat%val, CUDSS_MFORMAT_DENSE%val) /= 0) then
          call printDenseMatrix(A)
       else if (iand(mformat%val, CUDSS_MFORMAT_CSR%val) /= 0) then
          call printCSRMatrix(A)
       else
          print *, 'printMatrix(): unknown matrix type'
       end if
    end if
  end subroutine printMatrix

  subroutine printDenseMatrix(A)
    type(cudssMatrix), intent(in) :: A
    integer(c_int64_t) :: nrows, ncols, ld
    type(c_ptr) :: v_cdp
    integer(c_int) :: valueType
    type(cudssLayout) :: layout
    type(cudssStatus) :: statusDSS
    real(wp), allocatable :: r8(:)
    integer :: r, c

    print "('  format: dense')"
    statusDSS = cudssMatrixGetDn(A, nrows=nrows, ncols=ncols, ld=ld, values=v_cdp, &
         valueType=valueType, layout=layout)
    if (statusDSS /= cudss_status_success) print *, 'cudssMatrixGetDn: ', statusDSS
    print "('  nrows: ', i0)", nrows
    print "('  ncols: ', i0)", ncols
    print "('  ld: ', i0)", ld

    if (valueType == CUDSS_R_64F) then
       allocate(r8(int(nrows*ncols)))
       call dev_copy_d2h_r8(r8, v_cdp, size(r8))
       print *, ' values: '
       do r = 1, int(nrows)
          print *, (r, r8(r+(c-1)*int(ld)), c=1, int(ncols))
       end do
       deallocate(r8)
       print *, ' value type: real(8)'
    else if (valueType == CUDSS_C_64F) then
       print *, ' printDenseMatrix: complex dense not implemented in this build'
    else
       print *, ' printDenseMatrix: unsupported value type'
    end if

    if (layout == CUDSS_LAYOUT_COL_MAJOR) then
       print *, ' layout: column major'
    else if (layout == CUDSS_LAYOUT_ROW_MAJOR) then
       print *, ' layout: row major'
    end if
  end subroutine printDenseMatrix

  subroutine printUniformBatchDenseMatrix(A, uniformBatchSize)
    type(cudssMatrix), intent(in) :: A
    integer, intent(in) :: uniformBatchSize
    integer(c_int64_t) :: nrows, ncols, ld
    type(c_ptr) :: v_cdp
    integer(c_int) :: valueType
    type(cudssLayout) :: layout
    type(cudssStatus) :: statusDSS
    real(wp), allocatable :: r8(:)
    integer :: r, c, i

    print "('  format: uniform batched dense')"
    statusDSS = cudssMatrixGetDn(A, nrows=nrows, ncols=ncols, ld=ld, values=v_cdp, &
         valueType=valueType, layout=layout)
    if (statusDSS /= cudss_status_success) print *, 'cudssMatrixGetDn: ', statusDSS
    print "('  uniform batch size: ', i0)", uniformBatchSize
    print "('  nrows: ', i0)", nrows
    print "('  ncols: ', i0)", ncols
    print "('  ld: ', i0)", ld

    if (valueType == CUDSS_R_64F) then
       allocate(r8(int(nrows*ncols*uniformBatchSize)))
       call dev_copy_d2h_r8(r8, v_cdp, size(r8))
       do i = 1, uniformBatchSize
          print "('  Matrix: ', i0)", i
          print *, '   values: '
          do r = 1, int(nrows)
             print *, (r, r8(r+(c-1)*int(ld)+(i-1)*int(ld*ncols)), c=1, int(ncols))
          end do
       end do
       deallocate(r8)
       print *, ' value type: real(8)'
    end if
  end subroutine printUniformBatchDenseMatrix

  subroutine printBatchDenseMatrix(A)
    type(cudssMatrix), intent(in) :: A
    integer(c_int64_t) :: batchCount
    type(c_ptr) :: nrows_cp, ncols_cp, ld_cp, v_cdp
    integer(c_int) :: valueType, integerType
    type(cudssLayout) :: layout
    type(cudssStatus) :: statusDSS
    real(wp), allocatable :: r8(:)
    integer :: r, c, i, bc

    print "('  format: batched dense')"
    statusDSS = cudssMatrixGetBatchDn(A, batchCount=batchCount, nrows=nrows_cp, ncols=ncols_cp, &
         ld=ld_cp, values=v_cdp, integerType=integerType, valueType=valueType, layout=layout)
    if (statusDSS /= cudss_status_success) print *, 'cudssMatrixGetBatchDn: ', statusDSS
    print "('  batchCount: ', i0)", batchCount

    if (integerType == CUDSS_R_32I) then
       bc = int(batchCount)
       block
         integer(c_int32_t), pointer :: nrows_p(:), ncols_p(:), ld_p(:)
         integer(c_int64_t), allocatable, target :: ptrbuf(:)
         type(c_ptr) :: valptr
         integer :: cudaStatus
         allocate(ptrbuf(bc))
         cudaStatus = cudaMemcpy(c_loc(ptrbuf(1)), v_cdp, bc * c_sizeof(ptrbuf(1)), &
              cudaMemcpyDeviceToHost)
         if (cudaStatus /= cudaSuccess) error stop 'cudaMemcpy D2H failed (ptrbuf)'
         call c_f_pointer(nrows_cp, nrows_p, [bc])
         call c_f_pointer(ncols_cp, ncols_p, [bc])
         call c_f_pointer(ld_cp, ld_p, [bc])

         do i = 1, bc
            print "('  Matrix: ', i0)", i
            print "('    nrows: ', i0)", nrows_p(i)
            print "('    ncols: ', i0)", ncols_p(i)
            print "('    ld: ', i0)", ld_p(i)
            if (valueType == CUDSS_R_64F) then
               if (ncols_p(i) < 0 .or. ncols_p(i) > 1000 .or. &
                    nrows_p(i) < 0 .or. nrows_p(i) > 10000) then
                  print *, '    (values omitted: invalid batch metadata)'
                  cycle
               end if
               valptr = transfer(ptrbuf(i), c_null_ptr)
               allocate(r8(nrows_p(i)*ncols_p(i)))
               call dev_copy_d2h_r8(r8, valptr, size(r8))
               print *, '   values: '
               do r = 1, nrows_p(i)
                  print *, (r, r8(r+(c-1)*ld_p(i)), c=1, ncols_p(i))
               end do
               deallocate(r8)
            end if
         end do
         deallocate(ptrbuf)
       end block
    end if
  end subroutine printBatchDenseMatrix

  subroutine printCSRMatrix(A)
    type(cudssMatrix), intent(in) :: A
    integer(c_int64_t) :: nrows, ncols, nnz
    type(c_ptr) :: rowStart_cdp, colIndices_cdp, v_cdp
    integer(c_int) :: offsetType, indexType, valueType
    type(cudssMatrixType) :: mtype
    type(cudssMatrixViewType) :: mview
    type(cudssIndexBase) :: indexBase
    type(cudssStatus) :: statusDSS
    integer, allocatable :: i4(:)
    real(wp), allocatable :: r8(:)
    integer :: r

    print "('  format: sparse')"
    statusDSS = cudssMatrixGetCsr(A, nrows=nrows, ncols=ncols, nnz=nnz, rowStart=rowStart_cdp, &
         colIndices=colIndices_cdp, values=v_cdp, offsetType=offsetType, indexType=indexType, &
         valueType=valueType, mtype=mtype, mview=mview, indexBase=indexBase)
    if (statusDSS /= cudss_status_success) print *, 'cudssMatrixGetCsr: ', statusDSS
    print "('  nrows: ', i0)", nrows
    print "('  ncols: ', i0)", ncols
    print "('  nnz: ', i0)", nnz

    if (offsetType == CUDSS_R_32I) then
       allocate(i4(int(nrows)+1))
       call dev_copy_d2h_i4(i4, rowStart_cdp, size(i4))
       print *, ' csr offsets: ', i4
       deallocate(i4)
    end if
    if (indexType == CUDSS_R_32I) then
       allocate(i4(int(nnz)))
       call dev_copy_d2h_i4(i4, colIndices_cdp, size(i4))
       print *, ' col indices: ', i4
       deallocate(i4)
    end if
    if (valueType == CUDSS_R_64F) then
       allocate(r8(int(nnz)))
       call dev_copy_d2h_r8(r8, v_cdp, size(r8))
       print *, ' values: '
       do r = 1, int(nnz)
          print *, r, r8(r)
       end do
       deallocate(r8)
    end if

    if (mtype == CUDSS_MTYPE_SPD) print *, ' matrix type: SPD'
    if (mview == CUDSS_MVIEW_UPPER) print *, ' matrix view: upper'
    if (indexBase == CUDSS_BASE_ONE) print *, ' index base: one'
  end subroutine printCSRMatrix

  subroutine printUniformBatchCSRMatrix(A, uniformBatchSize)
    type(cudssMatrix), intent(in) :: A
    integer, intent(in) :: uniformBatchSize
    print "('  format: uniform batched sparse (summary only, batch=', i0, ')')", uniformBatchSize
    call printCSRMatrix(A)
  end subroutine printUniformBatchCSRMatrix

  subroutine printBatchCSRMatrix(A)
    type(cudssMatrix), intent(in) :: A
    integer(c_int64_t) :: batchCount
    type(cudssStatus) :: statusDSS

    print "('  format: batched sparse (summary only)')"
    statusDSS = cudssMatrixGetBatchCsr(A, batchCount=batchCount)
    if (statusDSS /= cudss_status_success) print *, 'cudssMatrixGetBatchCsr: ', statusDSS
    print "('  batchCount: ', i0)", batchCount
  end subroutine printBatchCSRMatrix

end module cudssutils
