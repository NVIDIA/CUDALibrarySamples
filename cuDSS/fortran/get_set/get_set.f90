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
  ! This example demonstrates usage of cuDSS APIs with a focus on
  ! using non-default settings and retrieving extra data from the solver.
  ! As in the main example, a system of linear algebraic equations
  ! with a sparse matrix is solved:
  !                             Ax = b,
  ! where:
  !     A is the sparse input matrix,
  !     b is the (dense) right-hand side vector (or matrix),
  !     x is the (dense) solution vector (or matrix).
  ! Matches examples/cudss_get_set.cpp.
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
  type(cudssMatrixType) :: mtype = CUDSS_MTYPE_SYMMETRIC  ! NB: different from other examples
  type(cudssMatrixViewType) :: mview = CUDSS_MVIEW_UPPER
  type(cudssIndexBase) :: base = CUDSS_BASE_ONE

  type(cudssReorderingAlg) :: reorderAlg
  type(cudssMatchingAlg) :: matchingAlg

  integer :: useSuperpanel
  logical :: usedMatching
  
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

  statusDSS = cudssConfigCreate(solverConfig)
  if (statusDSS /= cudss_status_success) print *, 'cudssConfigCreate: ', statusDSS

  statusDSS = cudssDataCreate(h, solverData)
  if (statusDSS /= cudss_status_success) print *, 'cudssDataCreate: ', statusDSS

  reorderAlg = CUDSS_REORDERING_ALG_DEFAULT
  statusDSS = cudssConfigSet(solverConfig, CUDSS_CONFIG_REORDERING_ALG, reorderAlg, c_sizeof(reorderAlg%val))
  if (statusDSS /= cudss_status_success) print *, 'cudssConfigSet reorder: ', statusDSS

  matchingAlg = CUDSS_MATCHING_ALG_AUTO
  statusDSS = cudssConfigSet(solverConfig, CUDSS_CONFIG_MATCHING_ALG, matchingAlg, c_sizeof(matchingAlg%val))
  if (statusDSS /= cudss_status_success) print *, 'cudssConfigSet match: ', statusDSS

  useSuperpanel = 0
  statusDSS = cudssConfigSet(solverConfig, CUDSS_CONFIG_USE_SUPERPANELS, useSuperpanel, c_sizeof(useSuperpanel))
  if (statusDSS /= cudss_status_success) print *, 'cudssConfigSet superPanel: ', statusDSS
  
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
    integer, target :: perm(5)
    integer(c_size_t), target :: sizeWritten
    integer :: forceNonsymmetricMatching

    statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_PERM_REORDER_ROW, perm, c_sizeof(perm), sizeWritten)
    if (statusDSS /= cudss_status_success) print *, 'cudssDataGet reorder row perm: ', statusDSS
    print "('row perm:', 5(1x,i0))", perm
    print *, 'sizeWritten: ', sizeWritten

    statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_PERM_REORDER_COL, perm, c_sizeof(perm), sizeWritten)
    if (statusDSS /= cudss_status_success) print *, 'cudssDataGet reorder col perm: ', statusDSS
    print "('col perm:', 5(1x,i0))", perm
    print *, 'sizeWritten: ', sizeWritten

    statusDSS = cudssConfigGet(solverConfig, CUDSS_CONFIG_MATCHING_ALG, matchingAlg, &
         c_sizeof(matchingAlg%val), sizeWritten)
    if (statusDSS /= cudss_status_success) print *, 'cudssConfigGet matching: ', statusDSS

    statusDSS = cudssConfigGet(solverConfig, CUDSS_CONFIG_FORCE_NONSYMMETRIC_MATCHING, &
         forceNonsymmetricMatching, c_sizeof(forceNonsymmetricMatching), sizeWritten)
    if (statusDSS /= cudss_status_success) print *, 'cudssConfigGet force_nonsymmetric_matching: ', statusDSS
    print "('matching alg = ', i0)", matchingAlg%val
    print "('matching force-nonsymmetric flag = ', i0)", forceNonsymmetricMatching

    usedMatching = matchingAlg /= CUDSS_MATCHING_ALG_NONE
    if (usedMatching) then
       statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_PERM_MATCHING, perm, c_sizeof(perm), sizeWritten)
       if (statusDSS /= cudss_status_success) print *, 'cudssDataGet matching perm: ', statusDSS
       print "('matching perm:', 5(1x,i0))", perm
       print *, 'sizeWritten: ', sizeWritten
    end if

    if (reorderAlg == CUDSS_REORDERING_ALG_BTF_COLAMD .or. reorderAlg == CUDSS_REORDERING_ALG_COLAMD) then
       statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_PERM_ROW, perm, c_sizeof(perm), sizeWritten)
       if (statusDSS /= cudss_status_success) print *, 'cudssDataGet row perm: ', statusDSS
       print "('final row perm:', 5(1x,i0))", perm
       print *, 'sizeWritten: ', sizeWritten
       
       statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_PERM_COL, perm, c_sizeof(perm), sizeWritten)
       if (statusDSS /= cudss_status_success) print *, 'cudssDataGet col perm: ', statusDSS
       print "('final col perm:', 5(1x,i0))", perm
       print *, 'sizeWritten: ', sizeWritten
    end if       
  end block

  block
    integer(c_int64_t), target :: memoryEstimates(16)
    integer(c_size_t), target :: sizeWritten

    statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_MEMORY_ESTIMATES, &
         c_loc(memoryEstimates(1)), c_sizeof(memoryEstimates), c_loc(sizeWritten))
    if (statusDSS /= cudss_status_success) print *, 'cudssDataGet memoryEstimates: ', statusDSS
    print "('memory estimates: device: ', i0, ' (stable) ', i0, ' (peak)')", memoryEstimates(1), memoryEstimates(2)
    print "('memory estimates: host: ', i0, ' (stable) ', i0, ' (peak)')", memoryEstimates(1), memoryEstimates(2)
    print "('memory estimates: hybrid peak: ', i0, ' (GPU) ', i0, ' (CPU)')", memoryEstimates(1), memoryEstimates(2)
  end block

  statusDSS = cudssExecute(h, CUDSS_PHASE_FACTORIZATION, &
       solverConfig, solverData, A, x, b)
  if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase factorization: ', statusDSS

  
  block
    integer(c_size_t), target :: sizeWritten
    integer, target :: info, npivots, inertia(2)
    integer(c_int64_t), target :: lu_nnz
    integer :: iterRefNsteps
    real(8) :: diag(n)
    type(c_ptr) :: diag_d
    real(8) :: row_scale(n), col_scale(n)
    type(c_ptr) :: row_scale_d, col_scale_d

    
    info = -1
    statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_INFO, info, c_sizeof(info), sizeWritten)
    if (statusDSS /= cudss_status_success) print *, 'cudssGetData info: ', statusDSS
    print "('cuDSS info = ', i0)", info

    npivots = -1
    statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_NPIVOTS, npivots, c_sizeof(npivots), sizeWritten)
    if (statusDSS /= cudss_status_success) print *, 'cudssGetData npivots: ', statusDSS
    print "('cuDSS npivots = ', i0)", npivots

    inertia = -1
    statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_INERTIA, inertia, c_sizeof(inertia), sizeWritten)
    if (statusDSS /= cudss_status_success) print *, 'cudssGetData inertia: ', statusDSS
    print "('cuDSS inertia =', 2(1x,i0))", inertia

    ! checking size for lu_nnz before retrieving it
    lu_nnz = -1
    statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_LU_NNZ, lu_nnz, int(0, c_size_t), sizeWritten)
    if (statusDSS /= cudss_status_success) print *, 'cudssGetData lu_nnz size: ', statusDSS
    print "('bytes requires for lu_nnz = ', i0)", sizeWritten
    
    lu_nnz = -1
    statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_LU_NNZ, lu_nnz, c_sizeof(lu_nnz), sizeWritten)
    if (statusDSS /= cudss_status_success) print *, 'cudssGetData lu_nnz: ', statusDSS
    print "('cuDSS nnz in LU = ', i0)", lu_nnz

    diag = 0.0d0
    call dev_malloc_r8(diag_d, diag)
    statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_DIAG, diag_d, c_sizeof(diag), c_loc(sizeWritten))
    if (statusDSS /= cudss_status_success) print *, 'cudssGetData diag: ', statusDSS
    call dev_copy_d2h_r8(diag, diag_d, n)
    print "('cuDSS diag =', 5f12.7)", diag

    if ((matchingAlg == CUDSS_MATCHING_ALG_AUTO) .or. (matchingAlg == CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT)) then
       call dev_malloc_r8(row_scale_d, row_scale)
       statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_SCALE_ROW, row_scale_d, c_sizeof(row_scale), c_loc(sizeWritten))
       if (statusDSS /= cudss_status_success) print *, 'cudssGetData row scale: ', statusDSS
       call dev_copy_d2h_r8(row_scale, row_scale_d, n)
       print "('row scale:', 5(1x,f12.5))", row_scale
       print *, 'sizeWritten: ', sizeWritten
       
       call dev_malloc_r8(col_scale_d, col_scale)
       statusDSS = cudssDataGet(h, solverData, CUDSS_DATA_SCALE_COL, col_scale_d, c_sizeof(col_scale), c_loc(sizeWritten))
       if (statusDSS /= cudss_status_success) print *, 'cudssGetData col scale: ', statusDSS
       call dev_copy_d2h_r8(col_scale, col_scale_d, n)
       print "('col scale:', 5(1x,f12.5))", col_scale
       print *, 'sizeWritten: ', sizeWritten
       call dev_free(row_scale_d)
       call dev_free(col_scale_d)
    end if
    call dev_free(diag_d)
           
    iterRefNsteps = 2
    statusDSS = cudssConfigSet(solverConfig, CUDSS_CONFIG_IR_N_STEPS, iterRefNsteps, c_sizeof(iterRefNSteps))
    if (statusDSS /= cudss_status_success) print *, 'cudssConfigSet IR: ', statusDSS
  end block

  statusDSS = cudssExecute(h, CUDSS_PHASE_SOLVE, &
       solverConfig, solverData, A, x, b)
  if (statusDSS /= cudss_status_success) print *, 'cudssExecute phase solve: ', statusDSS

  status = cudaStreamSynchronize(stream)
  if (status /= cudaSuccess) print *, 'cudaStreamSynchronize: ', status

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
  statusDSS = cudssMatrixDestroy(x)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixDestroy(x): ', statusDSS
  statusDSS = cudssMatrixDestroy(b)
  if (statusDSS /= cudss_status_success) print *, 'cudssMatrixDestroy(b): ', statusDSS
  statusDSS = cudssDataDestroy(h, solverData)
  if (statusDSS /= cudss_status_success) print *, 'cudssDataDestroy(h,solverData): ', statusDSS  
  statusDSS = cudssConfigDestroy(solverConfig)
  if (statusDSS /= cudss_status_success) print *, 'cudssConfigDestroy(solverConfig): ', statusDSS
  statusDSS = cudssDestroy(h)
  if (statusDSS /= cudss_status_success) print *, 'cudssDestroy: ', statusDSS

  status = cudaStreamDestroy(stream)
  if (status /= cudaSuccess) print *, 'cudaStreamDestroy: ', status
  
  call dev_free(csr_offsets_d)
  call dev_free(csr_columns_d)
  call dev_free(csr_values_d)
  call dev_free(x_values_d)
  call dev_free(b_values_d)
  deallocate(csr_offsets, csr_columns, csr_values, x_values, b_values)

  print *, 'Example cudss_get_set PASSED'
end program main
