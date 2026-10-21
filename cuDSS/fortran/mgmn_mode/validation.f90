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
!
! Shared numerical checks for Fortran cuDSS examples (symmetric SPD, upper CSR, 1-based indices).
!
module validation
  use iso_fortran_env, only: wp => real64
  implicit none
  private
  public :: validate_spd_upper_solution_r64, validate_spd_upper_solution_z64, &
       relative_residual_spd_upper_r64

contains

  pure function dnrm2_r(n, x) result(t)
    integer, intent(in) :: n
    real(wp), intent(in) :: x(n)
    real(wp) :: t
    t = sqrt(sum(x**2))
  end function dnrm2_r

  pure function dnrm2_z(n, x) result(t)
    integer, intent(in) :: n
    complex(wp), intent(in) :: x(n)
    real(wp) :: t
    t = sqrt(sum(abs(x)**2))
  end function dnrm2_z

  pure subroutine sym_upper_mv_r(n, csr_offsets, csr_columns, csr_values, x, y)
    integer, intent(in) :: n
    integer, intent(in) :: csr_offsets(n+1), csr_columns(*)
    real(wp), intent(in) :: csr_values(*), x(n)
    real(wp), intent(out) :: y(n)
    integer :: i, k, j
    real(wp) :: v
    y = 0.0_wp
    do i = 1, n
      do k = csr_offsets(i), csr_offsets(i+1)-1
        j = csr_columns(k)
        v = csr_values(k)
        y(i) = y(i) + v * x(j)
        if (j /= i) y(j) = y(j) + v * x(i)
      end do
    end do
  end subroutine sym_upper_mv_r

  pure subroutine sym_upper_mv_z(n, csr_offsets, csr_columns, csr_values, x, y)
    integer, intent(in) :: n
    integer, intent(in) :: csr_offsets(n+1), csr_columns(*)
    complex(wp), intent(in) :: csr_values(*), x(n)
    complex(wp), intent(out) :: y(n)
    integer :: i, k, j
    complex(wp) :: v
    y = (0.0_wp, 0.0_wp)
    do i = 1, n
      do k = csr_offsets(i), csr_offsets(i+1)-1
        j = csr_columns(k)
        v = csr_values(k)
        y(i) = y(i) + v * x(j)
        if (j /= i) y(j) = y(j) + v * x(i)
      end do
    end do
  end subroutine sym_upper_mv_z

  pure function frob_sym_upper_r(n, csr_offsets, csr_columns, csr_values) result(af)
    integer, intent(in) :: n
    integer, intent(in) :: csr_offsets(n+1), csr_columns(*)
    real(wp), intent(in) :: csr_values(*)
    real(wp) :: af, s
    integer :: i, k, j
    s = 0.0_wp
    do i = 1, n
      do k = csr_offsets(i), csr_offsets(i+1)-1
        j = csr_columns(k)
        if (j == i) then
          s = s + csr_values(k)**2
        else
          s = s + 2.0_wp * csr_values(k)**2
        end if
      end do
    end do
    af = sqrt(s)
  end function frob_sym_upper_r

  function relative_residual_spd_upper_r64(n, csr_offsets, csr_columns, csr_values, x, b) result(rel)
    integer, intent(in) :: n
    integer, intent(in) :: csr_offsets(n+1), csr_columns(*)
    real(wp), intent(in) :: csr_values(*), x(n), b(n)
    real(wp) :: rel
    real(wp) :: y(n), r(n), af, num, den
    call sym_upper_mv_r(n, csr_offsets, csr_columns, csr_values, x, y)
    r = y - b
    num = dnrm2_r(n, r)
    af = frob_sym_upper_r(n, csr_offsets, csr_columns, csr_values)
    den = af * dnrm2_r(n, x) + dnrm2_r(n, b)
    if (den <= 0.0_wp) den = 1.0_wp
    rel = num / den
  end function relative_residual_spd_upper_r64

  logical function validate_spd_upper_solution_r64(n, csr_offsets, csr_columns, csr_values, x, b, tol)
    integer, intent(in) :: n
    integer, intent(in) :: csr_offsets(n+1), csr_columns(*)
    real(wp), intent(in) :: csr_values(*), x(n), b(n)
    real(wp), intent(in) :: tol
    real(wp) :: rel
    rel = relative_residual_spd_upper_r64(n, csr_offsets, csr_columns, csr_values, x, b)
    validate_spd_upper_solution_r64 = (rel <= tol)
    if (.not. validate_spd_upper_solution_r64) print *, 'FAILED residual check: rel = ', rel, ' tol = ', tol
  end function validate_spd_upper_solution_r64

  logical function validate_spd_upper_solution_z64(n, csr_offsets, csr_columns, csr_values, x, b, tol)
    integer, intent(in) :: n
    integer, intent(in) :: csr_offsets(n+1), csr_columns(*)
    complex(wp), intent(in) :: csr_values(*), x(n), b(n)
    real(wp), intent(in) :: tol
    complex(wp) :: y(n), r(n)
    real(wp) :: num, den, af
    integer :: i, k, j
    complex(wp) :: v
    af = 0.0_wp
    do i = 1, n
      do k = csr_offsets(i), csr_offsets(i+1)-1
        j = csr_columns(k)
        if (j == i) then
          af = af + abs(csr_values(k))**2
        else
          af = af + 2.0_wp * abs(csr_values(k))**2
        end if
      end do
    end do
    af = sqrt(af)
    call sym_upper_mv_z(n, csr_offsets, csr_columns, csr_values, x, y)
    r = y - b
    num = dnrm2_z(n, r)
    den = af * dnrm2_z(n, x) + dnrm2_z(n, b)
    if (den <= 0.0_wp) den = 1.0_wp
    validate_spd_upper_solution_z64 = ((num / den) <= tol)
    if (.not. validate_spd_upper_solution_z64) print *, 'FAILED residual check (complex): rel = ', num/den
  end function validate_spd_upper_solution_z64

end module validation
