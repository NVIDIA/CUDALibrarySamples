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
! Minimal CUDA runtime helpers for the cuDSS Fortran examples.
! Provides alloc/copy/stream wrappers via iso_c_binding to libcudart.

module cudautils
  use, intrinsic :: iso_c_binding
  implicit none

  integer, parameter :: cuda_stream_kind = c_intptr_t

  integer, parameter :: cudaSuccess = 0
  integer, parameter :: cudaMemcpyHostToDevice = 1
  integer, parameter :: cudaMemcpyDeviceToHost = 2

  integer, parameter :: cudaDevAttrComputeCapabilityMajor = 75
  integer, parameter :: cudaDevAttrComputeCapabilityMinor = 76

  type, bind(C) :: cuda_devptr
    type(c_ptr) :: devptr
  end type cuda_devptr

  type(cuda_devptr), parameter :: C_NULL_CUDA_DEVPTR = cuda_devptr(C_NULL_PTR)

  interface
    integer(c_int) function cudaStreamCreate_c(stream) bind(C, name='cudaStreamCreate')
      import c_ptr, c_int
      type(c_ptr) :: stream
    end function cudaStreamCreate_c

    integer(c_int) function cudaStreamDestroy_c(stream) bind(C, name='cudaStreamDestroy')
      import c_ptr, c_int
      type(c_ptr), value :: stream
    end function cudaStreamDestroy_c

    integer(c_int) function cudaStreamSynchronize_c(stream) bind(C, name='cudaStreamSynchronize')
      import c_ptr, c_int
      type(c_ptr), value :: stream
    end function cudaStreamSynchronize_c

    integer(c_int) function cudaDeviceSynchronize() bind(C, name='cudaDeviceSynchronize')
      import c_int
    end function cudaDeviceSynchronize

    integer(c_int) function cudaMalloc(devPtr, size) bind(C, name='cudaMalloc')
      import c_ptr, c_size_t, c_int
      type(c_ptr) :: devPtr
      integer(c_size_t), value :: size
    end function cudaMalloc

    integer(c_int) function cudaFree(devPtr) bind(C, name='cudaFree')
      import c_ptr, c_int
      type(c_ptr), value :: devPtr
    end function cudaFree

    integer(c_int) function cudaMemcpy(dst, src, count, kind) bind(C, name='cudaMemcpy')
      import c_ptr, c_size_t, c_int
      type(c_ptr), value :: dst, src
      integer(c_size_t), value :: count
      integer(c_int), value :: kind
    end function cudaMemcpy

    integer(c_int) function cudaMallocAsync_c(devPtr, size, stream) bind(C, name='cudaMallocAsync')
      import c_ptr, c_size_t, c_int
      type(c_ptr) :: devPtr
      integer(c_size_t), value :: size
      type(c_ptr), value :: stream
    end function cudaMallocAsync_c

    integer(c_int) function cudaFreeAsync_c(devPtr, stream) bind(C, name='cudaFreeAsync')
      import c_ptr, c_int
      type(c_ptr), value :: devPtr, stream
    end function cudaFreeAsync_c

    integer(c_int) function cudaGetDeviceCount(count) bind(C, name='cudaGetDeviceCount')
      import c_int
      integer(c_int) :: count
    end function cudaGetDeviceCount

    integer(c_int) function cudaGetDevice(device) bind(C, name='cudaGetDevice')
      import c_int
      integer(c_int) :: device
    end function cudaGetDevice

    integer(c_int) function cudaSetDevice(device) bind(C, name='cudaSetDevice')
      import c_int
      integer(c_int), value :: device
    end function cudaSetDevice

    integer(c_int) function cudaDeviceGetAttribute(value, attr, device) &
         bind(C, name='cudaDeviceGetAttribute')
      import c_int
      integer(c_int) :: value
      integer(c_int), value :: attr
      integer(c_int), value :: device
    end function cudaDeviceGetAttribute
  end interface

  interface cudaStreamCreate
     module procedure cudaStreamCreate_i
  end interface

  interface cudaStreamDestroy
     module procedure cudaStreamDestroy_i
  end interface

  interface cudaStreamSynchronize
     module procedure cudaStreamSynchronize_i
  end interface

  interface cudaMallocAsync
     module procedure cudaMallocAsync_ptr, cudaMallocAsync_devptr
  end interface

  interface cudaFreeAsync
     module procedure cudaFreeAsync_ptr, cudaFreeAsync_devptr
  end interface

contains

  function stream_to_cptr(stream) result(p)
    integer(cuda_stream_kind), intent(in) :: stream
    type(c_ptr) :: p
    p = transfer(stream, p)
  end function stream_to_cptr

  integer function cudaStreamCreate_i(stream) result(status)
    integer(cuda_stream_kind), intent(out) :: stream
    type(c_ptr) :: p
    status = cudaStreamCreate_c(p)
    stream = transfer(p, stream)
  end function cudaStreamCreate_i

  integer function cudaStreamDestroy_i(stream) result(status)
    integer(cuda_stream_kind), intent(in) :: stream
    status = cudaStreamDestroy_c(stream_to_cptr(stream))
  end function cudaStreamDestroy_i

  integer function cudaStreamSynchronize_i(stream) result(status)
    integer(cuda_stream_kind), intent(in) :: stream
    status = cudaStreamSynchronize_c(stream_to_cptr(stream))
  end function cudaStreamSynchronize_i

  integer function cudaMallocAsync_ptr(devPtr, size, stream) result(status)
    type(c_ptr), intent(inout) :: devPtr
    integer(c_size_t), intent(in) :: size
    integer(cuda_stream_kind), intent(in) :: stream
    status = cudaMallocAsync_c(devPtr, size, stream_to_cptr(stream))
  end function cudaMallocAsync_ptr

  integer function cudaMallocAsync_devptr(devPtr, size, stream) result(status)
    type(cuda_devptr), intent(inout) :: devPtr
    integer(c_size_t), intent(in) :: size
    integer(cuda_stream_kind), intent(in) :: stream
    status = cudaMallocAsync_ptr(devPtr%devptr, size, stream)
  end function cudaMallocAsync_devptr

  integer function cudaFreeAsync_ptr(devPtr, stream) result(status)
    type(c_ptr), intent(in) :: devPtr
    integer(cuda_stream_kind), intent(in) :: stream
    status = cudaFreeAsync_c(devPtr, stream_to_cptr(stream))
  end function cudaFreeAsync_ptr

  integer function cudaFreeAsync_devptr(devPtr, stream) result(status)
    type(cuda_devptr), intent(in) :: devPtr
    integer(cuda_stream_kind), intent(in) :: stream
    status = cudaFreeAsync_ptr(devPtr%devptr, stream)
  end function cudaFreeAsync_devptr

  type(cuda_devptr) function c_devloc(dev)
    type(c_ptr), intent(in) :: dev
    c_devloc = cuda_devptr(dev)
  end function c_devloc

  subroutine dev_malloc_i4(dev, host)
    integer, intent(in), target :: host(:)
    type(c_ptr), intent(out) :: dev
    integer(c_size_t) :: nbytes
    integer :: status
    nbytes = c_sizeof(host(1)) * size(host)
    status = cudaMalloc(dev, nbytes)
    if (status /= cudaSuccess) error stop 'cudaMalloc failed (integer)'
    status = cudaMemcpy(dev, c_loc(host(1)), nbytes, cudaMemcpyHostToDevice)
    if (status /= cudaSuccess) error stop 'cudaMemcpy H2D failed (integer)'
  end subroutine dev_malloc_i4

  subroutine dev_malloc_i4_lb(dev, host, lb, ub)
    integer, intent(in) :: lb, ub
    integer, intent(in), target :: host(lb:ub)
    type(c_ptr), intent(out) :: dev
    integer(c_size_t) :: nbytes
    integer :: status, n
    n = ub - lb + 1
    nbytes = c_sizeof(host(lb)) * n
    status = cudaMalloc(dev, nbytes)
    if (status /= cudaSuccess) error stop 'cudaMalloc failed (integer lb)'
    status = cudaMemcpy(dev, c_loc(host(lb)), nbytes, cudaMemcpyHostToDevice)
    if (status /= cudaSuccess) error stop 'cudaMemcpy H2D failed (integer lb)'
  end subroutine dev_malloc_i4_lb

  subroutine dev_malloc_r8(dev, host)
    real(8), intent(in), target :: host(:)
    type(c_ptr), intent(out) :: dev
    integer(c_size_t) :: nbytes
    integer :: status
    nbytes = c_sizeof(host(1)) * size(host)
    status = cudaMalloc(dev, nbytes)
    if (status /= cudaSuccess) error stop 'cudaMalloc failed (real(8))'
    status = cudaMemcpy(dev, c_loc(host(1)), nbytes, cudaMemcpyHostToDevice)
    if (status /= cudaSuccess) error stop 'cudaMemcpy H2D failed (real(8))'
  end subroutine dev_malloc_r8

  subroutine dev_malloc_r8_lb(dev, host, lb, ub)
    integer, intent(in) :: lb, ub
    real(8), intent(in), target :: host(lb:ub)
    type(c_ptr), intent(out) :: dev
    integer(c_size_t) :: nbytes
    integer :: status, n
    n = ub - lb + 1
    nbytes = c_sizeof(host(lb)) * n
    status = cudaMalloc(dev, nbytes)
    if (status /= cudaSuccess) error stop 'cudaMalloc failed (real(8) lb)'
    status = cudaMemcpy(dev, c_loc(host(lb)), nbytes, cudaMemcpyHostToDevice)
    if (status /= cudaSuccess) error stop 'cudaMemcpy H2D failed (real(8) lb)'
  end subroutine dev_malloc_r8_lb

  subroutine dev_malloc_c8(dev, host)
    complex(8), intent(in), target :: host(:)
    type(c_ptr), intent(out) :: dev
    integer(c_size_t) :: nbytes
    integer :: status
    nbytes = c_sizeof(host(1)) * size(host)
    status = cudaMalloc(dev, nbytes)
    if (status /= cudaSuccess) error stop 'cudaMalloc failed (complex(8))'
    status = cudaMemcpy(dev, c_loc(host(1)), nbytes, cudaMemcpyHostToDevice)
    if (status /= cudaSuccess) error stop 'cudaMemcpy H2D failed (complex(8))'
  end subroutine dev_malloc_c8

  subroutine dev_copy_d2h_i4(host, dev, n)
    integer, intent(in) :: n
    integer, intent(out), target :: host(n)
    type(c_ptr), intent(in) :: dev
    integer(c_size_t) :: nbytes
    integer :: status
    nbytes = c_sizeof(host(1)) * n
    status = cudaMemcpy(c_loc(host(1)), dev, nbytes, cudaMemcpyDeviceToHost)
    if (status /= cudaSuccess) error stop 'cudaMemcpy D2H failed (integer)'
  end subroutine dev_copy_d2h_i4

  subroutine dev_copy_d2h_r8(host, dev, n)
    integer, intent(in) :: n
    real(8), intent(out), target :: host(n)
    type(c_ptr), intent(in) :: dev
    integer(c_size_t) :: nbytes
    integer :: status
    nbytes = c_sizeof(host(1)) * n
    status = cudaMemcpy(c_loc(host(1)), dev, nbytes, cudaMemcpyDeviceToHost)
    if (status /= cudaSuccess) error stop 'cudaMemcpy D2H failed (real(8))'
  end subroutine dev_copy_d2h_r8

  subroutine dev_copy_d2h_c8(host, dev, n)
    integer, intent(in) :: n
    complex(8), intent(out), target :: host(n)
    type(c_ptr), intent(in) :: dev
    integer(c_size_t) :: nbytes
    integer :: status
    nbytes = c_sizeof(host(1)) * n
    status = cudaMemcpy(c_loc(host(1)), dev, nbytes, cudaMemcpyDeviceToHost)
    if (status /= cudaSuccess) error stop 'cudaMemcpy D2H failed (complex(8))'
  end subroutine dev_copy_d2h_c8

  subroutine dev_copy_h2d(dev, host)
    type(c_ptr), intent(in) :: dev
    real(8), intent(in), target :: host(:)
    integer(c_size_t) :: nbytes
    integer :: status
    nbytes = c_sizeof(host(1)) * size(host)
    status = cudaMemcpy(dev, c_loc(host(1)), nbytes, cudaMemcpyHostToDevice)
    if (status /= cudaSuccess) error stop 'cudaMemcpy H2D failed'
  end subroutine dev_copy_h2d

  subroutine dev_copy_h2d_r8_lb(dev, host, lb, ub)
    integer, intent(in) :: lb, ub
    type(c_ptr), intent(in) :: dev
    real(8), intent(in), target :: host(lb:ub)
    integer(c_size_t) :: nbytes
    integer :: status, n
    n = ub - lb + 1
    nbytes = c_sizeof(host(lb)) * n
    status = cudaMemcpy(dev, c_loc(host(lb)), nbytes, cudaMemcpyHostToDevice)
    if (status /= cudaSuccess) error stop 'cudaMemcpy H2D failed (real lb)'
  end subroutine dev_copy_h2d_r8_lb

  subroutine dev_upload_ptr_array(dev_array, host_ptrs, count)
    type(cuda_devptr), intent(in) :: host_ptrs(:)
    type(c_ptr), intent(out) :: dev_array
    integer, intent(in) :: count
    type(c_ptr), allocatable, target :: table(:)
    integer(c_size_t) :: nbytes
    integer :: status, i
    allocate(table(count))
    do i = 1, count
       table(i) = host_ptrs(i)%devptr
    end do
    nbytes = c_sizeof(table(1)) * count
    status = cudaMalloc(dev_array, nbytes)
    if (status /= cudaSuccess) error stop 'cudaMalloc failed (ptr array)'
    status = cudaMemcpy(dev_array, c_loc(table(1)), nbytes, cudaMemcpyHostToDevice)
    if (status /= cudaSuccess) error stop 'cudaMemcpy H2D failed (ptr array)'
    deallocate(table)
  end subroutine dev_upload_ptr_array

  subroutine dev_free(dev)
    type(c_ptr), intent(in) :: dev
    integer :: status
    if (.not. c_associated(dev)) return
    status = cudaFree(dev)
    if (status /= cudaSuccess) error stop 'cudaFree failed'
  end subroutine dev_free

  pure function dev_ptr_value(p) result(v)
    type(c_ptr), intent(in) :: p
    integer(c_intptr_t) :: v
    v = transfer(p, v)
  end function dev_ptr_value

end module cudautils
