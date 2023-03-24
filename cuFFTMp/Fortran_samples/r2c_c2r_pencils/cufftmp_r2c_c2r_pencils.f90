
!
! This sample illustrates the use of cufftXtSetDistribution
! to support arbitrary user data distributions in the case of an R2C - C2R transform
! 
! It performs
! - Forward FFT
! - Element wise kernel
! - Inverse FFT

module cufft_required
    integer :: planr2c, planc2r
    integer :: local_rshape_in(3), local_rshape_out(3), local_cshape_out(3)

end module cufft_required


program cufftmp_r2c_c2r_pencils
    use iso_c_binding
    use cudafor
    use cufftXt
    use cufft
    use openacc
    use mpi
    use cufft_required
    implicit none

    integer :: size, rank, ndevices, ierr
    integer :: n, nx, ny, nz ! nx slowest
    integer :: nz_complex, nz_real_padded
    integer :: i, j, k, nranks1d
    integer :: my_nx, my_ny, my_nz, ranks_cutoff, whichgpu(1)
    real, dimension(:, :, :), allocatable :: u, ref
    complex, dimension(:,:,:), allocatable :: u_permuted
    real :: max_norm, max_diff

    ! cufft-related
    integer(c_size_t) :: worksize(1)
    type(cudaLibXtDesc), pointer :: u_desc
    type(cudaXtDesc), pointer    :: u_descptr
    complex, pointer, device     :: u_dptr(:,:,:)
    integer(kind=cuda_stream_kind) :: stream

    type :: Box3D
        integer(c_long_long), dimension(0:2) :: lower
        integer(c_long_long), dimension(0:2) :: upper
        integer(c_long_long), dimension(0:2) :: strides
    end type Box3D
    type(Box3D), dimension(:), allocatable :: input_boxes, output_boxes

    call mpi_init(ierr)
    call mpi_comm_size(MPI_COMM_WORLD,size,ierr)
    call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)

    call checkCuda(cudaGetDeviceCount(ndevices))
    nranks1d = sqrt(real(size))
    if (nranks1d * nranks1d .ne. size) then
        print*,"The number of MPI ranks should be a perfect square"
        call mpi_finalize(ierr)
    end if  

    call checkCuda(cudaSetDevice(mod(rank, ndevices)))
    whichgpu(1) = mod(rank, ndevices)
    print*,"Hello from rank ", rank, " gpu id", mod(rank, ndevices), "size", size

    ! Define custom data distribution
    n = 32
    nx = n * nranks1d
    ny = nx
    nz = nx
    nz_complex = nz/2+1
    nz_real_padded = 2 * nz_complex

    ! Input data is real, distributed in X and Y, contiguous at Z
    ! Output data is complex, distributed in X and Y (picked arbitrarily)
    local_rshape_in = [nz_real_padded, n, n]
    local_cshape_out = [nz_complex, n, n]  
    local_rshape_out = [nz_real_padded, n, n]  

    if (rank == 0) then
        write(*,*) "local_rshape_in  :", local_rshape_in(1), local_rshape_in(2), local_rshape_in(3)
        write(*,*) "local_cshape_out :", local_cshape_out(1), local_cshape_out(2), local_cshape_out(3)
    end if

    ! cufftBox3d boxes are defined as { {x_start, y_start, z_start}, {x_end, y_end, z_end}, {x%strides, y%strides, z%strides} }
    ! where
    ! - {x/y/z}_{start/end} are the lower and upper corner of the boxes relatived to the global 3D box (of size nx * ny * nz)
    ! - {x/y/z} strides are the local strides. Note the use of the padded stride at z
    allocate(input_boxes(0:size-1), output_boxes(0:size-1) )
    do i = 0 , nranks1d - 1
        do j = 0, nranks1d - 1
            ! input data are pencils in X and Y, contiguous at Z
            input_boxes( i*nranks1d + j )%lower = [i*n, j*n, 0]                              !x,y,z start global Idx
            input_boxes( i*nranks1d + j )%upper = [(i+1)*n, (j+1)*n, nz]                     !x,y,z end global Idx
            input_boxes( i*nranks1d + j )%strides = [n * nz_real_padded, nz_real_padded, 1]  !x,y,z stride local
            ! output data are pencils in X and Y, picked arbitrarily
            output_boxes( i*nranks1d + j )%lower = [i*n, j*n, 0]                             !x,y,z start global Idx
            output_boxes( i*nranks1d + j )%upper = [(i+1)*n, (j+1)*n, nz_complex]            !x,y,z end global Idx
            output_boxes( i*nranks1d + j )%strides = [n * nz_complex, nz_complex, 1]         !x,y,z stride local
        end do  
    end do

	write(*,'(A18, I1, A10, 3I10)') "my rank", rank, "lower", input_boxes(rank)%lower
    write(*,'(A18, I1, A10, 3I10)') "my rank", rank, "upper", input_boxes(rank)%upper
    write(*,'(A18, I1, A10, 3I10)') "my rank", rank, "strides", input_boxes(rank)%strides

    ! Generate local, distributed data
    allocate(u(local_rshape_in(1), local_rshape_in(2), local_rshape_in(3)))
    allocate(u_permuted(local_cshape_out(1), local_cshape_out(2), local_cshape_out(3)))
    allocate(ref(local_rshape_in(1), local_rshape_in(2), local_rshape_in(3)))
    print*,'shape of u is ', shape(u)
    print*,'shape of u_permuted is ', shape(u_permuted)
    call generate_random(nz, local_rshape_in(1), local_rshape_in(2), local_rshape_in(3), u)
    ref = u
    u_permuted = (0.0,0.0)

    call checkNorm(nz, local_rshape_in(1), local_rshape_in(2), local_rshape_in(3), u, max_norm)
    print*, "initial data on ", rank, " max_norm is ", max_norm

    call checkCufft(cufftCreate(planr2c))
    call checkCufft(cufftCreate(planc2r))
    call checkCuda(cudaStreamCreate(stream))
    call checkCufft(cufftMpAttachComm(planr2c, CUFFT_COMM_MPI, MPI_COMM_WORLD), 'cufftMpAttachComm error')
    call checkCufft(cufftMpAttachComm(planc2r, CUFFT_COMM_MPI, MPI_COMM_WORLD), 'cufftMpAttachComm error')

    call checkCufft(cufftSetStream(planr2c, stream), 'cufftSetStream error')
    call checkCufft(cufftSetStream(planc2r, stream), 'cufftSetStream error')
    call checkCufft(cufftXtSetDistribution(planr2c, 3, input_boxes(rank)%lower, input_boxes(rank)%upper, &
                    output_boxes(rank)%lower, output_boxes(rank)%upper, &
                    input_boxes(rank)%strides, output_boxes(rank)%strides), 'cufftXtSetDistribution error')
    call checkCufft(cufftXtSetDistribution(planc2r, 3, input_boxes(rank)%lower, input_boxes(rank)%upper, &
                    output_boxes(rank)%lower, output_boxes(rank)%upper, &
                    input_boxes(rank)%strides, output_boxes(rank)%strides), 'cufftXtSetDistribution error')

    call checkCufft(cufftMakePlan3d(planr2c, nz, ny, nx, CUFFT_R2C, worksize), 'cufftMakePlan3d r2c error')
    call checkCufft(cufftMakePlan3d(planc2r, nz, ny, nx, CUFFT_C2R, worksize), 'cufftMakePlan3d c2r error')

    call checkCufft(cufftXtMalloc(planr2c, u_desc, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT), 'cufftXtMalloc error')
    call cufft_memcpyH2D(u_desc, u, CUFFT_XT_FORMAT_INPLACE, .true.)
    ! now reset u to make sure the check later is valid 
    u = 0.0
    
    !xxxxxxxxxxxxxxxxxxxxxxxxxx Forward 
    call checkCufft(cufftXtExecDescriptor(planr2c, u_desc, u_desc, CUFFT_FORWARD),'forward fft failed')
    call checkCuda(cudaStreamSynchronize(stream))

    ! in case we want to check the results after Forward 
    call checkCufft(cufftXtMemcpy(planr2c, u_permuted, u_desc, CUFFT_COPY_DEVICE_TO_HOST), 'permuted D2H error')
    call checkNormComplex(local_cshape_out(1), local_cshape_out(2), local_cshape_out(3), u_permuted, max_norm)
    write(*,'(A18, I1, A14, F25.8)') "after R2C ", rank, " max_norm is ", max_norm

    ! Data is now distributed as Y-Slab. We need to scale the output
    call c_f_pointer(u_desc%descriptor, u_descptr)
    call c_f_pointer(u_descptr%data(1), u_dptr, [local_cshape_out(1), local_cshape_out(2), local_cshape_out(3)])
    !$cuf kernel do (3)
    do k =1, local_cshape_out(3)
        do j = 1, local_cshape_out(2)
            do i = 1, local_cshape_out(1)
                u_dptr(i,j,k) = u_dptr(i,j,k) / real(nx*ny*nz)
            end do
        end do
    end do
    call checkCuda(cudaDeviceSynchronize())

    ! in case we want to check again after scaling 
    call checkCufft(cufftXtMemcpy(planr2c, u_permuted, u_desc, CUFFT_COPY_DEVICE_TO_HOST), 'permuted D2H error')
    call checkNormComplex(local_cshape_out(1), local_cshape_out(2), local_cshape_out(3), u_permuted, max_norm)
    write(*,'(A18, I1, A14, F25.8)') "after scaling ", rank, " max_norm is ", max_norm


    !xxxxxxxxxxxxxxxxxxxxxxxxxxxx inverse
    call checkCufft(cufftXtExecDescriptor(planc2r, u_desc, u_desc, CUFFT_INVERSE), 'inverse fft failed')
    call cufft_memcpyD2H(u, u_desc, CUFFT_XT_FORMAT_INPLACE, .true.)
    call checkCufft(cufftXtFree(u_desc))
    call checkCufft(cufftDestroy(planr2c))
    call checkCufft(cufftDestroy(planc2r))
    call checkCufft(cudaStreamDestroy(stream))

    call checkNormDiff(nz, local_rshape_in(1), local_rshape_in(2), local_rshape_in(3), u, ref, max_norm, max_diff)
    write(*,'(A18, I1, A14, F25.8, A14, F15.8)') "after C2R ", rank, " max_norm is ", max_norm, " max_diff is ", max_diff
    write(*,'(A25, I1, A14, F25.8)') "Relative Linf on rank ", rank, " is ", max_diff/max_norm
    deallocate(u)
    deallocate(ref)
    deallocate(u_permuted)
    
    call mpi_finalize(ierr)

    if(max_diff / max_norm > 1e-5) then
        print*, ">>>> FAILED on rank ", rank
        stop 1
    else 
        print*, ">>>> PASSED on rank ", rank
    end if  
    
    
contains 
    subroutine checkCuda(istat, message)
        implicit none
        integer, intent(in)                   :: istat
        character(len=*),intent(in), optional :: message
        if (istat /= cudaSuccess) then
            write(*,"('Error code: ',I0, ': ')") istat
            write(*,*) cudaGetErrorString(istat)
            if(present(message)) write(*,*) message
            call mpi_finalize(ierr)
        endif
    end subroutine checkCuda

    subroutine checkCufft(istat, message)
        implicit none
        integer, intent(in)                   :: istat
        character(len=*),intent(in), optional :: message
        if (istat /= CUFFT_SUCCESS) then
            write(*,"('Error code: ',I0, ': ')") istat
            write(*,*) cudaGetErrorString(istat)
            if(present(message)) write(*,*) message
            call mpi_finalize(ierr)
        endif
    end subroutine checkCufft

    subroutine generate_random(nz1, nz, ny, nx, data)
        implicit none
        integer, intent(in) :: nx, ny, nz, nz1
        real, dimension(nz, ny, nx), intent(out) :: data
        real :: rand(1)
        integer :: i,j,k
        do k =1, nx
            do j = 1, ny
                do i = 1, nz1
                    call random_number(rand)
                    data(i,j,k) = rand(1)
                end do
            end do
        end do
        
    end subroutine generate_random

    subroutine checkNorm(nz1, nz, ny, nx, data, max_norm)
        implicit none
        integer, intent(in)  :: nx, ny, nz, nz1
        real, dimension(nz, ny, nx), intent(in) :: data
        real :: max_norm
        integer :: i, j, k
        max_norm = 0
        do k =1, nx
            do j = 1, ny
                do i = 1, nz1
                    max_norm = max(max_norm, abs(data(i,j,k)))
                end do
            end do
        end do
    end subroutine checkNorm

    subroutine checkNormComplex(nz, ny, nx, data, max_norm)
        implicit none
        integer, intent(in)  :: nx, ny, nz
        complex, dimension(nz, ny, nx), intent(in) :: data
        real :: max_norm, max_diff
        integer :: i,j,k
        max_norm = 0
        do k =1, nx
            do j = 1, ny
                do i = 1, nz
                    max_norm = max(max_norm, abs(data(i,j,k)%re))
                    max_norm = max(max_norm, abs(data(i,j,k)%im))
                end do
            end do
        end do
    end subroutine checkNormComplex

    subroutine checkNormDiff(nz1, nz, ny, nx, data, ref, max_norm, max_diff)
        implicit none
        integer, intent(in)  :: nx, ny, nz, nz1
        real, dimension(nz, ny, nx), intent(in) :: data, ref
        real :: max_norm, max_diff
        max_norm = 0
        max_diff = 0
        do k =1, nx
            do j = 1, ny
                do i = 1, nz1
                    max_norm = max(max_norm, abs(data(i,j,k)))
                    max_diff = max(max_diff, abs(ref(i,j,k)-data(i,j,k)))
                    if (abs(ref(i,j,k)-data(i,j,k)) > 0.0001) write(*,'(A10 I3 I3 I3 A2 F18.8 A7 F18.8 A10 I2)') "diff ref[", &
                    i, j, k, "]", ref(i,j,k), "data ", data(i,j,k), " at rank ", rank
                end do
            end do
        end do
    end subroutine checkNormDiff

#ifdef ACC
    subroutine scalingData(nz, ny, nx, data, factor)
        implicit none
        integer, intent(in)  :: nx, ny, nz
        complex, dimension(nz, ny, nz) :: data
        !$acc declare deviceptr(data)
        real, intent(in)     :: factor

    !$acc parallel loop collapse(3)
        do k =1, nx
            do j = 1, ny
                do i = 1, nz
                    data(i, j, k) = data(i, j, k) / factor
                end do
            end do
        end do

    end subroutine scalingData
#endif

subroutine cufft_memcpyH2D(ulibxt, u_h, data_format, ismemcpy)
    implicit none
    type(cudaLibXtDesc), pointer, intent(out) :: ulibxt
    real, dimension(*), intent(in)          :: u_h
    integer, intent(in)                         :: data_format
    logical, intent(in)                         :: ismemcpy
    type(cudaXtDesc), pointer  :: uxt
    real, dimension(:,:,:), device, pointer :: u_d

    if(data_format == CUFFT_XT_FORMAT_INPLACE_SHUFFLED) then
      if (ismemcpy == .false.) then
        call checkCufft(cufftXtMemcpy(planc2r, ulibxt, u_h, CUFFT_COPY_HOST_TO_DEVICE), "cufft_memcpyHToD pinv Error")
      else
        call c_f_pointer(ulibxt%descriptor, uxt)
        call c_f_pointer(uxt%data(1), u_d, local_rshape_out)
        call checkCuda(cudaMemcpy(u_d, u_h, product(int(local_rshape_out,kind=8))), "cudamemcpy H2D Error")
        nullify(u_d, uxt)
      endif
    endif 

    if (data_format == CUFFT_XT_FORMAT_INPLACE) then
      if (ismemcpy == .false.) then
        call checkCufft(cufftXtMemcpy(planr2c, ulibxt, u_h, CUFFT_COPY_HOST_TO_DEVICE), "cufft_memcpyHToD pfor Error")
      else 
        call c_f_pointer(ulibxt%descriptor, uxt) 
        call c_f_pointer(uxt%data(1), u_d, local_rshape_in)
        call checkCuda(cudaMemcpy(u_d, u_h, product(int(local_rshape_in,kind=8))), "cudamemcpy H2D Error")
        nullify(u_d, uxt)
      endif
    endif 
end subroutine cufft_memcpyH2D


subroutine cufft_memcpyD2H(u_h, ulibxt, data_format,ismemcpy)
    implicit none
    type(cudaLibXtDesc), pointer, intent(in) :: ulibxt
    real, dimension(*), intent(out)      :: u_h
    integer, intent(in)                      :: data_format
    logical, intent(in)                      :: ismemcpy
    type(cudaXtDesc), pointer  :: uxt
    real, dimension(:,:,:), device, pointer :: u_d

    if(data_format == CUFFT_XT_FORMAT_INPLACE_SHUFFLED) then
      if (ismemcpy == .false.) then
        call checkCufft(cufftXtMemcpy(planr2c, u_h, ulibxt, CUFFT_COPY_DEVICE_TO_HOST), "cufft_memcpyDToH pfor Error")
      else
        call c_f_pointer(ulibxt%descriptor, uxt)
        call c_f_pointer(uxt%data(1), u_d, local_rshape_out)
        call checkCuda(cudaMemcpy(u_h, u_d, product(int(local_rshape_out,kind=8))), "cudamemcpy D2H Error")
        nullify(u_d, uxt)
      endif 
    endif
    
    if (data_format == CUFFT_XT_FORMAT_INPLACE) then
      if (ismemcpy == .false.) then
        call checkCufft(cufftXtMemcpy(planc2r, u_h, ulibxt, CUFFT_COPY_DEVICE_TO_HOST), "cufft_memcpyDToH pinv Error")
      else
        call c_f_pointer(ulibxt%descriptor, uxt)
        call c_f_pointer(uxt%data(1), u_d, local_rshape_in)
        call checkCufft(cudamemcpy(u_h, u_d, product(int(local_rshape_in,kind=8))), "cufft_memcpyD2H error")
        nullify(u_d, uxt)
      endif
    endif 
end subroutine cufft_memcpyD2H


end program cufftmp_r2c_c2r_pencils
