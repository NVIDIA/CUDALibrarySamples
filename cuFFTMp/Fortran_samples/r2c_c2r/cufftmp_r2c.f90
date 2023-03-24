
!This samples illustrates a basic use of cuFFTMp using the built-in, optimized, data distributions.
!  
!  It assumes the CPU data is initially distributed according to CUFFT_XT_FORMAT_INPLACE, a.k.a. X-Slabs.
!  Given a global array of size X!  Y!  Z, every MPI rank owns approximately (X / ngpus)!  Y*Z entries.
!  More precisely, 
!  - The first (ngpus % X) MPI rank each own (X / ngpus + 1) planes of size Y * Z, 
!  - The remaining MPI rank each own (X / ngpus) planes of size Y*Z
!  
!  The CPU data is then copied on GPU and a forward transform is applied.
!  
!  After that transform, GPU data is distributed according to CUFFT_XT_FORMAT_INPLACE_SHUFFLED, a.k.a. Y-Slabs.
!  Given a global array of size X * Y * Z, every MPI rank owns approximately X * (Y / ngpus) * Z entries.
!  More precisely, 
!  - The first (ngpus % Y) MPI rank each own (Y / ngpus + 1) planes of size X * Z, 
!  - The remaining MPI rank each own (Y / ngpus) planes of size X * Z
!  
!  A scaling kerel is applied, on the distributed GPU data (distributed according to CUFFT_XT_FORMAT_INPLACE)
!  This kernel prints some elements to illustrate the CUFFT_XT_FORMAT_INPLACE_SHUFFLED data distribution and
!  normalize entries by (nx * ny * nz)
!  
!  Finally, a backward transform is applied.
!  After this, data is again distributed according to CUFFT_XT_FORMAT_INPLACE, same as the input data.
!  
!  Data is finally copied back to CPU and compared to the input data. They should be almost identical.
module cufft_required
    integer :: planr2c, planc2r
    integer :: local_rshape(3), local_rshape_permuted(3), local_permuted_cshape(3)

end module cufft_required


program cufftmp_r2c
    use iso_c_binding
    use cudafor
    use cufftXt
    use cufft
    use openacc
    use mpi
    use cufft_required
    implicit none

    integer :: size, rank, ndevices, ierr
    integer :: nx, ny, nz ! nx slowest
    integer :: i, j, k
    integer :: my_nx, my_ny, my_nz, ranks_cutoff, whichgpu(1)
    real, dimension(:, :, :), allocatable :: u, ref
    complex, dimension(:,:,:), allocatable :: u_permuted
    real :: max_norm, max_diff

    ! cufft stuff
    integer(c_size_t) :: worksize(1)
    type(cudaLibXtDesc), pointer :: u_desc
    type(cudaXtDesc), pointer    :: u_descptr
#ifdef ACC
    complex, pointer             :: u_dptr(:,:,:)
    type(c_ptr)                  :: tmpcptr
#else
    complex, pointer, device     :: u_dptr(:,:,:)
#endif
    integer(kind=cuda_stream_kind) :: stream

    call mpi_init(ierr)
    call mpi_comm_size(MPI_COMM_WORLD,size,ierr)
    call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)

    call checkCuda(cudaGetDeviceCount(ndevices))
    call checkCuda(cudaSetDevice(mod(rank, ndevices)))
    whichgpu(1) = mod(rank, ndevices)
    
    print*,"Hello from rank ", rank, " gpu id", mod(rank, ndevices), "size", size

    nx = 256
    ny = nx
    nz = nx

    ! We start with X-Slabs
    ! Ranks 0 ... (nx % size - 1) have 1 more element in the X dimension
    ! and every rank own all elements in the Y and Z dimensions.
    ranks_cutoff = mod(nx, size)
    my_nx = nx / size 
    if (rank < ranks_cutoff) my_nx = my_nx + 1
    my_ny =  ny;
    my_nz =  nz;
    local_rshape = [2*(nz/2+1), ny, my_nx]
    local_permuted_cshape = [nz/2+1, ny/size, nx]  
    local_rshape_permuted = [2*(nz/2+1), ny/size, nx]  
    if (mod(ny, size) > 0) then
        print*," ny has to divide evenly by mpi_procs"
        call mpi_finalize(ierr)
    end if 
    if (rank == 0) then
        write(*,*) "local_rshape          :", local_rshape(1), local_rshape(2), local_rshape(3)
        write(*,*) "local_permuted_cshape :", local_permuted_cshape(1), local_permuted_cshape(2), local_permuted_cshape(3)
    end if

    ! Generate local, distributed data
    allocate(u(local_rshape(1), local_rshape(2), local_rshape(3)))
    allocate(u_permuted(local_permuted_cshape(1), local_permuted_cshape(2), local_permuted_cshape(3)))
    allocate(ref(local_rshape(1), local_rshape(2), local_rshape(3)))
    print*,'shape of u is ', shape(u)
    print*,'shape of u_permuted is ', shape(u_permuted)
    call generate_random(nz, local_rshape(1), local_rshape(2), local_rshape(3), u)
    ref = u
    u_permuted = (0.0,0.0)

    call checkNorm(nz, local_rshape(1), local_rshape(2), local_rshape(3), u, max_norm)
    print*, "initial data on ", rank, " max_norm is ", max_norm

    call checkCufft(cufftCreate(planr2c))
    call checkCufft(cufftCreate(planc2r))
    !call checkCuda(cudaStreamCreate(stream))
    call checkCufft(cufftMpAttachComm(planr2c, CUFFT_COMM_MPI, MPI_COMM_WORLD), 'cufftMpAttachComm error')
    call checkCufft(cufftMpAttachComm(planc2r, CUFFT_COMM_MPI, MPI_COMM_WORLD), 'cufftMpAttachComm error')

    call checkCufft(cufftMakePlan3d(planr2c, nz, ny, nx, CUFFT_R2C, worksize), 'cufftMakePlan3d r2c error')
    call checkCufft(cufftMakePlan3d(planc2r, nz, ny, nx, CUFFT_C2R, worksize), 'cufftMakePlan3d c2r error')
    !call checkCufft(cufftSetStream(plan, stream), 'cufftSetStream error')

    call checkCufft(cufftXtMalloc(planr2c, u_desc, CUFFT_XT_FORMAT_INPLACE), 'cufftXtMalloc error')
    !call checkCufft(cufftXtMemcpy(planr2c, u_desc, u, CUFFT_COPY_HOST_TO_DEVICE), 'cufftXtMemcpy error')
    call cufft_memcpyH2D(u_desc, u, CUFFT_XT_FORMAT_INPLACE, .true.)
    ! now reset u to make sure the check later is valid 
    u = 0.0
    
    !xxxxxxxxxxxxxxxxxxxxxxxxxx Forward 
    call checkCufft(cufftXtExecDescriptor(planr2c, u_desc, u_desc, CUFFT_FORWARD),'forward fft failed')

    ! in case we want to check the results after Forward 
    !call checkCufft(cufftXtMemcpy(planr2c, u_permuted, u_desc, CUFFT_COPY_DEVICE_TO_HOST), 'permuted D2H error')
    !call checkNormComplex(local_permuted_cshape(1), local_permuted_cshape(2), local_permuted_cshape(3), u_permuted, max_norm)
    !write(*,'(A18, I1, A14, F25.8)') "after R2C ", rank, " max_norm is ", max_norm

    ! Data is now distributed as Y-Slab. We need to scale the output
    call c_f_pointer(u_desc%descriptor, u_descptr)
    
    ! print*,'doris rank ', rank, u_descptr%nGPUs, " " , u_descptr%GPUs(0)
#ifndef ACC
    call c_f_pointer(u_descptr%data(1), u_dptr, [local_permuted_cshape(1), local_permuted_cshape(2), local_permuted_cshape(3)])
    !$cuf kernel do (3)
    do k =1, local_permuted_cshape(3)
        do j = 1, local_permuted_cshape(2)
            do i = 1, local_permuted_cshape(1)
                u_dptr(i,j,k) = u_dptr(i,j,k) / real(nx*ny*nz)
            end do
        end do
    end do
    call checkCuda(cudaDeviceSynchronize())
#else
    tmpcptr = transfer(u_descptr%data(1), tmpcptr)
    call c_f_pointer(tmpcptr, u_dptr, [local_permuted_cshape(1), local_permuted_cshape(2), local_permuted_cshape(3)])
    
    call scalingData(local_permuted_cshape(1), local_permuted_cshape(2), local_permuted_cshape(3), u_dptr, real(nx*ny*nz))
#endif

    ! in case we want to check again after scaling 
    call checkCufft(cufftXtMemcpy(planr2c, u_permuted, u_desc, CUFFT_COPY_DEVICE_TO_HOST), 'permuted D2H error')
    call checkNormComplex(local_permuted_cshape(1), local_permuted_cshape(2), local_permuted_cshape(3), u_permuted, max_norm)
    write(*,'(A18, I1, A14, F25.8)') "after scaling ", rank, " max_norm is ", max_norm


    !xxxxxxxxxxxxxxxxxxxxxxxxxxxx inverse
    call checkCufft(cufftXtExecDescriptor(planc2r, u_desc, u_desc, CUFFT_INVERSE), 'inverse fft failed')
    !call checkCufft(cufftXtMemcpy(planc2r, u, u_desc, CUFFT_COPY_DEVICE_TO_HOST), 'D2H failed')
    call cufft_memcpyD2H(u, u_desc, CUFFT_XT_FORMAT_INPLACE, .true.)
    call checkCufft(cufftXtFree(u_desc))
    call checkCufft(cufftDestroy(planr2c))
    call checkCufft(cufftDestroy(planc2r))
    !call checkCufft(cudaStreamDestroy(stream))

    call checkNormDiff(nz, local_rshape(1), local_rshape(2), local_rshape(3), u, ref, max_norm, max_diff)
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
        !call random_seed(put=(/seed, seed+1/))
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
                    !write(*,'(A10 I3 I3 I3 A2 F18.8 A10 I2)') "ref[", &
                    !i, j, k, "]", ref(i,j,k), " at rank ", rank
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
        call c_f_pointer(uxt%data(1), u_d, local_rshape_permuted)
        call checkCuda(cudaMemcpy(u_d, u_h, product(int(local_rshape_permuted,kind=8))), "cudamemcpy H2D Error")
        nullify(u_d, uxt)
      endif
    endif 

    if (data_format == CUFFT_XT_FORMAT_INPLACE) then
      if (ismemcpy == .false.) then
        call checkCufft(cufftXtMemcpy(planr2c, ulibxt, u_h, CUFFT_COPY_HOST_TO_DEVICE), "cufft_memcpyHToD pfor Error")
      else 
        call c_f_pointer(ulibxt%descriptor, uxt) 
        call c_f_pointer(uxt%data(1), u_d, local_rshape)
        call checkCuda(cudaMemcpy(u_d, u_h, product(int(local_rshape,kind=8))), "cudamemcpy H2D Error")
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
        call c_f_pointer(uxt%data(1), u_d, local_rshape_permuted)
        call checkCuda(cudaMemcpy(u_h, u_d, product(int(local_rshape_permuted,kind=8))), "cudamemcpy D2H Error")
        nullify(u_d, uxt)
      endif 
    endif
    
    if (data_format == CUFFT_XT_FORMAT_INPLACE) then
      if (ismemcpy == .false.) then
        call checkCufft(cufftXtMemcpy(planc2r, u_h, ulibxt, CUFFT_COPY_DEVICE_TO_HOST), "cufft_memcpyDToH pinv Error")
      else
        call c_f_pointer(ulibxt%descriptor, uxt)
        call c_f_pointer(uxt%data(1), u_d, local_rshape)
        call checkCufft(cudamemcpy(u_h, u_d, product(int(local_rshape,kind=8))), "cufft_memcpyD2H error")
        nullify(u_d, uxt)
      endif
    endif 
end subroutine cufft_memcpyD2H


end program cufftmp_r2c
