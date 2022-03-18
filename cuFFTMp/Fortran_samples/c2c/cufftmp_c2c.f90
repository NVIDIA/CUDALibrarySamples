
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


program cufftmp_c2c
    use iso_c_binding
    use cudafor
    use cufftXt
    use cufft
    use openacc
    use mpi
    implicit none

    integer :: size, rank, ndevices, ierr
    integer :: nx, ny, nz ! nx slowest
    integer :: i, j, k
    integer :: my_nx, my_ny, my_nz, ranks_cutoff, whichgpu(1)
    complex, dimension(:, :, :), allocatable :: u, ref, u_permuted
    real :: max_norm, max_diff

    ! cufft stuff
    integer :: plan
    integer(c_size_t) :: worksize(1)
    type(cudaLibXtDesc), pointer :: u_desc
    type(cudaXtDesc), pointer    :: u_descptr
#ifdef ACC
    complex, pointer             :: u_dptr(:,:,:)
    type(c_ptr)                  :: tmpcptr
#else
    complex, pointer, device     :: u_dptr(:,:,:)
#endif
    integer :: local_cshape(3), local_permuted_cshape(3)


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
    local_cshape = [nz, ny, my_nx]
    local_permuted_cshape = [nz, ny/size, nx]  
    if (mod(ny, size) > 0) then
        print*," ny has to divide evenly by mpi_procs"
        call mpi_finalize(ierr)
    end if 
    if (rank == 0) then
        write(*,*) "local_cshape          :", local_cshape(1), local_cshape(2), local_cshape(3)
        write(*,*) "local_permuted_cshape :", local_permuted_cshape(1), local_permuted_cshape(2), local_permuted_cshape(3)
    end if

    ! Generate local, distributed data
    allocate(u(local_cshape(1), local_cshape(2), local_cshape(3)))
    allocate(u_permuted(local_permuted_cshape(1), local_permuted_cshape(2), local_permuted_cshape(3)))
    allocate(ref(local_cshape(1), local_cshape(2), local_cshape(3)))
    call generate_random(local_cshape(1), local_cshape(2), local_cshape(3), u)
    ref = u
    u_permuted = (0.0,0.0)

    call checkNorm(local_cshape(1), local_cshape(2), local_cshape(3), u, max_norm)
    print*, "initial data on ", rank, " max_norm is ", max_norm

    call checkCufft(cufftCreate(plan))
    call checkCufft(cufftMpAttachComm(plan, CUFFT_COMM_MPI, MPI_COMM_WORLD), 'cufftMpAttachComm error')

    call checkCufft(cufftMakePlan3d(plan, nz, ny, nx, CUFFT_C2C, worksize), 'cufftMakePlan3d error')

    call checkCufft(cufftXtMalloc(plan, u_desc, CUFFT_XT_FORMAT_INPLACE), 'cufftXtMalloc error')
    call checkCufft(cufftXtMemcpy(plan, u_desc, u, CUFFT_COPY_HOST_TO_DEVICE), 'cufftXtMemcpy error')
    ! now reset u to make sure the check later is valid 
    u = (0.0,0.0)
    
    !xxxxxxxxxxxxxxxxxxxxxxxxxx Forward 
    call checkCufft(cufftXtExecDescriptor(plan, u_desc, u_desc, CUFFT_FORWARD),'forward fft failed')

    ! in case we want to check the results after Forward 
    call checkCufft(cufftXtMemcpy(plan, u_permuted, u_desc, CUFFT_COPY_DEVICE_TO_HOST), 'permuted D2H error')
    call checkNorm(local_permuted_cshape(1), local_permuted_cshape(2), local_permuted_cshape(3), u_permuted, max_norm)
    write(*,'(A18, I1, A14, F25.8)') "after C2C forward ", rank, " max_norm is ", max_norm

    ! Data is now distributed as Y-Slab. We need to scale the output
    call c_f_pointer(u_desc%descriptor, u_descptr)
    
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
    call checkCufft(cufftXtMemcpy(plan, u_permuted, u_desc, CUFFT_COPY_DEVICE_TO_HOST), 'permuted D2H error')
    call checkNorm(local_permuted_cshape(1), local_permuted_cshape(2), local_permuted_cshape(3), u_permuted, max_norm)
    write(*,'(A18, I1, A14, F25.8)') "after scaling ", rank, " max_norm is ", max_norm


    !xxxxxxxxxxxxxxxxxxxxxxxxxxxx inverse
    call checkCufft(cufftXtExecDescriptor(plan, u_desc, u_desc, CUFFT_INVERSE), 'inverse fft failed')
    call checkCufft(cufftXtMemcpy(plan, u, u_desc, CUFFT_COPY_DEVICE_TO_HOST), 'D2H failed')
    call checkCufft(cufftXtFree(u_desc))
    call checkCufft(cufftDestroy(plan))
    
    call checkNormDiff(local_cshape(1), local_cshape(2), local_cshape(3), u, ref, max_norm, max_diff)
    write(*,'(A18, I1, A14, F25.8, A14, F15.8)') "after C2C inverse ", rank, " max_norm is ", max_norm, " max_diff is ", max_diff
    write(*,'(A25, I1, A14, F25.8)') "Relative Linf on rank ", rank, " is ", max_diff/max_norm
    
    call mpi_finalize(ierr)

    if(max_diff / max_norm > 1.5e-6) then
        print*, ">>>> FAILED on rank ", rank
        stop 1
    else 
        print*, ">>>> PASSED on rank ", rank
    end if  
    
contains 
    subroutine checkCuda(istat, message)
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
        integer, intent(in)                   :: istat
        character(len=*),intent(in), optional :: message
        if (istat /= CUFFT_SUCCESS) then
            write(*,"('Error code: ',I0, ': ')") istat
            write(*,*) cudaGetErrorString(istat)
            if(present(message)) write(*,*) message
            call mpi_finalize(ierr)
        endif
    end subroutine checkCufft

    subroutine generate_random(nz, ny, nx, data)
        integer, intent(in) :: nx, ny, nz
        complex, dimension(nz, ny, nx), intent(out) :: data
        real :: rand(2)
        integer :: i,j,k
        do k =1, nx
            do j = 1, ny
                do i = 1, nz
                    call random_number(rand)
                    data(i,j,k)%re = rand(1)
                    data(i,j,k)%im = rand(2)
                end do
            end do
        end do
        
    end subroutine generate_random

    subroutine checkNorm(nz, ny, nx, data, max_norm)
        integer, intent(in)  :: nx, ny, nz
        complex, dimension(nz, ny, nx), intent(in) :: data
        real :: max_norm, max_diff
        max_norm = 0
        do k =1, nx
            do j = 1, ny
                do i = 1, nz
                    max_norm = max(max_norm, abs(data(i,j,k)%re))
                    max_norm = max(max_norm, abs(data(i,j,k)%im))
                end do
            end do
        end do
    end subroutine checkNorm

    subroutine checkNormDiff(nz, ny, nx, data, ref, max_norm, max_diff)
        integer, intent(in)  :: nx, ny, nz
        complex, dimension(nz, ny, nx), intent(in) :: data, ref
        real :: max_norm, max_diff
        max_norm = 0
        max_diff = 0
        do k =1, nx
            do j = 1, ny
                do i = 1, nz
                    max_norm = max(max_norm, abs(data(i,j,k)%re))
                    max_norm = max(max_norm, abs(data(i,j,k)%im))
                    max_diff = max(max_diff, abs(ref(i,j,k)%re-data(i,j,k)%re))
                    max_diff = max(max_diff, abs(ref(i,j,k)%im-data(i,j,k)%im))
                end do
            end do
        end do
    end subroutine checkNormDiff

#ifdef ACC
    subroutine scalingData(nz, ny, nx, data, factor)
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

end program cufftmp_c2c
