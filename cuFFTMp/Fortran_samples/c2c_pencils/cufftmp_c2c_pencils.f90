
!
! This sample illustrates the use of cufftBox3d and cufftXtSetDistribution
! to support arbitrary user data distributions.
! 
! It performs
! - Forward FFT
! - Element wise kernel
! - Inverse FFT
! 
program cufftmp_c2c_pencils
    use iso_c_binding
    use cudafor
    use cufftXt
    use cufft
    use mpi
    implicit none

    integer :: size, rank, ndevices, ierr
    integer :: n, nx, ny, nz ! nx slowest
    integer :: i, j, k, nranks1d
    integer :: my_nx, my_ny, my_nz, ranks_cutoff, whichgpu(1)
    complex, dimension(:, :, :), allocatable :: u, ref, u_permuted
    real :: max_norm, max_diff
    
    ! cufft-related
    integer :: plan
    integer(c_size_t) :: worksize(1)
    type(cudaLibXtDesc), pointer :: u_desc
    type(cudaXtDesc), pointer    :: u_descptr
    complex, pointer, device     :: u_dptr(:,:,:)
    integer(kind=cuda_stream_kind) :: stream
    integer :: local_cshape_in(3), local_cshape_out(3)
    type(cufftBox3d), dimension(:), allocatable :: input_boxes, output_boxes


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

    ! Let input data distributed in X and Y, contiguous at Z
    ! Let output data distributed in X and Z
    local_cshape_in = [nz, n, n]
    local_cshape_out = [n, ny, n]
    if (rank == 0) then
        write(*,*) "local_cshape_in (z,y,x) z fast  :", local_cshape_in(1), local_cshape_in(2), local_cshape_in(3)
        write(*,*) "local_cshape_out (z,y,x) z fast:", local_cshape_out(1), local_cshape_out(2), local_cshape_out(3)
    end if

    ! cufftBox3d boxes are defined as { {x_start, y_start, z_start}, {x_end, y_end, z_end}, {x%strides, y%strides, z%strides} }
    ! where
    ! - {x/y/z}_{start/end} are the lower and upper corner of the boxes relatived to the global 3D box (of size nx * ny * nz)
    ! - {x/y/z} strides are the local strides
    allocate(input_boxes(0:size-1), output_boxes(0:size-1) )
    do i = 0 , nranks1d - 1
        do j = 0, nranks1d - 1
            ! input data are pencils in X and Y, contiguous at Z
            input_boxes( i*nranks1d + j )%lower = [i*n, j*n, 0]                !x,y,z start global Idx
            input_boxes( i*nranks1d + j )%upper = [(i+1)*n, (j+1)*n, nz]       !x,y,z end global Idx
            input_boxes( i*nranks1d + j )%strides = [n * nz, nz, 1]            !x,y,z stride local

            ! output data are pencils in X and Z
            output_boxes( i*nranks1d + j )%lower = [i*n, 0, j*n]              !x,y,z start global Idx
            output_boxes( i*nranks1d + j )%upper = [(i+1)*n, ny, (j+1)*n]     !x,y,z end global Idx
            output_boxes( i*nranks1d + j )%strides = [n * ny, n, 1]           !x,y,z stride local
        end do  
    end do

	write(*,'(A18, I1, A10, 3I10)') "my rank", rank, "lower", input_boxes(rank)%lower
    write(*,'(A18, I1, A10, 3I10)') "my rank", rank, "upper", input_boxes(rank)%upper
    write(*,'(A18, I1, A10, 3I10)') "my rank", rank, "strides", input_boxes(rank)%strides

    ! Generate local, distributed CPU data
    allocate(u(local_cshape_in(1), local_cshape_in(2), local_cshape_in(3)))
    allocate(u_permuted(local_cshape_out(1), local_cshape_out(2), local_cshape_out(3)))
    allocate(ref(local_cshape_in(1), local_cshape_in(2), local_cshape_in(3)))
    call generate_random(local_cshape_in(1), local_cshape_in(2), local_cshape_in(3), u)
    ref = u
    u_permuted = (0.0,0.0)

    call checkNorm(local_cshape_in(1), local_cshape_in(2), local_cshape_in(3), u, max_norm)
    print*, "initial data on ", rank, " max_norm is ", max_norm

    call checkCufft(cufftCreate(plan))
    call checkCuda(cudaStreamCreate(stream))
    call checkCufft(cufftMpAttachComm(plan, CUFFT_COMM_MPI, MPI_COMM_WORLD), 'cufftMpAttachComm error')
    call checkCufft(cufftXtSetDistribution(plan, input_boxes(rank), output_boxes(rank)))

    call checkCufft(cufftMakePlan3d(plan, nz, ny, nx, CUFFT_C2C, worksize), 'cufftMakePlan3d error')
    call checkCufft(cufftSetStream(plan, stream), 'cufftSetStream error')

    call checkCufft(cufftXtMalloc(plan, u_desc, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT), 'cufftXtMalloc error')
    call checkCufft(cufftXtMemcpy(plan, u_desc, u, CUFFT_COPY_HOST_TO_DEVICE), 'cufftXtMemcpy error')
    ! now reset u to make sure the check later is valid 
    u = (0.0,0.0)
    
    !xxxxxxxxxxxxxxxxxxxxxxxxxx Forward 
    call checkCufft(cufftXtExecDescriptor(plan, u_desc, u_desc, CUFFT_FORWARD),'forward fft failed')
    call checkCuda(cudaStreamSynchronize(stream))

    ! in case we want to check the results after Forward 
    call checkCufft(cufftXtMemcpy(plan, u_permuted, u_desc, CUFFT_COPY_DEVICE_TO_HOST), 'permuted D2H error')
    call checkNorm(local_cshape_out(1), local_cshape_out(2), local_cshape_out(3), u_permuted, max_norm)
    write(*,'(A18, I1, A14, F25.8)') "after C2C forward ", rank, " max_norm is ", max_norm

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
    call checkCufft(cufftXtMemcpy(plan, u_permuted, u_desc, CUFFT_COPY_DEVICE_TO_HOST), 'permuted D2H error')
    call checkNorm(local_cshape_out(1), local_cshape_out(2), local_cshape_out(3), u_permuted, max_norm)
    write(*,'(A18, I1, A14, F25.8)') "after scaling ", rank, " max_norm is ", max_norm


    !xxxxxxxxxxxxxxxxxxxxxxxxxxxx inverse
    call checkCufft(cufftXtExecDescriptor(plan, u_desc, u_desc, CUFFT_INVERSE), 'inverse fft failed')
    call checkCufft(cufftXtMemcpy(plan, u, u_desc, CUFFT_COPY_DEVICE_TO_HOST), 'D2H failed')
    call checkCufft(cufftXtFree(u_desc))
    call checkCufft(cufftDestroy(plan))
    call checkCufft(cudaStreamDestroy(stream))
    
    call checkNormDiff(local_cshape_in(1), local_cshape_in(2), local_cshape_in(3), u, ref, max_norm, max_diff)
    write(*,'(A18, I1, A14, F25.8, A14, F15.8)') "after C2C inverse ", rank, " max_norm is ", max_norm, " max_diff is ", max_diff
    write(*,'(A25, I1, A14, F25.8)') "Relative Linf on rank ", rank, " is ", max_diff/max_norm
    
    deallocate(input_boxes, output_boxes)
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

end program cufftmp_c2c_pencils
