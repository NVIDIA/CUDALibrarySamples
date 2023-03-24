! This sample shows how to use cuFFTMp reshape APIs to re-distribute data across GPUs.
program cufftmp_reshape
    use iso_c_binding
    use cudafor
    use cufftXt
    use cufft
    use mpi
    use nvshmem
    implicit none

    integer :: size, rank, ndevices, ierr, err
    integer :: nx, ny, nz, nsize, i ! nx slowest
    integer, dimension(:), allocatable :: dst_host_expected
    integer, dimension(:), allocatable :: dst_host, src_host
    
    ! cufft-related
    type(c_ptr) :: reshapeHandle
    integer(c_size_t) :: worksize(1), dataSize, elementSize
    integer(kind=cuda_stream_kind) :: stream
    type(c_devptr)  :: workArea, dst, src

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
    call checkCuda(cudaSetDevice(mod(rank, ndevices)))
    print*,"Hello from rank ", rank, " gpu id", mod(rank, ndevices), "size", size
    
    !  Assume we have a 4x4 "global" world with the following data
    !      ------------> fast
    !    | [0  1  2  3 ]
    !    | [4  5  6  7 ]
    !    | [8  9  10 11]
    !    | [12 13 14 15]
    ! 
    ! 
    !  Initially, data is distributed as follow
    !  [0  1   | 2  3  ]
    !  [4  5   | 6  7  ]
    !  [8  9   | 10 11 ]
    !  [12 13  | 14 15 ]
    !  ^rank 0^ ^rank 1^
    !  where every rank owns part of it, stored in row major format
    ! 
    !  and we wish to redistribute it as
    !  rank 0 [0  1   2  3  ]
    !         [4  5   6  7  ]
    !         ---------------
    !  rank 1 [8  9   10 11 ]
    !         [12 13  14 15 ]
    !  where every rank should own part of it, stored in row major format
    ! 
    !  To do so, we describe the data using boxes
    !  where every box is described as low-high, where
    !  low and high are the lower and upper corners, respectively
    ! 
    !  - The "world" box is of size 4x4
    !  - The input boxes are  [ (0,0)-(4,2), (0,2)--(4,4) ]
    !  - The output boxes are [ (0,0)-(2,4), (2,0)--(4,4) ]
    ! 
    !  Since our data is 2D, in the following, the first dimension (z dim) is always 0-1

    nx = 1
    ny = 4
    nz = 4
    nsize = nx*ny*nz/size

    ! cufftBox3d boxes are defined as { {x_start, y_start, z_start}, {x_end, y_end, z_end}, {x%strides, y%strides, z%strides} }
    ! where
    ! - {x/y/z}_{start/end} are the lower and upper corner of the boxes relatived to the global 3D box (of size nx * ny * nz)
    ! - Note 0-indexing as wrapping from C functions
    ! - {x/y/z} strides are the local strides
    allocate(input_boxes(0:size-1), output_boxes(0:size-1) )
    
    input_boxes(0)%lower = [0, 0, 0]             !x,y,z start global Idx
    input_boxes(0)%upper = [1, 4, 2]             !x,y,z end global Idx
    input_boxes(0)%strides = [8, 2, 1]           !x,y,z stride local
    input_boxes(1)%lower = [0, 0, 2]             
    input_boxes(1)%upper = [1, 4, 4]             
    input_boxes(1)%strides = [8, 2, 1]         
    
    output_boxes(0)%lower = [0, 0, 0]             !x,y,z start global Idx
    output_boxes(0)%upper = [1, 2, 4]             !x,y,z end global Idx
    output_boxes(0)%strides = [8, 4, 1]           !x,y,z stride local
    output_boxes(1)%lower = [0, 2, 0]             
    output_boxes(1)%upper = [1, 4, 4]             
    output_boxes(1)%strides = [8, 4, 1]

    allocate(src_host(nsize), dst_host_expected(nsize), dst_host(nsize))
    if (rank == 0) then
        src_host = [0, 1, 4, 5, 8,  9,  12, 13]
        dst_host_expected = [0, 1, 2, 3, 4, 5, 6, 7]
    else
        src_host = [2, 3, 6, 7, 10, 11, 14, 15]
        dst_host_expected = [8, 9, 10, 11, 12, 13, 14, 15]
    endif   

    write(*,'(A30, I1, A1, 8I3)') "Input data on rank ", rank, ":", src_host
    write(*,'(A30, I1, A1, 8I3)') "Expected output data on rank ", rank, ":", dst_host_expected

    call checkCufft(cufftMpCreateReshape(reshapeHandle))
    call checkCufft(cufftMpAttachReshapeComm(reshapeHandle, CUFFT_COMM_MPI, MPI_COMM_WORLD), 'cufftMpAttachReshapeComm error')

    call checkCuda(cudaStreamCreate(stream))

    elementSize = 4  ! size of integer
    call checkCufft(cufftMpMakeReshape(reshapeHandle, elementSize, 3, input_boxes(rank)%lower, input_boxes(rank)%upper, &
                    output_boxes(rank)%lower, output_boxes(rank)%upper, &
                    input_boxes(rank)%strides, output_boxes(rank)%strides),  'cufftMpMakeReshape error')

    ! Optional: For now workArea is not needed
    call checkCufft(cufftMpGetReshapeSize(reshapeHandle, worksize), 'cufftMpGetReshapeSize error')

    ! Move CPU data to GPU
    dataSize = nsize*4
    src = nvshmem_malloc(dataSize)
    dst = nvshmem_malloc(dataSize)
    call checkCuda(cudaMemcpy(src, c_loc(src_host), dataSize, cudaMemcpyHostToDevice))

    ! Apply reshape
    call checkCufft(cufftMpExecReshapeAsync(reshapeHandle, dst, src, workArea, stream), 'cufftMpExecReshapeAsync error')

    ! Move GPU data to CPU
    call checkCuda(cudaStreamSynchronize(stream), 'cudaStreamSynchroize error')
    
    dst_host=0
    call checkCuda(cudaMemcpy(c_loc(dst_host), dst, dataSize, cudaMemcpyDeviceToHost), 'cudaMemcpy error')
    write(*,'(A30, I1, A1, 8I3)') "  Output data on rank ", rank, ":", dst_host
    call nvshmem_free(src)
    call nvshmem_free(dst)
    call checkCufft(cufftMpDestroyReshape(reshapeHandle))
    call checkCuda(cudaStreamDestroy(stream))
    

    err = 0
    do i=1, nsize
        if (dst_host(i) .ne. dst_host_expected(i)) then
            print*,"differ ", dst_host(i), dst_host_expected(i)
            err = err + 1
        endif
    enddo
    
    deallocate(input_boxes, output_boxes)
    deallocate(dst_host, dst_host_expected, src_host)
    call mpi_finalize(ierr)

    if(err > 0) then
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


end program cufftmp_reshape
