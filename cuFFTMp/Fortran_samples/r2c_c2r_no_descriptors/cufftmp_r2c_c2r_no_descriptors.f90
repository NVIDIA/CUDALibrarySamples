
!This samples illustrates an example of distributed R2C-C2R 3D FFT computation,
! by performing a FORWARD R2C FFT -> element-wise transform -> INVERSE C2R FFT, without multi-GPU descriptors.

program cufftmp_r2c_c2r_no_descriptors
   use iso_c_binding
   use nvshmem
   use cufftXt
   use cufft
   use mpi
   implicit none

   integer :: size, rank, ndevices, ierr
   integer :: nx, ny, nz ! nx slowest
   integer :: i, j, k
   integer :: my_nx, my_ny, my_nz, ranks_cutoff
   real, dimension(:, :, :), allocatable :: u, ref
   real :: max_norm, max_diff

   ! cufft stuff
   integer :: planr2c, planc2r
   integer(c_size_t) :: worksize(1)
   type(c_devptr)                :: tmp_cptr
   real, pointer, device, contiguous         :: u_real_dptr(:,:,:)
   integer :: local_rshape(3), local_permuted_cshape(3)

   ! nvshmem
   type(nvshmemx_status) :: nvstat

   call mpi_init(ierr)
   call mpi_comm_size(MPI_COMM_WORLD,size,ierr)
   call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)

   call checkCuda(cudaGetDeviceCount(ndevices))
   call checkCuda(cudaSetDevice(mod(rank, ndevices)))
   print*,"Hello from rank ", rank, " gpu id", mod(rank, ndevices), "total MPI rank", size

   nvstat = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, MPI_COMM_WORLD)
   if (nvstat%member /= 0) then
      write(*,"('nvshmemx_init_attr Error code: ',I0, ': ')") nvstat%member
      call mpi_finalize(ierr)
   endif

   nx = 128
   ny = nx / 2
   nz = nx * 2

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
   allocate(ref(local_rshape(1), local_rshape(2), local_rshape(3)))
   call generate_random(nz, local_rshape(1), local_rshape(2), local_rshape(3), u)
   ref = u

   ! allocate device pointer u_real_dptr
   tmp_cptr = nvshmem_malloc(product(local_rshape)*4)
   call c_f_pointer(tmp_cptr, u_real_dptr, [local_rshape(1), local_rshape(2), local_rshape(3)])
   call checkCuda(cudaMemcpy(u_real_dptr, u, product(local_rshape)))

   call checkNorm(nz, local_rshape(1), local_rshape(2), local_rshape(3), u, max_norm)
   print*, "initial data on ", rank, " max_norm is ", max_norm

   ! now reset u to make sure the check later is valid
   u = (0.0,0.0)

   call checkCufft(cufftCreate(planr2c))
   call checkCufft(cufftCreate(planc2r))
   call checkCufft(cufftMpAttachComm(planr2c, CUFFT_COMM_MPI, MPI_COMM_WORLD), 'cufftMpAttachComm planr2c error')
   call checkCufft(cufftMpAttachComm(planc2r, CUFFT_COMM_MPI, MPI_COMM_WORLD), 'cufftMpAttachComm planc2r error')

   call checkCufft(cufftXtSetSubformatDefault(planr2c, CUFFT_XT_FORMAT_INPLACE, CUFFT_XT_FORMAT_INPLACE_SHUFFLED), 'cufftXtSetSubformatDefault r2c error')
   call checkCufft(cufftXtSetSubformatDefault(planc2r, CUFFT_XT_FORMAT_INPLACE, CUFFT_XT_FORMAT_INPLACE_SHUFFLED), 'cufftXtSetSubformatDefault c2r error')

   call checkCufft(cufftMakePlan3d(planr2c, nx, ny, nz, CUFFT_R2C, worksize), 'cufftMakePlan3d r2c error')
   call checkCufft(cufftMakePlan3d(planc2r, nx, ny, nz, CUFFT_C2R, worksize), 'cufftMakePlan3d c2r error')

   !xxxxxxxxxxxxxxxxxxxxxxxxxx Forward
   ! Fortran interface for ExecR2C/C2R accepts either real or complex data for input/output
   call checkCufft(cufftExecR2C(planr2c, u_real_dptr, u_real_dptr),'forward fft failed')

   ! Data is now distributed as Y-Slab. We need to scale the output
   !$cuf kernel do (3)
   do k =1, local_rshape(3)
      do j = 1, local_rshape(2)
         do i = 1, local_rshape(1)
            u_real_dptr(i,j,k) = u_real_dptr(i,j,k) / real(nx*ny*nz)
         end do
      end do
   end do
   call checkCuda(cudaDeviceSynchronize())

   !xxxxxxxxxxxxxxxxxxxxxxxxxxxx inverse
   call checkCufft(cufftExecC2R(planc2r, u_real_dptr, u_real_dptr), 'inverse fft failed')

   call checkCuda(cudaMemcpy(u, u_real_dptr, product(local_rshape)))

   call checkCufft(cufftDestroy(planr2c))
   call checkCufft(cufftDestroy(planc2r))

   call checkNormDiff(nz, local_rshape(1), local_rshape(2), local_rshape(3), u, ref, max_norm, max_diff)
   write(*,'(A18, I1, A14, F25.8, A14, F15.8)') "after C2C inverse ", rank, " max_norm is ", max_norm, " max_diff is ", max_diff
   write(*,'(A25, I1, A14, F25.8)') "Relative Linf on rank ", rank, " is ", max_diff/max_norm

   deallocate(u)
   deallocate(ref)
   call nvshmem_free(u_real_dptr)
   call nvshmem_finalize()
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
         write(*,"('checkCuda Error code: ',I0, ': ')") istat
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
            end do
         end do
      end do
   end subroutine checkNorm


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


end program cufftmp_r2c_c2r_no_descriptors
