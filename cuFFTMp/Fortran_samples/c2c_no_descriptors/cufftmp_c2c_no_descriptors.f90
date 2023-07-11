
!This samples illustrates a simple example of complex-to-complex distributed FFT computation,
! by performing a FORWARD FFT -> element-wise transform -> INVERSE FFT, without multi-GPU descriptors.

program cufftmp_c2c_no_descriptors
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
   complex, dimension(:, :, :), allocatable :: u, ref
   real :: max_norm, max_diff

   ! cufft stuff
   integer :: plan
   integer(c_size_t) :: worksize(1)
   type(c_devptr)                :: tmp_cptr
   complex, pointer, device      :: u_dptr(:,:,:)
   integer :: local_cshape(3), local_permuted_cshape(3)

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
   nz = 256 

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
   allocate(ref(local_cshape(1), local_cshape(2), local_cshape(3)))
   call generate_random(local_cshape(1), local_cshape(2), local_cshape(3), u)
   ref = u

   ! allocate device pointer u_dptr
   tmp_cptr = nvshmem_malloc(product(local_cshape)*8)
   call c_f_pointer(tmp_cptr, u_dptr, [local_cshape(1), local_cshape(2), local_cshape(3)])
   call checkCuda(cudaMemcpy(u_dptr, u, product(local_cshape)))

   call checkNorm(local_cshape(1), local_cshape(2), local_cshape(3), u, max_norm)
   print*, "initial data on ", rank, " max_norm is ", max_norm

   ! now reset u to make sure the check later is valid
   u = (0.0,0.0)

   call checkCufft(cufftCreate(plan))
   call checkCufft(cufftMpAttachComm(plan, CUFFT_COMM_MPI, MPI_COMM_WORLD), 'cufftMpAttachComm error')

   call checkCufft(cufftXtSetSubformatDefault(plan, CUFFT_XT_FORMAT_INPLACE, CUFFT_XT_FORMAT_INPLACE_SHUFFLED), 'cufftXtSetSubformatDefault error')

   call checkCufft(cufftMakePlan3d(plan, nx, ny, nz, CUFFT_C2C, worksize), 'cufftMakePlan3d error')

   !xxxxxxxxxxxxxxxxxxxxxxxxxx Forward
   call checkCufft(cufftExecC2C(plan, u_dptr, u_dptr, CUFFT_FORWARD),'forward fft failed')

   ! Data is now distributed as Y-Slab. We need to scale the output
   !$cuf kernel do (3)
   do k =1, local_permuted_cshape(3)
      do j = 1, local_permuted_cshape(2)
         do i = 1, local_permuted_cshape(1)
            u_dptr(i,j,k) = u_dptr(i,j,k) / real(nx*ny*nz)
         end do
      end do
   end do
   call checkCuda(cudaDeviceSynchronize())

   !xxxxxxxxxxxxxxxxxxxxxxxxxxxx inverse
   call checkCufft(cufftExecC2C(plan, u_dptr, u_dptr, CUFFT_INVERSE), 'inverse fft failed')

   call checkCuda(cudaMemcpy(u, u_dptr, product(local_cshape)))

   call checkCufft(cufftDestroy(plan))

   call checkNormDiff(local_cshape(1), local_cshape(2), local_cshape(3), u, ref, max_norm, max_diff)
   write(*,'(A18, I1, A14, F25.8, A14, F15.8)') "after C2C inverse ", rank, " max_norm is ", max_norm, " max_diff is ", max_diff
   write(*,'(A25, I1, A14, F25.8)') "Relative Linf on rank ", rank, " is ", max_diff/max_norm

   deallocate(u)
   deallocate(ref)
   call nvshmem_free(u_dptr)
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


end program cufftmp_c2c_no_descriptors
