# Building bootstraps for other versions of MPI

cuFFTMp uses NVSHMEM. In order to interoperate with MPI, a bootstrap plugin is required. NVSHMEM ships with a bootstrap compatible with HPC-X.
However, you can easily build you own bootstrap, compatible with another MPI implementation. To do so,
```
MPI_HOME=/path/to/mpi/home/ CUDA_HOME=/path/to/cuda/home DEST=myMPI make mpi_bootstrap
```
will download NVSHMEM, build the bootstrap library and place it in the `myMPI` folder.

After this, you can run any sample by
```
MPI_HOME=/path/to/my/mpi/ NVSHMEM_LIB="../../extra_bootstraps/myMPI" make run
```
which effectively place `myMPI` in your `LD_LIBRARY_PATH`.
