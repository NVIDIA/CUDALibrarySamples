# Building bootstraps for other versions of MPI

cuFFTMp ships with an NVSHMEM bootstrap designed for OpenMPI 3 and 4.
However, you can easily build you own bootstrap, compatible with another version of OpenMPI or another MPI implementation. To do so,
```
MPICC=mpicc DEST=myMPI make mpi_bootstrap
```
will download NVSHMEM, build the bootstrap library and place it in the `myMPI` folder.

After this, you can run any sample by
```
MPI_HOME=/path/to/my/mpi/ NVSHMEM_LIB="../../extra_bootstraps/myMPI" make run
```
which effectively place `myMPI` in your `LD_LIBRARY_PATH`.
