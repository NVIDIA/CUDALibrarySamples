DEST ?= "myMPI"
MPICC ?= mpicc

all: mpi_bootstrap

${DEST}/nvshmem_bootstrap_mpi.so:
	mkdir -p ${DEST}
	wget https://developer.download.nvidia.com/compute/redist/nvshmem/2.4.1/source/nvshmem_src_2.4.1-3.txz
	tar Jxvf nvshmem_src_2.4.1-3.txz
	mpicc nvshmem_src_2.4.1-3/src/bootstrap/bootstrap_mpi.c -shared -o ${DEST}/nvshmem_bootstrap_mpi.so -fPIC -Invshmem_src_2.4.1-3/src/include/
	echo "Bootstrap built!"
	ldd ${DEST}/nvshmem_bootstrap_mpi.so

mpi_bootstrap: ${DEST}/nvshmem_bootstrap_mpi.so

clean:
	rm -rf ./nvshmem_src_2.4.1-3.txz ./nvshmem_src_2.4.1-3 ./${DEST}
