DEST ?= "myMPI"
MPICC ?= mpicc
NVSHMEM_VER ?= 2.5.0
NVSHMEM_VER_RC ?= 2.5.0-19

all: mpi_bootstrap

${DEST}/nvshmem_bootstrap_mpi.so:
	mkdir -p ${DEST}
	wget https://developer.download.nvidia.com/compute/redist/nvshmem/${NVSHMEM_VER}/source/nvshmem_src_${NVSHMEM_VER_RC}.txz
	tar Jxvf nvshmem_src_${NVSHMEM_VER_RC}.txz
	mpicc nvshmem_src_${NVSHMEM_VER_RC}/src/bootstrap/bootstrap_mpi.c -shared -o ${DEST}/nvshmem_bootstrap_mpi.so -fPIC -Invshmem_src_${NVSHMEM_VER_RC}/src/include/
	echo "Bootstrap built!"
	ldd ${DEST}/nvshmem_bootstrap_mpi.so
	rm -rf ./nvshmem_src_${NVSHMEM_VER_RC}.txz ./nvshmem_src_${NVSHMEM_VER_RC}

mpi_bootstrap: ${DEST}/nvshmem_bootstrap_mpi.so

clean:
	rm -rf ./nvshmem_src_${NVSHMEM_VER_RC}.txz ./nvshmem_src_${NVSHMEM_VER_RC} ./${DEST}
