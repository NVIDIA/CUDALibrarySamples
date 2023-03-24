DEST ?= "myMPI"
MPICC ?= mpicc
NVSHMEM_VER ?= 2.8.0
NVSHMEM_VER_RC ?= 2.8.0-3

all: mpi_bootstrap

${DEST}/nvshmem_bootstrap_mpi.so:
	mkdir -p ${DEST}
	wget https://developer.download.nvidia.com/compute/redist/nvshmem/${NVSHMEM_VER}/source/nvshmem_src_${NVSHMEM_VER_RC}.txz
	tar Jxvf nvshmem_src_${NVSHMEM_VER_RC}.txz
	make -C nvshmem_src_${NVSHMEM_VER_RC} $(shell pwd)/nvshmem_src_${NVSHMEM_VER_RC}/build/lib/nvshmem_bootstrap_mpi.so.${NVSHMEM_VER}
	echo "Bootstrap built!"
	mv $(shell pwd)/nvshmem_src_${NVSHMEM_VER_RC}/build/lib/nvshmem_bootstrap_mpi* ${DEST}
	ldd ${DEST}/nvshmem_bootstrap_mpi.so
	rm -rf ./nvshmem_src_${NVSHMEM_VER_RC}.txz ./nvshmem_src_${NVSHMEM_VER_RC}

mpi_bootstrap: ${DEST}/nvshmem_bootstrap_mpi.so

clean:
	rm -rf ./nvshmem_src_${NVSHMEM_VER_RC}.txz ./nvshmem_src_${NVSHMEM_VER_RC} ./${DEST}
