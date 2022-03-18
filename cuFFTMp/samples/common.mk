MPI_HOME    ?= /opt/nvidia/hpc_sdk/Linux_x86_64/${HPCSDK_VERSION}/comm_libs/mpi
NVSHMEM_LIB ?= ../../cufft/lib
CUDA_HOME   ?= $(shell dirname $$(command -v nvcc))/..
CUFFT_LIB   ?= ../../cufft/lib/
CUFFT_INC   ?= ../../cufft/include/
ARCH        ?= $(shell uname -m)
ifeq ($(ARCH), ppc64le)
MPI         ?= mpi_ibm
else
MPI         ?= mpi
endif
