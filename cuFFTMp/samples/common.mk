MPI_HOME    ?= /opt/nvidia/hpc_sdk/Linux_x86_64/${HPCSDK_VERSION}/comm_libs/mpi
NVSHMEM_LIB ?= /opt/nvidia/hpc_sdk/Linux_x86_64/${HPCSDK_VERSION}/comm_libs/nvshmem/lib
NVSHMEM_INC ?= /opt/nvidia/hpc_sdk/Linux_x86_64/${HPCSDK_VERSION}/comm_libs/nvshmem/include
CUDA_HOME   ?= $(shell dirname $$(command -v nvcc))/..
CUFFT_LIB   ?= ../../cufft/lib/
CUFFT_INC   ?= ../../cufft/include/
ARCH        ?= $(shell uname -m)
ifeq ($(ARCH), ppc64le)
MPI         ?= mpi_ibm
else
MPI         ?= mpi
endif
CXXFLAGS = -std=c++17 --generate-code arch=compute_70,code=sm_70 --generate-code arch=compute_80,code=sm_80 --generate-code arch=compute_90,code=sm_90
INCFLAGS = -I${CUFFT_INC} -I${NVSHMEM_INC} -I${MPI_HOME}/include
LDFLAGS  = -lcuda -L${CUFFT_LIB} -L${NVSHMEM_LIB}  -lcufftMp -lnvshmem_device -lnvshmem_host -L${MPI_HOME}/lib -l${MPI}
