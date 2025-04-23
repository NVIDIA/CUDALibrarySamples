NVSHMEM_LIB ?= ../../cufft/lib
CUFFT_LIB   ?= ../../cufft/lib
CUFFT_INC   ?= ../../cufft/include

f90   := mpif90

WRAPPERS_DIR = ../Fortran_wrappers_nvhpc
FLAGS  = -O3 -Mfree -fast -Mextend -Mpreprocess -Minform=warn
FLAGS += -I./ -I${WRAPPERS_DIR}/ -I${CUFFT_INC}/
# Add flags -gpu to build for specific architecture. E.g., -gpu=cc70,cc80,cc90
# See https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-user-guide/index.html#compute-capability
# Also see https://docs.nvidia.com/cuda/cufftmp/usage/requirements.html for supported architectures
FLAGS += -Minfo=accel -cuda -cudalib=cufftmp 
LINKER := -L$(HPCSDK_ROOT)/compilers/lib -lnvhpcwrapcufft -lnvhpcwrapcufftmp -L${NVSHMEM_LIB} -lnvshmem_host -lnvshmem_device -L${CUFFT_LIB} 

