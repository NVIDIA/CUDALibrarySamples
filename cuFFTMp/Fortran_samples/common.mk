NVSHMEM_LIB ?= ../../cufft/lib
CUFFT_LIB   ?= ../../cufft/lib
CUFFT_INC   ?= ../../cufft/include

f90   := mpif90

WRAPPERS_DIR = ../Fortran_wrappers_nvhpc
FLAGS  = -O3 -Mfree -fast -Mextend -Mpreprocess -Minform=warn
FLAGS += -I./ -I${WRAPPERS_DIR}/ -I${CUFFT_INC}/
FLAGS += -Minfo=accel -cuda -gpu=cc70,cc80,cc90 -cudalib=cufftmp 
LINKER := -L$(HPCSDK_ROOT)/compilers/lib -lnvhpcwrapcufft -lnvhpcwrapcufftmp -L${NVSHMEM_LIB} -lnvshmem_host -lnvshmem_device -L${CUFFT_LIB} 

