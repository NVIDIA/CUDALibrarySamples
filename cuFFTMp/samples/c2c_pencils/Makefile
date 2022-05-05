include ../common.mk

exe = cufftmp_c2c_pencils

all: $(exe)

.PHONY: clean

clean: 
	rm -rf $(exe)

$(exe): $(exe).cu ../iterators/box_iterator.hpp
	${CUDA_HOME}/bin/nvcc $< -o $@ -lcuda -std=c++17 \
		-L${CUFFT_LIB} -I${CUFFT_INC} -lcufftMp \
		-I${MPI_HOME}/include -L${MPI_HOME}/lib -l${MPI}

build: $(exe)

run: $(exe)
	LD_LIBRARY_PATH="${NVSHMEM_LIB}:${CUFFT_LIB}:${LD_LIBRARY_PATH}" mpirun -oversubscribe -n 4 $(exe)
