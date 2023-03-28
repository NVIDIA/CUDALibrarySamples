include ../common.mk

exe = cufftmp_reshape

all: $(exe)

.PHONY: clean

clean: 
	rm -rf $(exe)

$(exe): $(exe).cu
	${CUDA_HOME}/bin/nvcc $< -o $@ ${CXXFLAGS} ${INCFLAGS} ${LDFLAGS}

build: $(exe)

run: $(exe)
	LD_LIBRARY_PATH="${NVSHMEM_LIB}:${CUFFT_LIB}:${LD_LIBRARY_PATH}" mpirun -oversubscribe -n 2 $(exe)
