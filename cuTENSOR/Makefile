NVCC_VERSION := $(shell nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/' 2>/dev/null || echo "")

ifndef NVCC_VERSION
    ifdef CUDA_VERSION_MAJOR
        NVCC_VERSION := $(CUDA_VERSION_MAJOR)
    endif
endif

ifdef NVCC_VERSION
    CUDA_VERSION_MAJOR := $(shell echo $(NVCC_VERSION) | sed 's/\..*//' 2>/dev/null || echo $(NVCC_VERSION) | awk -F. '{print $$1}' 2>/dev/null || echo "12")
else
    CUDA_VERSION_MAJOR := 12
endif

LIB_DIR := $(shell if [ -d "$(CUTENSOR_ROOT)/lib/$(CUDA_VERSION_MAJOR)" ]; then echo "/lib/$(CUDA_VERSION_MAJOR)"; else echo "/lib"; fi)
CXX_FLAGS=-std=c++17 -I${CUTENSOR_ROOT}/include -L${CUTENSOR_ROOT}/${LIB_DIR} -lcutensor -lcudart

all:
	nvcc einsum.cu -o einsum ${CXX_FLAGS}
	nvcc contraction.cu -o contraction ${CXX_FLAGS}
	nvcc contraction_jit.cu -o contraction_jit ${CXX_FLAGS}
	nvcc elementwise_binary.cu -o elementwise_binary ${CXX_FLAGS}
	nvcc elementwise_permute.cu -o elementwise_permute ${CXX_FLAGS}
	nvcc elementwise_trinary.cu -o elementwise_trinary ${CXX_FLAGS}
	nvcc reduction.cu -o reduction ${CXX_FLAGS}

run:
	./einsum
	./contraction
	./contraction_jit
	./elementwise_binary
	./elementwise_permute
	./elementwise_trinary
	./reduction

clean:
	rm -f contraction contraction_jit elementwise_binary elementwise_permute elementwise_trinary reduction einsum
