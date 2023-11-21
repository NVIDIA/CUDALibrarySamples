CXX_FLAGS=-std=c++11 -I${CUTENSOR_ROOT}/include -L${CUTENSOR_ROOT}/lib -lcutensor -lcudart

all:
	nvcc einsum.cu -o  einsum ${CXX_FLAGS}
	nvcc contraction.cu -o  contraction ${CXX_FLAGS}
	nvcc contraction_jit.cu -o  contraction_jit ${CXX_FLAGS}
	nvcc elementwise_binary.cu -o  elementwise_binary ${CXX_FLAGS}
	nvcc elementwise_permute.cu -o  elementwise_permute ${CXX_FLAGS}
	nvcc elementwise_trinary.cu -o  elementwise_trinary ${CXX_FLAGS}
	nvcc reduction.cu -o  reduction ${CXX_FLAGS}

run:
	./einsum
	./contraction
	./contraction_jit
	./elementwise_binary
	./elementwise_permute
	./elementwise_trinary
	./reduction

clean:
	rm -f contraction contraction_jit elementwise_binary elementwise_permute elementwise_trinary reduction
