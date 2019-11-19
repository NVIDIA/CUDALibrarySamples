CXX_FLAGS=-std=c++11 -I../include -L../lib -lcutensor -lcudart

all:
	nvcc contraction.cu -o  contraction ${CXX_FLAGS}
	nvcc contraction_simple.cu -o  contraction_simple ${CXX_FLAGS}
	nvcc contraction_autotuning.cu -o  contraction_autotuning ${CXX_FLAGS}
	nvcc elementwise_binary.cu -o  elementwise_binary ${CXX_FLAGS}
	nvcc elementwise_permute.cu -o  elementwise_permute ${CXX_FLAGS}
	nvcc elementwise_trinary.cu -o  elementwise_trinary ${CXX_FLAGS}
	nvcc reduction.cu -o  reduction ${CXX_FLAGS}

clean:
	rm -f contraction contraction_simple contraction_autotuning elementwise_binary elementwise_permute elementwise_trinary reduction
