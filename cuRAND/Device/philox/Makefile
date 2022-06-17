CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include -I../../utils
LIBS         := -L$(CUDA_TOOLKIT)/lib64 -lcudart -lcurand
FLAGS        := -O3 -std=c++11

all: curand_philox_example

curand_philox_example: curand_philox_example.cpp
	nvcc $(FLAGS) $(INC) curand_philox_example.cpp -o curand_philox_example $(LIBS)

clean:
	rm -f curand_philox_example

test:
	@echo "\n==== PHILOX Test ====\n"
	./curand_philox_example

.PHONY: clean all test
