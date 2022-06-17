CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include -I../../utils
LIBS         := -L$(CUDA_TOOLKIT)/lib64 -lcudart -lcurand
FLAGS        := -O3 -std=c++11

all: curand_xorwow_example

curand_xorwow_example: curand_xorwow_example.cpp
	nvcc $(FLAGS) $(INC) curand_xorwow_example.cpp -o curand_xorwow_example $(LIBS)

clean:
	rm -f curand_xorwow_example

test:
	@echo "\n==== XORWOW Test ====\n"
	./curand_xorwow_example

.PHONY: clean all test
