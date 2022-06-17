CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include -I../../utils
LIBS         := -L$(CUDA_TOOLKIT)/lib64 -lcudart -lcurand
FLAGS        := -O3 -std=c++11

ROUTINES	 := curand_xorwow_uniform_example \
				curand_xorwow_normal_example \
				curand_xorwow_lognormal_example\
				curand_xorwow_poisson_example

all: $(ROUTINES)

%: %.cpp
	nvcc $(FLAGS) $(INC) $^ -o $@ $(LIBS)

clean:
	rm -f $(ROUTINES)

.PHONY: clean all test
