CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include -I../../utils
LIBS         := -L$(CUDA_TOOLKIT)/lib64 -lcudart -lcurand
FLAGS        := -O3 -std=c++11

ROUTINES	 := curand_mrg32k3a_uniform_example \
				curand_mrg32k3a_normal_example \
				curand_mrg32k3a_lognormal_example\
				curand_mrg32k3a_poisson_example

all: $(ROUTINES)

%: %.cpp
	nvcc $(FLAGS) $(INC) $^ -o $@ $(LIBS)

clean:
	rm -f $(ROUTINES)

.PHONY: clean all test
