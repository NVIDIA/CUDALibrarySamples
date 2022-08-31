CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include -I../../utils
LIBS         := -L$(CUDA_TOOLKIT)/lib64 -lcudart -lcurand
FLAGS        := -O3 -std=c++11

ROUTINES	 := curand_scrambled_sobol64_uniform_example \
				curand_scrambled_sobol64_normal_example \
				curand_scrambled_sobol64_lognormal_example\
				curand_scrambled_sobol64_poisson_example

all: $(ROUTINES)

%: %.cpp
	nvcc $(FLAGS) $(INC) $^ -o $@ $(LIBS)

clean:
	rm -f $(ROUTINES)

.PHONY: clean all test
