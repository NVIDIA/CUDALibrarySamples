###########################################################################################################
# User defines

CXX   ?= gcc
NVCC  ?= nvcc
BIN2C ?= bin2c

CUDA_PATH ?= /usr/local/cuda
CUDA_ARCH ?= 75 80 86 90 100 120
CUDA_LTO_ARCH ?= 75

###########################################################################################################
# Paths

INCLUDES = -I$(CUDA_PATH)/include -I./build
LIBS = -L$(CUDA_PATH)/lib64

$(shell   mkdir -p ./bin)
$(shell   mkdir -p ./build)

###########################################################################################################
# Flags

GENCODE_FLAGS_LTO = --generate-code arch=compute_$(CUDA_LTO_ARCH),code=lto_$(CUDA_LTO_ARCH)
GENCODE_FLAGS := $(foreach arch, $(CUDA_ARCH), --generate-code arch=compute_$(arch),code=[compute_$(arch),sm_$(arch)])

DEFINES = -DCUDA_ARCH=$(CUDA_LTO_ARCH) -DCUDA_PATH=$(CUDA_PATH) -DSOURCE_PATH=${CURDIR}/src

# Uncomment to use constant memory version of the callback
# CB_SWITCH = -DCB_USE_CONSTANT_MEMORY

###########################################################################################################
# Rules

all: bin/r2c_c2r_lto_callback_example bin/r2c_c2r_lto_nvrtc_callback_example bin/r2c_c2r_legacy_callback_example

# LTO example
bin/r2c_c2r_lto_callback_example: build/r2c_c2r_lto_callback_example.o build/r2c_c2r_reference.o build/common.o
	$(CXX) $(LIBS) $^ -o $@ -lcufft -lcudart

build/r2c_c2r_lto_callback_example.o: src/r2c_c2r_lto_callback_example.cpp build/r2c_c2r_lto_callback_device_fatbin.h
	$(CXX) $(INCLUDES) -c $< -o $@

build/r2c_c2r_lto_callback_device_fatbin.h: build/r2c_c2r_lto_callback_device.fatbin
	$(BIN2C) --name window_callback --type longlong $< > $@

build/r2c_c2r_lto_callback_device.fatbin: src/r2c_c2r_lto_callback_device.cu
	$(NVCC) --std=c++11 $(GENCODE_FLAGS_LTO) $(CB_SWITCH) -dc -fatbin $< -o $@

# NVRTC + LTO example
bin/r2c_c2r_lto_nvrtc_callback_example: build/r2c_c2r_lto_nvrtc_callback_example.o build/r2c_c2r_reference.o build/common.o
	$(CXX) $(LIBS) $^ -o $@ -lcufft -lnvrtc -lcudart

build/r2c_c2r_lto_nvrtc_callback_example.o: src/r2c_c2r_lto_nvrtc_callback_example.cpp
	$(CXX) $(INCLUDES) $(DEFINES) $(CB_SWITCH) -c $< -o $@

# Regular callback example
# NOTE: we compile the sample with nvcc and do device-linking against the static library. With the
# LTO sample, on the other hand, we only need to do separate compilation of the callback function;
# we can use the dynamic library and compile the host code with the host compiler
build/r2c_c2r_legacy_callback_example.o: src/r2c_c2r_legacy_callback_example.cu
	$(NVCC) $(INCLUDES) --std=c++11 $(GENCODE_FLAGS) $(CB_SWITCH) -dc -c $< -o $@

bin/r2c_c2r_legacy_callback_example: build/r2c_c2r_legacy_callback_example.o build/r2c_c2r_reference.o build/common.o
	$(NVCC) $(LIBS) $(GENCODE_FLAGS) $(CB_SWITCH) -o $@ $^ -lcufft_static -lcudart -lculibos

# Reference
build/r2c_c2r_reference.o: src/r2c_c2r_reference.cu
	$(NVCC) $(INCLUDES) --std=c++11 $(GENCODE_FLAGS) -c $< -o $@

build/common.o: src/common.cpp
	$(CXX) $(INCLUDES) --std=c++11 -c $< -o $@

clean:
	rm -f build/* bin/*
###########################################################################################################
