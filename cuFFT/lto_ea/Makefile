###########################################################################################################
# User defines

CXX   ?= gcc
NVCC  ?= nvcc
BIN2C ?= bin2c

CUDA_PATH ?= /usr/local/cuda
CUDA_ARCH ?= 50

###########################################################################################################
# Paths

INCLUDES = $(CUDA_PATH)/include
LIBS = $(CUDA_PATH)/lib64

$(shell   mkdir -p ./bin)
$(shell   mkdir -p ./build)

###########################################################################################################
# Flags

GENCODE_FLAGS_LTO = --generate-code arch=compute_$(CUDA_ARCH),code=lto_$(CUDA_ARCH)
GENCODE_FLAGS     = --generate-code arch=compute_$(CUDA_ARCH),code=compute_$(CUDA_ARCH)

DEFINES = -DCUDA_ARCH=$(CUDA_ARCH) -DCUDA_PATH=$(CUDA_PATH) -DSOURCE_PATH=${CURDIR}/src

###########################################################################################################
# Rules

all: bin/r2c_c2r_lto_callback_example bin/r2c_c2r_lto_nvrtc_callback_example bin/r2c_c2r_callback_example

# LTO example
bin/r2c_c2r_lto_callback_example: build/r2c_c2r_lto_callback_example.o build/r2c_c2r_reference.o build/common.o
	$(CXX) -L $(LIBS) $^ -o $@ -lcufft -lcudart

build/r2c_c2r_lto_callback_example.o: src/r2c_c2r_lto_callback_example.cpp src/r2c_c2r_lto_callback_device_fatbin.h
	$(CXX) -I $(INCLUDES) -c $< -o $@

src/r2c_c2r_lto_callback_device_fatbin.h: build/r2c_c2r_lto_callback_device.fatbin
	$(BIN2C) --name window_callback --type longlong $< > $@

build/r2c_c2r_lto_callback_device.fatbin: src/r2c_c2r_lto_callback_device.cu
	$(NVCC) --std=c++11 $(GENCODE_FLAGS_LTO) -dc -fatbin $< -o $@

# NVRTC + LTO example
bin/r2c_c2r_lto_nvrtc_callback_example: build/r2c_c2r_lto_nvrtc_callback_example.o build/r2c_c2r_reference.o build/common.o
	$(CXX) -L $(LIBS) $^ -o $@ -lcufft -lnvrtc -lcudart

build/r2c_c2r_lto_nvrtc_callback_example.o: src/r2c_c2r_lto_nvrtc_callback_example.cpp
	$(CXX) -I $(INCLUDES) $(DEFINES) -c $< -o $@

# Regular callback example
# NOTE: we compile the sample with nvcc and do device-linking against the static library. With the
# LTO sample, on the other hand, we only need to do separate compilation of the callback function; 
# we can use the dynamic library and compile the host code with the host compiler
build/r2c_c2r_callback_example.o: src/r2c_c2r_callback_example.cu
	$(NVCC) -I $(INCLUDES) --std=c++11 $(GENCODE_FLAGS) -dc -c $< -o $@ 

bin/r2c_c2r_callback_example: build/r2c_c2r_callback_example.o build/r2c_c2r_reference.o build/common.o
	$(NVCC) -L $(LIBS) $(GENCODE_FLAGS) -o $@ $^ -lcufft_static -lcudart -lculibos

# Reference
build/r2c_c2r_reference.o: src/r2c_c2r_reference.cu
	$(NVCC) -I $(INCLUDES) --std=c++11 $(GENCODE_FLAGS) -c $< -o $@ 

build/common.o: src/common.cpp
	$(CXX) -I $(INCLUDES) --std=c++11 -c $< -o $@

clean:
	rm -f build/* bin/* src/r2c_c2r_lto_callback_device_fatbin.h
###########################################################################################################
