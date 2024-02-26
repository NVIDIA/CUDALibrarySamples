NVCC=nvcc
NVCC_FLAGS=-std=c++17 -O3 --generate-code arch=compute_70,code=sm_70 --generate-code arch=compute_75,code=sm_75 --generate-code arch=compute_80,code=sm_80 --generate-code arch=compute_86,code=sm_86

CUFFTDX_INCLUDE_DIR=../include/
COMMONDX_INCLUDE_DIR=../../commondx/include/
CUDA_BIN_DIR=$(shell dirname `which $(NVCC)`)
CUDA_INCLUDE_DIR=$(CUDA_BIN_DIR)/../include
NVRTC_DEFINES=-DCUDA_INCLUDE_DIR="\"$(CUDA_INCLUDE_DIR)\"" -DCOMMONDX_INCLUDE_DIR="\"$(COMMONDX_INCLUDE_DIR)\"" -DCUFFTDX_INCLUDE_DIRS="\"$(CUFFTDX_INCLUDE_DIR)\""

SRCS=$(filter-out nvrtc_fft_block.cu nvrtc_fft_thread.cu, $(wildcard *.cu))
TARGETS=$(patsubst %.cu,%,$(SRCS))

NVRTC_SRCS=$(wildcard nvrtc_*.cu)
NVRTC_TARGETS=$(patsubst %.cu,%,$(NVRTC_SRCS))

$(TARGETS): %: %.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS) -I$(COMMONDX_INCLUDE_DIR) -I$(CUFFTDX_INCLUDE_DIR) -lcufft

$(NVRTC_TARGETS): %: %.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS) -I$(COMMONDX_INCLUDE_DIR) -I$(CUFFTDX_INCLUDE_DIR) $(NVRTC_DEFINES) -lnvrtc -lcuda

.PHONY: all clean

all: $(TARGETS) $(NVRTC_TARGETS) $(CUFFT_TARGETS)
	$(echo $(NVRTC_TARGETS))

clean:
	rm -f $(TARGETS) $(NVRTC_TARGETS) $(CUFFT_TARGETS)

.DEFAULT_GOAL := all
