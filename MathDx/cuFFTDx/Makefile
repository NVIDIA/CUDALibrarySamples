CXX?=g++
NVCC?=nvcc

CUDA_ARCHITECTURES?=70 75 80 86
NVCC_FLAGS=-std=c++17 -O3 $(foreach arch, $(CUDA_ARCHITECTURES), --generate-code arch=compute_$(arch),code=sm_$(arch))
NVCC_LTO_FLAGS=-std=c++17 -O3 $(foreach arch, $(CUDA_ARCHITECTURES), --generate-code arch=compute_$(arch),code=lto_$(arch))
CXX_FLAGS=-std=c++17 -O3

CUFFTDX_INCLUDE_DIR?=../../include
COMMONDX_INCLUDE_DIR?=../../include/commondx
CUFFT_HOME?=../../../cufft

CUDA_BIN_DIR=$(shell dirname `which $(NVCC)`)
CUDA_INCLUDE_DIR=$(CUDA_BIN_DIR)/../include
NVCC_DEFINES=-DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY
NVRTC_DEFINES=-DCUDA_INCLUDE_DIR='"$(CUDA_INCLUDE_DIR)"' -DCOMMONDX_INCLUDE_DIR='"$(COMMONDX_INCLUDE_DIR)"' -DCUFFTDX_INCLUDE_DIRS='"$(CUFFTDX_INCLUDE_DIR)"'
LTO_DEFINES=-DCUFFTDX_ENABLE_CUFFT_DEPENDENCY

# NVRTC LTO sources and targets
NVRTC_LTO_SRCS=$(wildcard **/nvrtc_*lto*.cu)
NVRTC_LTO_TARGETS=$(foreach src,$(NVRTC_LTO_SRCS),$(patsubst %.cu,%,$(basename $(src))))

# NVRTC sources and targets (excluding LTO)
NVRTC_SRCS=$(filter-out ${NVRTC_LTO_SRCS}, $(wildcard **/nvrtc_*.cu))
NVRTC_TARGETS=$(foreach src,$(NVRTC_SRCS),$(patsubst %.cu,%,$(basename $(src))))

# Standalone sources and targets
STANDALONE_SRCS=09_introduction_lto_example/introduction_lto_example.cu \
                10_cufft_device_api_example/cufft_device_api_example.cu
STANDALONE_TARGETS=$(foreach src,$(STANDALONE_SRCS),$(patsubst %.cu,%,$(basename $(src))))

# LTO sources and targets
LTO_SRCS=$(filter-out ${NVRTC_LTO_SRCS} ${STANDALONE_SRCS}, $(wildcard **/*lto*.cu))
LTO_TARGETS=$(foreach src,$(LTO_SRCS),$(patsubst %.cu,%,$(basename $(src))))

# Regular sources and targets (excluding all special cases)
SRCS=$(filter-out ${NVRTC_SRCS} ${NVRTC_LTO_SRCS} ${LTO_SRCS} ${STANDALONE_SRCS}, $(wildcard **/*.cu))
TARGETS=$(foreach src,$(SRCS),$(patsubst %.cu,%,$(basename $(src))))

TARGETS_TWIDDLES=$(patsubst %.cu,%_twd,$(SRCS))
ADDITIONAL_SRCS=../src/liblut/lut.cu
ADDITIONAL_DEFINES=-DCUFFTDX_USE_SEPARATE_TWIDDLES
EXTRA_FLAGS=-rdc=true

$(TARGETS): %: %.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS) -I$(COMMONDX_INCLUDE_DIR) -I$(CUFFTDX_INCLUDE_DIR) $(NVCC_DEFINES) -lcufft

$(NVRTC_TARGETS): %: %.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS) -I$(COMMONDX_INCLUDE_DIR) -I$(CUFFTDX_INCLUDE_DIR) $(NVCC_DEFINES) $(NVRTC_DEFINES) -lnvrtc -lcuda

$(NVRTC_LTO_TARGETS): %: %.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS) -I$(COMMONDX_INCLUDE_DIR) -I$(CUFFTDX_INCLUDE_DIR) -I$(CUFFT_HOME)/include -L$(CUFFT_HOME)/lib $(NVCC_DEFINES) $(NVRTC_DEFINES) ${LTO_DEFINES} -lnvrtc -lcuda -lcufft -lnvJitLink

$(TARGETS_TWIDDLES): %: $(SRCS)
	$(NVCC) $(ADDITIONAL_DEFINES) -o $@ $(ADDITIONAL_SRCS) $< $(NVCC_FLAGS) $(EXTRA_FLAGS) $(ADDITIONAL_DEFINES) -I$(COMMONDX_INCLUDE_DIR) -I$(CUFFTDX_INCLUDE_DIR) -lcufft

cufftdx_cufft_lto_helper: lto_helper/cufftdx_cufft_lto_helper.cpp
	$(CXX) -o $@ $< $(CXX_FLAGS) -I$(CUFFT_HOME)/include -I$(CUDA_INCLUDE_DIR) -L$(CUFFT_HOME)/lib -L$(CUDA_BIN_DIR)/../lib64 -lcufft_static -lculibos -ldl -lpthread

$(LTO_TARGETS:%=%_artifacts): %_artifacts: %_cases.csv cufftdx_cufft_lto_helper
	mkdir -p $(dir $*)$(subst /,_,$*)_artifacts
	./cufftdx_cufft_lto_helper $(dir $*)$(subst /,_,$*)_artifacts $<

$(LTO_TARGETS:%=%.o): %.o: %.cu %_artifacts
	$(NVCC) -o $@ $< -dc $(NVCC_LTO_FLAGS) -I$(COMMONDX_INCLUDE_DIR) -I$(CUFFTDX_INCLUDE_DIR) -I$(dir $*)$(subst /,_,$*)_artifacts/

$(LTO_TARGETS:%=%_dlink.o): %_dlink.o: %.o
	$(NVCC) -o $@ -dlink -dlto $(NVCC_FLAGS) $< $(wildcard $(dir $(basename $<))$(subst /,_,$(basename $<))_artifacts/*.fatbin) $(wildcard $(dir $(basename $<))$(subst /,_,$(basename $<))_artifacts/*.ltoir)

$(LTO_TARGETS): %: %.o %_dlink.o
	$(CXX) -o $@ $^ -L$(CUDA_BIN_DIR)/../lib64 -lcudart

$(STANDALONE_TARGETS): %: %.cu
	(cd $(dir $<) && $(MAKE))

.PHONY: all clean all_twiddles

all: $(TARGETS) $(NVRTC_TARGETS)

all_twiddles: $(TARGETS_TWIDDLES)

all_lto: $(LTO_TARGETS) $(NVRTC_LTO_TARGETS) $(STANDALONE_TARGETS)

clean:
	rm -f $(TARGETS) $(NVRTC_TARGETS) $(TARGETS_TWIDDLES)
	rm -f $(LTO_TARGETS) $(NVRTC_LTO_TARGETS) $(LTO_TARGETS:%=%.o) $(LTO_TARGETS:%=%_dlink.o) cufftdx_cufft_lto_helper
	rm -rf $(LTO_TARGETS:%=%_artifacts)
	$(foreach dir,$(STANDALONE_TARGETS),(cd $(dir $(dir)) && $(MAKE) clean);)

.DEFAULT_GOAL := all
