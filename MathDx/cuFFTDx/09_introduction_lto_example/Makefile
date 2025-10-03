# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CXX?=g++
NVCC?=nvcc

CUDA_ARCHITECTURES?=75 80 86
SM_ARCH_FLAGS = $(foreach arch, $(CUDA_ARCHITECTURES), --generate-code arch=compute_$(arch),code=sm_$(arch))
LTO_ARCH_FLAGS = $(foreach arch, $(CUDA_ARCHITECTURES), --generate-code arch=compute_$(arch),code=lto_$(arch))

CXX_FLAGS=-std=c++17 -O3

CUFFTDX_INCLUDE_DIR?=../../../include
COMMONDX_INCLUDE_DIR?=../../../include/commondx
CUFFT_HOME?=../../../../cufft
CUDA_PATH?=/usr/local/cuda

TARGET=introduction_lto_example

LTO_HELPER_TARGET=cufftdx_cufft_lto_helper
LTO_HELPER_DIR=../lto_helper
LTO_HELPER=$(LTO_HELPER_DIR)/$(LTO_HELPER_TARGET)
LTO_ARTIFACTS_DIR=./$(TARGET)_artifacts

CSV_FILE=introduction_lto_cases.csv

$(LTO_HELPER) : $(LTO_HELPER).cpp
	$(CXX) -o $@ $< $(CXX_FLAGS) -I$(CUFFT_HOME)/include -L$(CUFFT_HOME)/lib -I$(CUDA_PATH)/include -lcufft_static -L$(CUDA_PATH)/lib64 -lculibos -ldl -lpthread

run_lto_helper : $(LTO_HELPER)
	mkdir -p $(LTO_ARTIFACTS_DIR)
	./$(LTO_HELPER) $(LTO_ARTIFACTS_DIR) $(CSV_FILE)

%.o : %.cu run_lto_helper
	$(NVCC) -o $@ $< -dc $(CXX_FLAGS) $(LTO_ARCH_FLAGS) -I$(COMMONDX_INCLUDE_DIR) -I$(CUFFTDX_INCLUDE_DIR) -I$(LTO_ARTIFACTS_DIR)

%_dlink.o : %.o
	$(NVCC) -o $@ $< -dlto -dlink $(SM_ARCH_FLAGS) $(wildcard $(LTO_ARTIFACTS_DIR)/*.fatbin) $(wildcard $(LTO_ARTIFACTS_DIR)/*.ltoir)

% : %.o %_dlink.o
	$(CXX) -o $@ $^ -L$(CUDA_PATH)/lib64 -lcudart

clean:
	rm -rf $(TARGET) $(LTO_HELPER) $(LTO_ARTIFACTS_DIR)

.PHONY: all $(LTO_HELPER) run_lto_helper clean

# disable default/implicit make rule that attempts to link object files
.SUFFIXES:

all: $(TARGET)

.DEFAULT_GOAL := all