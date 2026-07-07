# SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#

ifndef CUTENSOR_ROOT
    $(error CUTENSOR_ROOT is not set. Please point it at your cuTENSOR installation, e.g. `make CUTENSOR_ROOT=/path/to/cutensor`.)
endif

NVCC_VERSION := $(shell nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/' 2>/dev/null || echo "")

ifndef NVCC_VERSION
    ifdef CUDA_VERSION_MAJOR
        NVCC_VERSION := $(CUDA_VERSION_MAJOR)
    endif
endif

ifdef NVCC_VERSION
    CUDA_VERSION_MAJOR := $(shell echo $(NVCC_VERSION) | sed 's/\..*//' 2>/dev/null || echo $(NVCC_VERSION) | awk -F. '{print $$1}' 2>/dev/null || echo "12")
else
    CUDA_VERSION_MAJOR := 12
endif

LIB_DIR := $(shell if [ -d "$(CUTENSOR_ROOT)/lib/$(CUDA_VERSION_MAJOR)" ]; then echo "lib/$(CUDA_VERSION_MAJOR)"; else echo "lib"; fi)
CXX_FLAGS=-std=c++17 -O2 -I${CUTENSOR_ROOT}/include -L${CUTENSOR_ROOT}/${LIB_DIR} -lcutensor -lcudart -lcublasLt

OBJECTS = einsum contraction contraction_jit contraction_plan_cache \
          contraction_trinary elementwise_binary elementwise_permute \
          elementwise_permute_padding elementwise_trinary reduction \
          blocksparse

%: %.cu utils.cuh
	nvcc $< -o $@ ${CXX_FLAGS}

.PHONY: all
all: $(OBJECTS)

.PHONY: run
run:
	./einsum
	./contraction
	./contraction_jit
	./contraction_plan_cache
	./contraction_trinary
	./elementwise_binary
	./elementwise_permute
	./elementwise_permute_padding
	./elementwise_trinary
	./reduction
	./blocksparse

.PHONY: clean
clean:
	rm -f $(OBJECTS)

