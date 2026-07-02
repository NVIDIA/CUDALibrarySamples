# SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

