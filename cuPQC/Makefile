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

NVCC=nvcc
NVCC_FLAGS=-dlto -arch=native -std=c++17 -O3 


CUPQC_DIR?=/usr/lib/nvidia/cupqc-pkg-0.4.0/
CUPQC_INCLUDE_DIR=$(CUPQC_DIR)/include/cupqc
COMMONDX_INCLUDE_DIR=$(CUPQC_DIR)/include/
#/commondx

CUDA_BIN_DIR=$(shell dirname `which $(NVCC)`)
CUDA_INCLUDE_DIR=$(CUDA_BIN_DIR)/../include

SRCS = $(wildcard *.cu)
TARGETS=$(patsubst %.cu,%,$(SRCS))

$(TARGETS): %: %.cu
	$(NVCC) $(NVCC_FLAGS) -L$(CUPQC_DIR)/lib/ -lcupqc -lcuhash -o $@ $<  -I$(COMMONDX_INCLUDE_DIR) -I$(CUPQC_INCLUDE_DIR) 
.PHONY: all clean

all: $(TARGETS) 

clean:
	rm -f $(TARGETS)

.DEFAULT_GOAL := all