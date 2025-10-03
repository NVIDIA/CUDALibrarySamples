# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
LIBS         := -lcusparse

all: sddmm_csr_batched_example

sddmm_csr_batched_example: sddmm_csr_batched_example.c
	nvcc $(INC) sddmm_csr_batched_example.c -o sddmm_csr_batched_example $(LIBS)

clean:
	rm -f sddmm_csr_batched_example

test:
	@echo "\n==== Batched SDDMM CSR Test ====\n"
	./sddmm_csr_batched_example

.PHONY: clean all test