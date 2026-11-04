# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
LIBS         := -lcusparse

all: spmat_convert_example

spmat_convert_example: spmat_convert_example.c
	nvcc $(INC) spmat_convert_example.c -o spmat_convert_example $(LIBS)

clean:
	rm -f spmat_convert_example

test:
	@echo "\n==== Sparse Matrix Convert (CSR to CSC) Test ====\n"
	./spmat_convert_example

.PHONY: clean all test
