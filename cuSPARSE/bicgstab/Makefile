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
LIBS         := -lcudart -lcusparse -lcublas
ifeq ($(DEBUG), 1)
    DEBUG_FLAG := -g -Wno-unused-result
else
    DEBUG_FLAG := -DNDEBUG -O3 -Wno-unused-result
endif

all: bicgstab_example

bicgstab_example: bicgstab_example.c
	gcc $(DEBUG_FLAG) $(INC) bicgstab_example.c -o bicgstab_example $(LIBS)

clean:
	rm -f bicgstab_example

test:
	@echo "\n==== BiCGStab Test ====\n"
	./bicgstab_example

.PHONY: clean all test