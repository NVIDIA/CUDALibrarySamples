# SPDX-FileCopyrightText: Copyright (c) 1993-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
INC          := -I$(CUDA_TOOLKIT)/include -I../utils
LIBS         := -L$(CUDA_TOOLKIT)/lib64 -lcufft
FLAGS        := -O3 -std=c++11

all: 1d_r2c_c2r_example

1d_r2c_c2r_example: 1d_r2c_c2r_example.cpp
	nvcc -x cu $(FLAGS) $(INC) 1d_r2c_c2r_example.cpp -o 1d_r2c_c2r_example $(LIBS)

clean:
	rm -f 1d_r2c_c2r_example

.PHONY: clean all