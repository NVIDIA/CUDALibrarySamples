# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

all: check-vars matmul_example matmul_example_static

check-vars:
ifndef CUSPARSELT_PATH
	${error CUSPARSELT_PATH must be set}
endif
ifndef CUDA_TOOLKIT_PATH
	${error CUDA_TOOLKIT_PATH must be set}
endif

INCS := -I$(CUDA_TOOLKIT_PATH)/include -I${CUSPARSELT_PATH}/include
LIBS := -lcusparse -ldl
NVCC := ${CUDA_TOOLKIT_PATH}/bin/nvcc

matmul_example: matmul_example.cpp
	${NVCC} --std=c++14  ${INCS}  matmul_example.cpp -o matmul_example            \
		 -L${CUSPARSELT_PATH}/lib64  -lcusparseLt ${LIBS}

matmul_example_static: matmul_example.cpp
	${NVCC} --std=c++14 ${INCS} matmul_example.cpp -o matmul_example_static       \
         -L${CUSPARSELT_PATH}/lib64 -lcusparseLt_static ${LIBS}

test:
	@echo "\n==== cusparseLt Matmul Test ====\n"
	./matmul_example
	./matmul_example_static

clean:
	rm -f matmul_example matmul_example_static

.PHONY: clean all test