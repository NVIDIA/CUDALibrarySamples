#******************************************************************************
# * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# *
# * NVIDIA CORPORATION and its licensors retain all intellectual property
# * and proprietary rights in and to this software, related documentation
# * and any modifications thereto.  Any use, reproduction, disclosure or
# * distribution of this software and related documentation without an express
# * license agreement from NVIDIA CORPORATION is strictly prohibited.
# ******************************************************************************

all: check-vars matmul_advanced_example matmul_advanced_example_static

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

matmul_advanced_example: matmul_advanced_example.cpp
	${NVCC} --std=c++14  ${INCS}  matmul_advanced_example.cpp                  \
         -o matmul_advanced_example                                            \
         -L${CUSPARSELT_PATH}/lib64 -lcusparseLt ${LIBS}

matmul_advanced_example_static: matmul_advanced_example.cpp
	${NVCC} --std=c++14 ${INCS} matmul_advanced_example.cpp                    \
         -o matmul_advanced_example_static                                     \
         -L${CUSPARSELT_PATH}/lib64 -lcusparseLt_static                        \
         ${LIBS}

test:
	@echo "\n==== cusparseLt Matmul Advanced Test ====\n"
	./matmul_advanced_example
	./matmul_advanced_example_static


clean:
	rm -f matmul_advanced_example matmul_advanced_example_static

.PHONY: clean all test
