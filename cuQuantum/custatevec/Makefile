#  
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  - Neither the name(s) of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

CUDA_TOOLKIT    := $(shell dirname $$(command -v nvcc))/..
CUSTATEVEC_ROOT ?= ${CUDA_TOOLKIT}

INC             := -I$(CUSTATEVEC_ROOT)/include
LIBS            := -L$(CUSTATEVEC_ROOT)/lib64 -lcudart -lcustatevec

CXX_FLAGS= -std=c++11 $(INC) $(LIBS)

all:
	nvcc gate_application.cu  -o gate_application  ${CXX_FLAGS}
	nvcc exponential_pauli.cu -o exponential_pauli ${CXX_FLAGS}
	nvcc expectation.cu       -o expectation       ${CXX_FLAGS}
	nvcc sampler.cu           -o sampler           ${CXX_FLAGS}
	nvcc measure_zbasis.cu    -o measure_zbasis    ${CXX_FLAGS}
	nvcc batch_measure.cu     -o batch_measure     ${CXX_FLAGS}

clean:
	rm -f gate_application \
		exponential_pauli \
		expectation \
		sampler \
		measure_zbasis \
		batch_measure
