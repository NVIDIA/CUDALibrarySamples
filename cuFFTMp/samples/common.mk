# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

MPI_HOME    ?= /opt/nvidia/hpc_sdk/Linux_x86_64/${HPCSDK_VERSION}/comm_libs/mpi
NVSHMEM_LIB ?= /opt/nvidia/hpc_sdk/Linux_x86_64/${HPCSDK_VERSION}/comm_libs/nvshmem/lib
NVSHMEM_INC ?= /opt/nvidia/hpc_sdk/Linux_x86_64/${HPCSDK_VERSION}/comm_libs/nvshmem/include
NCCL_LIB    ?= /opt/nvidia/hpc_sdk/Linux_x86_64/${HPCSDK_VERSION}/comm_libs/nccl/lib
NCCL_INC    ?= /opt/nvidia/hpc_sdk/Linux_x86_64/${HPCSDK_VERSION}/comm_libs/nccl/include
CUDA_HOME   ?= $(shell dirname $$(command -v nvcc))/..
CUFFT_LIB   ?= ../../cufft/lib/
CUFFT_INC   ?= ../../cufft/include/
ARCH        ?= $(shell uname -m)
ifeq ($(ARCH), ppc64le)
MPI         ?= mpi_ibm
else
MPI         ?= mpi
endif
# Also see https://docs.nvidia.com/cuda/cufftmp/usage/requirements.html for supported architectures
CXXFLAGS = -std=c++17 --generate-code arch=compute_75,code=sm_75 --generate-code arch=compute_80,code=sm_80 --generate-code arch=compute_86,code=sm_86 --generate-code arch=compute_89,code=sm_89 --generate-code arch=compute_90,code=sm_90 --generate-code arch=compute_100,code=sm_100 --generate-code arch=compute_120,code=sm_120
INCFLAGS = -I${CUFFT_INC} -I${NVSHMEM_INC} -I${MPI_HOME}/include
LDFLAGS  = -lcuda -L${CUFFT_LIB} -L${NVSHMEM_LIB}  -lcufftMp -lnvshmem_device -l:libnvshmem_host.so.3 -L${MPI_HOME}/lib -l${MPI}
# The NCCL samples do not use NVSHMEM. -l:libnccl.so.2: the NCCL pip wheel has no libnccl.so link
NCCL_INCFLAGS = -I${CUFFT_INC} -I${NCCL_INC} -I${MPI_HOME}/include
NCCL_LDFLAGS  = -L${CUFFT_LIB} -L${NCCL_LIB} -lcufftMp -l:libnccl.so.2 -L${MPI_HOME}/lib -l${MPI}
