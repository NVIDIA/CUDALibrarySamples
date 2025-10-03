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

DEST ?= "myMPI"
MPICC ?= mpicc
NVSHMEM_VER ?= 2.8.0
NVSHMEM_VER_RC ?= 2.8.0-3

all: mpi_bootstrap

${DEST}/nvshmem_bootstrap_mpi.so:
	mkdir -p ${DEST}
	wget https://developer.download.nvidia.com/compute/redist/nvshmem/${NVSHMEM_VER}/source/nvshmem_src_${NVSHMEM_VER_RC}.txz
	tar Jxvf nvshmem_src_${NVSHMEM_VER_RC}.txz
	make -C nvshmem_src_${NVSHMEM_VER_RC} $(shell pwd)/nvshmem_src_${NVSHMEM_VER_RC}/build/lib/nvshmem_bootstrap_mpi.so.${NVSHMEM_VER}
	echo "Bootstrap built!"
	mv $(shell pwd)/nvshmem_src_${NVSHMEM_VER_RC}/build/lib/nvshmem_bootstrap_mpi* ${DEST}
	ldd ${DEST}/nvshmem_bootstrap_mpi.so
	rm -rf ./nvshmem_src_${NVSHMEM_VER_RC}.txz ./nvshmem_src_${NVSHMEM_VER_RC}

mpi_bootstrap: ${DEST}/nvshmem_bootstrap_mpi.so

clean:
	rm -rf ./nvshmem_src_${NVSHMEM_VER_RC}.txz ./nvshmem_src_${NVSHMEM_VER_RC} ./${DEST}