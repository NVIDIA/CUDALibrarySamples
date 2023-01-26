# 
# Copyright 2023 NVIDIA Corporation.  All rights reserved.
# 
# NOTICE TO LICENSEE:
# 
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
# 
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
# 
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
# 
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
# 
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
# 

# --------------------------------------------
#     Defines, includes and libraries
# --------------------------------------------

# cusolverMp depends on HPC-X inside HPC SDK distribution, so it's required to 
# initialize HPC-X using one of the initialization scripts 
# (i.e. $HPCSDKROOT/comm_libs/hpcx/latest/hpcx-init-ompi.sh)
# and load HPC-X environment with `hpcx_load` function
# This will make `mpicc` command available which will provide required MPI compile 
# and link flags for compiler
COMPILER?=nvc++ -cuda -gpu cuda11.8

# Includes and linker flags
LIBS = -cudalib=cublas,cusolver,nccl,cusolvermp

# --------------------------------------------
#     Define targets
# --------------------------------------------

all: mp_getrf_getrs mp_potrf_potrs mp_geqrf mp_syevd mp_gels

clean:
	rm -f mp_getrf_getrs mp_potrf_potrs mp_geqrf mp_syevd mp_gels

# --------------------------------------------
#     Build commands
# --------------------------------------------
mp_getrf_getrs: mp_getrf_getrs.cpp
	$(COMPILER) $(INCS) `mpicc --showme:compile` $+ -o $@ $(LIBS) `mpicc --showme:link`

mp_potrf_potrs: mp_potrf_potrs.cpp
	$(COMPILER) $(INCS) `mpicc --showme:compile` $+ -o $@ $(LIBS) `mpicc --showme:link`

mp_geqrf: mp_geqrf.cpp
	$(COMPILER) $(INCS) `mpicc --showme:compile` $+ -o $@ $(LIBS) `mpicc --showme:link`

mp_syevd: mp_syevd.cpp
	$(COMPILER) $(INCS) `mpicc --showme:compile` $+ -o $@ $(LIBS) `mpicc --showme:link`

mp_gels: mp_gels.cpp
	$(COMPILER) $(INCS) `mpicc --showme:compile` $+ -o $@ $(LIBS) `mpicc --showme:link`

.PHONY: all clean
