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

include ../common.mk 

exe = cufftmp_r2c
exe_acc = cufftmp_r2c_acc

all : $(exe) $(exe_acc) 

$(exe) : $(exe).o   
	$(f90) -o $@ $+ $(FLAGS) $(LINKER)

$(exe_acc) : $(exe_acc).o   
	$(f90) -o $@ $+ $(FLAGS) $(LINKER) -acc 

%.o : %.f90
	$(f90) -c $< -o $@ $(FLAGS)

%_acc.o : %.f90
	$(f90) -c $< -o $@ $(FLAGS) -DACC -acc 

.PHONY: clean

clean:
	@rm -rf *.mod *.o  $(exe) $(exe_acc)

run: $(exe) $(exe_acc)
	LD_LIBRARY_PATH="${NVSHMEM_LIB}:${CUFFT_LIB}:${LD_LIBRARY_PATH}" mpirun -n 2 $(exe)
	LD_LIBRARY_PATH="${NVSHMEM_LIB}:${CUFFT_LIB}:${LD_LIBRARY_PATH}" mpirun -n 2 $(exe_acc) 