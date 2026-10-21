# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Standalone Makefile for the cuDSS CUDA Fortran MGMN example "mgmn_mode" (nvfortran).
# Set the variables below to point at your cuDSS and CUDA installation.
# Run with: $(MPIRUN) -np $(MPI_NP) ./mgmn_mode   (needs a cuDSS comm layer).

EXAMPLE   := mgmn_mode

# GNU make predefines FC=f77 (origin "default"), which ?= would not override, so
# default it explicitly while still honoring an FC given on the command line or in
# the environment.
ifeq ($(origin FC),default)
FC := nvfortran
endif
MPIF90    ?= mpif90
MPICC     ?= mpicc
FFLAGS    ?= -O2 -cuda
CFLAGS    ?= -O2 -fPIC
CUDSS_INC ?= cudss_inc
CUDSS_LIB ?= cudss_lib
CUDA_INC  ?= cuda_inc
CUDSS_INTERFACE_DIR ?= $(CUDSS_INC)/fortran
MPIRUN            ?= mpirun
MPI_NP            ?= 2
CUDSS_COMM_LAYER  ?= $(CUDSS_LIB)/libcudss_commlayer_openmpi.so

MODFLAG := -module .
LIBS    := $(CUDSS_LIB)/libcudss.so -cudalib=cublas -Wl,-rpath,$(CUDSS_LIB)
FFLAGS  += -I.

$(EXAMPLE): cudss.o cudss_mpi.o cudss_fortran_mpi.o validation.o cudssutils.o $(EXAMPLE).o
	$(MPIF90) $(FFLAGS) $(LDFLAGS) $(MODFLAG) $^ $(LIBS) -o $@

cudss.o: $(CUDSS_INTERFACE_DIR)/cudss.f90
	$(MPIF90) $(FFLAGS) $(MODFLAG) -c $< -o $@
cudss_mpi.o: $(CUDSS_INTERFACE_DIR)/cudss_mpi.f90 cudss.o
	$(MPIF90) $(FFLAGS) $(MODFLAG) -c $< -o $@
cudss_fortran_mpi.o: $(CUDSS_INTERFACE_DIR)/cudss_fortran_mpi.c
	$(MPICC) $(CFLAGS) -I$(CUDSS_INC) -I$(CUDA_INC) -c $< -o $@
validation.o: validation.f90
	$(MPIF90) $(FFLAGS) $(MODFLAG) -c $< -o $@
cudssutils.o: cudssutils.cuf cudss.o
	$(MPIF90) $(FFLAGS) $(MODFLAG) -c $< -o $@
$(EXAMPLE).o: $(EXAMPLE).cuf cudss.o cudss_mpi.o cudssutils.o
	$(MPIF90) $(FFLAGS) $(MODFLAG) -c $< -o $@

.PHONY: clean
clean:
	rm -f *.o *.mod $(EXAMPLE)
