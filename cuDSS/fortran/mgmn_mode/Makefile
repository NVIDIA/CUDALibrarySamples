# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Standalone Makefile for the cuDSS Fortran MGMN example "mgmn_mode".
# Set the variables below to point at your cuDSS and CUDA installation.
# Run with: $(MPIRUN) -np $(MPI_NP) ./mgmn_mode   (needs a cuDSS comm layer).

EXAMPLE   := mgmn_mode

# Standard-Fortran examples: default to the generic Fortran compiler; set FC to
# your toolchain (e.g. gfortran, ifx, or nvfortran).
FC        ?= f77
MPIF90    ?= mpif90
MPICC     ?= mpicc
FFLAGS    ?= -O2 -fPIC
CFLAGS    ?= -O2 -fPIC
CUDSS_INC ?= cudss_inc
CUDSS_LIB ?= cudss_lib
CUDA_INC  ?= cuda_inc
CUDA_LIB  ?= cuda_lib
CUDSS_INTERFACE_DIR ?= $(CUDSS_INC)/fortran
MPIRUN            ?= mpirun
MPI_NP            ?= 2
CUDSS_COMM_LAYER  ?= $(CUDSS_LIB)/libcudss_commlayer_openmpi.so

# MPIF90 is a wrapper (e.g. mpif90), so detect the underlying compiler via -show;
# otherwise an nvfortran/ifx-backed MPI would fall into the GNU branch below.
MPIF90_DRIVER := $(notdir $(firstword $(shell $(MPIF90) -show 2>/dev/null)))
MPIF90_BASE := $(if $(MPIF90_DRIVER),$(MPIF90_DRIVER),$(notdir $(MPIF90)))
ifneq ($(filter nvfortran ifx ifort mpiifx mpiifort,$(MPIF90_BASE)),)
MODFLAG := -module .
LIBS    := $(CUDSS_LIB)/libcudss.so -cudalib=cublas -Wl,-rpath,$(CUDSS_LIB)
else
MODFLAG := -J.
LIBS    := -L$(CUDSS_LIB) -lcudss -L$(CUDA_LIB) -lcudart -lcublas \
           -Wl,-rpath,$(CUDSS_LIB) -Wl,-rpath,$(CUDA_LIB)
endif
FFLAGS  += -I.

$(EXAMPLE): cudss.o cudss_mpi.o cudss_fortran_mpi.o cudautils.o validation.o cudssutils.o $(EXAMPLE).o
	$(MPIF90) $(FFLAGS) $(LDFLAGS) $(MODFLAG) $^ $(LIBS) -o $@

cudss.o: $(CUDSS_INTERFACE_DIR)/cudss.f90
	$(MPIF90) $(FFLAGS) $(MODFLAG) -c $< -o $@
cudss_mpi.o: $(CUDSS_INTERFACE_DIR)/cudss_mpi.f90 cudss.o
	$(MPIF90) $(FFLAGS) $(MODFLAG) -c $< -o $@
cudss_fortran_mpi.o: $(CUDSS_INTERFACE_DIR)/cudss_fortran_mpi.c
	$(MPICC) $(CFLAGS) -I$(CUDSS_INC) -I$(CUDA_INC) -c $< -o $@
cudautils.o: cudautils.f90
	$(MPIF90) $(FFLAGS) $(MODFLAG) -c $< -o $@
validation.o: validation.f90
	$(MPIF90) $(FFLAGS) $(MODFLAG) -c $< -o $@
cudssutils.o: cudssutils.f90 cudss.o cudautils.o
	$(MPIF90) $(FFLAGS) $(MODFLAG) -c $< -o $@
$(EXAMPLE).o: $(EXAMPLE).f90 cudss.o cudss_mpi.o cudautils.o cudssutils.o
	$(MPIF90) $(FFLAGS) $(MODFLAG) -c $< -o $@

.PHONY: clean
clean:
	rm -f *.o *.mod $(EXAMPLE)
