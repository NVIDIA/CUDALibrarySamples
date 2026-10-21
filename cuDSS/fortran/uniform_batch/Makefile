# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Standalone Makefile for the cuDSS Fortran example "uniform_batch".
# Set the variables below to point at your cuDSS and CUDA installation.

EXAMPLE   := uniform_batch

# Standard-Fortran examples: default to the generic Fortran compiler; set FC to
# your toolchain (e.g. gfortran, ifx, or nvfortran).
FC        ?= f77
FFLAGS    ?= -O2 -fPIC
# cuDSS install: C headers in $(CUDSS_INC), Fortran interface sources in
# $(CUDSS_INTERFACE_DIR), libraries in $(CUDSS_LIB); CUDA libraries in $(CUDA_LIB).
CUDSS_INC ?= cudss_inc
CUDSS_LIB ?= cudss_lib
CUDA_INC  ?= cuda_inc
CUDA_LIB  ?= cuda_lib
CUDSS_INTERFACE_DIR ?= $(CUDSS_INC)/fortran

FC_BASE := $(notdir $(FC))
ifneq ($(filter nvfortran ifx ifort,$(FC_BASE)),)
MODFLAG := -module .
LIBS    := $(CUDSS_LIB)/libcudss.so -cudalib=cublas -Wl,-rpath,$(CUDSS_LIB)
else
MODFLAG := -J.
LIBS    := -L$(CUDSS_LIB) -lcudss -L$(CUDA_LIB) -lcudart -lcublas \
           -Wl,-rpath,$(CUDSS_LIB) -Wl,-rpath,$(CUDA_LIB)
endif
FFLAGS  += -I.

$(EXAMPLE): cudss.o cudautils.o validation.o cudssutils.o $(EXAMPLE).o
	$(FC) $(FFLAGS) $(LDFLAGS) $(MODFLAG) $^ $(LIBS) -o $@

cudss.o: $(CUDSS_INTERFACE_DIR)/cudss.f90
	$(FC) $(FFLAGS) $(MODFLAG) -c $< -o $@
cudautils.o: cudautils.f90
	$(FC) $(FFLAGS) $(MODFLAG) -c $< -o $@
validation.o: validation.f90
	$(FC) $(FFLAGS) $(MODFLAG) -c $< -o $@
cudssutils.o: cudssutils.f90 cudss.o cudautils.o
	$(FC) $(FFLAGS) $(MODFLAG) -c $< -o $@
$(EXAMPLE).o: $(EXAMPLE).f90 cudss.o cudautils.o cudssutils.o validation.o
	$(FC) $(FFLAGS) $(MODFLAG) -c $< -o $@

.PHONY: clean
clean:
	rm -f *.o *.mod $(EXAMPLE)
